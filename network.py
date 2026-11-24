import argparse
import os
import pickle
import random
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from layers import *
from utils import *
from CLIP.clip import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
attr_words = [
                "Nromal brain volume","Uniform cortical thickness", "Symmetrical brain structure", "Normal ventricular morphology",
                "Altered brain volume", "Abnormal functional connectivity", "White matter integrity issues","Different activation patterns"
                ]

text = clip.tokenize(attr_words).to(device)
_prompt_text_features = None


def get_prompt_text_features():
    """Load and encode the fixed semantic prompts once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad():
            _prompt_text_features = clip_model.encode_text(text).float().detach()
    return _prompt_text_features

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.t1 = nn.Linear(input_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.t1(x))
        x = F.relu(self.t2(x))
        return x

class PLN(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(PLN, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        # self.input_data_dims = [x * 2 for x in self.input_data_dims]
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.modal_width = self.d_v * self.n_head
        self.n_parts = 4
        self.part_width, remainder = divmod(self.modal_width, self.n_parts)
        if self.part_width == 0 or remainder:
            raise ValueError("PEN requires the modality embedding width to be divisible by four")
        self.d_out = self.modal_width * self.modal_num
        self.fusion_weight = nn.Parameter(torch.ones([6, 1]))
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            #encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)
            
            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout = self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        self.Outputlayer = OutputLayer(self.d_out, self.modal_width, self.n_class, self.modal_num, self.dropout)
        self.avg_pool = nn.AvgPool1d(kernel_size=self.n_parts, stride=self.n_parts)
        self.ViT_model = None
        self.register_buffer('prompt_text_features', None)
        if getattr(hyperpm, 'datname', None) == 'TADPOLE':
            self.prompt_text_features = get_prompt_text_features()
        else:
            self.ViT_model, _ = clip.load('RN50', device)
        self.MLP = MLP(input_dim=4096, hidden_dim=128, output_dim=self.part_width)
        
    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        combine_part = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())
        text_features = self.prompt_text_features
        if self.ViT_model is not None:
            with torch.no_grad():
                text_features = self.ViT_model.encode_text(text).float()
        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        text_features = self.MLP(text_features)
        
        
        part_x = x.view(bs, self.modal_num, self.n_parts, self.part_width)
        part_x_norm = F.normalize(part_x, p=2, dim=-1)
        text_features_norm = F.normalize(text_features, p=2, dim=-1)

        text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        part_x_norm_expanded = part_x_norm.unsqueeze(-2)
        similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))
        similarity_matrix = similarity_matrix.squeeze(-2)

        # 进行数值截断以避免数值问题
        similarity_matrix = similarity_matrix.clamp(min=-5, max=5)

        # 计算 softmax
        similarity_matrix = torch.softmax(similarity_matrix, dim=-1).to(device)

        # 获取 similarity_matrix 的最大值
        similarity_max, _ = torch.max(similarity_matrix, dim=-1)

        # 进行最终的计算
        part_x = similarity_max.unsqueeze(-1) * part_x
        # 对最后一个维度进行平均池化
        
        # Pool the modality embedding to the width of a local part.
        reshaped_x = x.view(-1, 1, self.modal_width)
        global_x = self.avg_pool(reshaped_x)
        m = part_x.shape[2]

        global_x = global_x.view(bs, self.modal_num, self.part_width)
        for i in range(m):
            part = part_x[:,:,i,:]
            Vsim = F.softmax(part*global_x, dim=-1) 
            part_embedding = Vsim*part
            combine_part.append(part_embedding)
        x = x + torch.cat(combine_part,dim=2)
        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1) 
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map
  
class GraphLearn(nn.Module):
    def __init__(self, input_dim, th, mode = 'Sigmoid-like'):
        super(GraphLearn, self).__init__()
        self.mode = mode
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = th    
    def forward(self, x):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)
        th = self.th
        x = self.p(x)
        x_norm = F.normalize(x,dim=-1)
        score = torch.matmul(x_norm, x_norm.T)
        mask = (score > th).detach().float()
        markoff_value = 0
        output = score * mask + markoff_value * (1 - mask)
        return output  
    
    
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(input_dim, output_dim)
        self.gcn2 = GraphConv(input_dim, output_dim)
        
    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        return x
    


# 获取权重矩阵

def adj_normalization(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

    
class GFR(nn.Module):
    def __init__(self, input_dim , output_dim ,th , mode='Sigmoid-like', label_drop = 0.,proj_drop =0.):
        super(GFR, self).__init__()
        self.th = th
        self.mode = mode
        # breakpoint()
        
        input_sum = sum(input_dim) 
    
        self.fusion_weight = nn.Parameter(torch.ones([len(input_dim),1]))
        self.combined_weight = nn.Parameter(torch.ones([len(input_dim),1]))
        
        for i, dim in enumerate(input_dim):
            self.add_module(f'GraphConstruct{i}', GraphLearn(input_dim=dim, th=th, mode=mode))
            self.add_module(f'MessagePassing{i}', GCN(input_dim=input_sum, output_dim=input_sum))

        self.proj_drop = nn.Dropout(proj_drop)
        self.label_drop = nn.Dropout(label_drop)


    def forward(self, x, model_num, label, index):
        all_adj = []
        new_label = label.clone()
        new_label[new_label == 0] = -1
        train_mask = torch.zeros_like(new_label, dtype=torch.bool)
        train_mask[torch.as_tensor(index, device=label.device, dtype=torch.long)] = True
        new_label = torch.where(train_mask, new_label, torch.zeros_like(new_label))
        adj1 = new_label.unsqueeze(1).eq(new_label.unsqueeze(0))
        eye = torch.eye(adj1.size(0), device=label.device, dtype=x[0].dtype)
        A = normalize_adj(adj1.to(dtype=x[0].dtype) + eye)
        raw = torch.cat(x, dim=1)
        combined_feat = raw
        ALoss = raw.new_zeros(())
 
        for i in range(model_num):  
            f_w = self.fusion_weight[i]
            f_w = torch.sigmoid(f_w)
            adj = getattr(self, f'GraphConstruct{i}')(x[i])
            normalized_adj = normalize_adj(adj + eye)
            ALoss = ALoss + torch.linalg.norm(normalized_adj - A)
            message = getattr(self, f'MessagePassing{i}')(raw, normalized_adj)
            if i < 2:
                # Preserve the overwrite present in the released four-modal
                # implementation for the first two relations.
                combined_feat = f_w * message + raw
            else:
                combined_feat = f_w * message + combined_feat
            all_adj.append(adj)
                   
        return combined_feat , all_adj, ALoss/model_num
    
    
        
