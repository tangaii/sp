"""The project TADPOLE model with controlled changes to its PEN and GFR."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from CLIP.clip import clip
from layers import EncodeLayer, FeedForwardLayer, GraphConv, OutputLayer, VariLengthInputLayer
from utils import my_weight_init, normalize_adj


CLIP_WEIGHTS = Path.home() / '.cache/clip/RN50.pt'
_text_cache = {}


def encode_prompts(prompts, device, xavier=False):
    clip_model, _ = clip.load(str(CLIP_WEIGHTS), device)
    if xavier:
        # Reproduce the unrestricted Linear visitor on the released RN50.
        clip_model.apply(my_weight_init)
    clip_model.eval().requires_grad_(False)
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        return clip_model.encode_text(tokens).float().detach()


def prompt_features(prompts, device, init):
    key = (tuple(prompts), device)
    if key not in _text_cache:
        _text_cache[key] = encode_prompts(prompts, device)
    if init == 'frozen_pretrained':
        return _text_cache[key]
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        return encode_prompts(prompts, device, xavier=True)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.t1 = nn.Linear(input_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return F.relu(self.t2(F.relu(self.t1(x))))


class PLN(nn.Module):
    def __init__(self, input_data_dims, hyperpm, prompts, device):
        super().__init__()
        self.modal_num = len(input_data_dims)
        self.modal_width = hyperpm.n_hidden * hyperpm.n_head
        self.n_parts = 4
        self.part_width, remainder = divmod(self.modal_width, self.n_parts)
        if self.part_width == 0 or remainder:
            raise ValueError('The modality width must be divisible by four')
        self.pen = hyperpm.pen
        self.semantic_null = bool(getattr(hyperpm, 'semantic_null', False))
        self.calibration_ratio = getattr(hyperpm, 'calibration_ratio', None)
        self.calibration_eps = float(getattr(hyperpm, 'calibration_eps', 1e-12))
        self.fusion_weight = nn.Parameter(torch.ones([self.modal_num, 1]))
        self.InputLayer = VariLengthInputLayer(
            input_data_dims, hyperpm.n_hidden, hyperpm.n_hidden, hyperpm.n_head, hyperpm.dropout,
        )
        self.Encoder = []
        self.FeedForward = []
        for index in range(hyperpm.nlayer):
            encoder = EncodeLayer(
                self.modal_width, hyperpm.n_hidden, hyperpm.n_hidden, hyperpm.n_head, hyperpm.dropout,
            )
            encoder.attention.dropout.p = hyperpm.attention_dropout
            self.add_module(f'encode_{index}', encoder)
            self.Encoder.append(encoder)
            feedforward = FeedForwardLayer(self.modal_width, self.modal_width, dropout=hyperpm.dropout)
            self.add_module(f'feed_{index}', feedforward)
            self.FeedForward.append(feedforward)

        prediction_width = self.part_width if self.pen == 'equations_5_6' else self.modal_width
        self.Outputlayer = OutputLayer(
            prediction_width * self.modal_num, self.modal_width,
            hyperpm.nclass, self.modal_num, hyperpm.dropout,
        )
        if self.pen in ('public', 'equations_5_6', 'res_only'):
            self.avg_pool = nn.AvgPool1d(kernel_size=self.n_parts, stride=self.n_parts)
        if self.pen in ('public', 'equations_5_6'):
            texts_per_group = len(next(iter(prompts.values())))
            texts = [text for group in prompts.values() for text in group]
            features = prompt_features(texts, device, hyperpm.clip_init)
            self.register_buffer('prompt_text_features', features.reshape(len(prompts), texts_per_group, -1))
            prompt_width = features.size(-1) * texts_per_group
            self.MLP = MLP(prompt_width, 128, self.part_width)

    def forward(self, x):
        x, _ = self.InputLayer(x)
        for encoder, feedforward in zip(self.Encoder, self.FeedForward):
            x, _ = encoder(q=x, k=x, v=x, modal_num=self.modal_num)
            x = feedforward(x)

        x_before_residual = x
        grouped_text = None
        if self.pen == 'public':
            grouped_text = self.MLP(self.prompt_text_features.flatten(1))
            if getattr(self, 'semantic_null', False):
                grouped_text = grouped_text.mean(dim=0, keepdim=True).expand_as(grouped_text)
            x_res = self.legacy_recalibrate(x, grouped_text)
        elif self.pen == 'res_only':
            x_res = self.legacy_recalibrate(x, None)
        else:
            x_res = x
        delta = x_res - x_before_residual
        if self.calibration_ratio is None:
            x = x_res
            scale = x.new_ones(x.size(0))
        else:
            input_norm = x_before_residual.flatten(1).norm(dim=1)
            delta_norm = delta.flatten(1).norm(dim=1)
            scale = (float(self.calibration_ratio) * input_norm /
                     (delta_norm + self.calibration_eps)).detach()
            x = x_before_residual + scale[:, None, None] * delta
        self.last_residual = delta.detach()
        self.last_input_norm = x_before_residual.detach().flatten(1).norm(dim=1)
        self.last_delta_norm = delta.detach().flatten(1).norm(dim=1)
        self.last_output_delta_norm = (x - x_before_residual).detach().flatten(1).norm(dim=1)
        self.last_scale = scale.detach()
        self.last_grouped_text = None if grouped_text is None else grouped_text
        output, _ = self.Outputlayer(x.flatten(1), None)
        return output

    def legacy_recalibrate(self, x, grouped_text):
        batch_size = x.size(0)
        parts = x.view(batch_size, self.modal_num, self.n_parts, self.part_width)
        if grouped_text is None:
            part_strength = parts.new_ones(batch_size, self.modal_num, self.n_parts)
        else:
            normalized_parts = F.normalize(parts, p=2, dim=-1).unsqueeze(-2)
            normalized_text = F.normalize(grouped_text, p=2, dim=-1)[None, None, None]
            similarity = torch.matmul(normalized_parts, normalized_text.transpose(-1, -2)).squeeze(-2)
            part_strength = torch.softmax(similarity.clamp(min=-5, max=5), dim=-1).max(dim=-1).values
        parts = part_strength.unsqueeze(-1) * parts
        global_feat = self.avg_pool(x.view(-1, 1, self.modal_width))
        global_feat = global_feat.view(batch_size, self.modal_num, self.part_width)
        weighted_parts = [F.softmax(part * global_feat, dim=-1) * part for part in parts.unbind(2)]
        return x + torch.cat(weighted_parts, dim=2)

    def recalibrate(self, x, grouped_text):
        # Eq. 5 selects one class similarity for each projected local part.
        local = torch.stack([
            projection(x[:, index]) for index, projection in enumerate(self.local_projection)
        ], dim=1).unflatten(-1, (self.n_parts, self.part_width))
        global_feat = torch.stack([
            projection(x[:, index]) for index, projection in enumerate(self.global_projection)
        ], dim=1)
        similarities = F.normalize(local, dim=-1) @ F.normalize(grouped_text, dim=-1).T
        local = similarities.max(dim=-1).values.unsqueeze(-1) * local
        # Eq. 6 normalizes over local parts and returns one vector per modality.
        part_scores = (local * global_feat.unsqueeze(-2)).sum(dim=-1)
        weights = F.softmax(part_scores, dim=-1)
        return global_feat + (weights.unsqueeze(-1) * local).sum(dim=-2)


class GraphLearn(nn.Module):
    def __init__(self, input_dim, th, mode='fixed', quantile=0.97, temperature=0.1):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = th
        self.mode = mode
        self.quantile = quantile
        self.temperature = temperature

    def forward(self, x, train_index=None):
        x_norm = F.normalize(self.p(x), dim=-1)
        score = torch.matmul(x_norm, x_norm.T)
        threshold = score.new_tensor(self.th)
        if self.mode in ('adaptive', 'adaptive_straight_through'):
            if train_index is None:
                raise ValueError('Adaptive graph construction requires train indices')
            train_score = score[train_index[:, None], train_index[None, :]]
            offdiag = ~torch.eye(len(train_index), dtype=torch.bool, device=score.device)
            threshold = torch.quantile(train_score[offdiag].detach(), self.quantile)
        self.last_threshold = threshold.detach()
        hard_mask = (score > threshold).float()
        if self.mode in ('straight_through', 'adaptive_straight_through'):
            soft_mask = torch.sigmoid((score - threshold.detach()) / self.temperature)
            mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        else:
            mask = hard_mask.detach()
        return score * mask, score


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gcn1 = GraphConv(input_dim, output_dim)
        self.gcn2 = GraphConv(input_dim, output_dim)

    def forward(self, x, adj):
        return self.gcn2(F.relu(self.gcn1(x, adj)), adj)


class GFR(nn.Module):
    def __init__(self, input_dims, threshold, fusion, alpha, graph_mode='fixed', quantile=0.97, temperature=0.1, biomarker_mode='normal'):
        super().__init__()
        self.fusion = fusion
        self.alpha = alpha
        self.modalities = len(input_dims)
        self.graph_mode = graph_mode
        self.biomarker_mode = biomarker_mode
        input_width = sum(input_dims)
        self.fusion_weight = nn.Parameter(torch.ones([self.modalities, 1]))
        self.combined_weight = nn.Parameter(torch.ones([self.modalities, 1]))
        for index, dim in enumerate(input_dims):
            module_mode = 'adaptive' if biomarker_mode == 'quantile' and index == 2 else graph_mode
            self.add_module(f'GraphConstruct{index}', GraphLearn(dim, threshold, module_mode, quantile, temperature))
            self.add_module(f'MessagePassing{index}', GCN(input_width, input_width))

    def forward(self, raw, modalities, eye, train_index=None):
        weights = self.fusion_weight
        weights = weights.softmax(dim=0) if self.fusion == 'softmax_sum' else weights.sigmoid()
        residual = self.alpha * raw
        fused = residual
        graphs = []
        propagation_graphs = []
        scores = []
        message_norms = []
        for index, modal in enumerate(modalities):
            adj, score = getattr(self, f'GraphConstruct{index}')(modal, train_index)
            graphs.append(adj)
            propagation_graph = torch.zeros_like(adj) if index == 2 and self.biomarker_mode == 'self_only' else adj
            propagation_graphs.append(propagation_graph)
            normalized = normalize_adj(propagation_graph + eye)
            message = getattr(self, f'MessagePassing{index}')(raw, normalized)
            message_norms.append(float(torch.linalg.vector_norm(message.detach()).cpu()))
            if self.fusion == 'legacy' and index < 2:
                fused = weights[index] * message + residual
            else:
                fused = weights[index] * message + fused
            scores.append(score)
        self.last_propagation_graphs = propagation_graphs
        self.last_message_norms = message_norms
        self.last_fusion_weights = weights.detach().flatten().cpu().tolist()
        return fused, graphs, scores
