"""Stable content SHA for model, checkpoint and RNG state trees."""
from hashlib import sha256
import numpy as np
import torch


def content_sha(value):
    digest = sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor {tensor.dtype} {tuple(tensor.shape)}\n".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"array {item.dtype} {item.shape}\n".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(f"dict {len(item)}\n".encode())
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__} {len(item)}\n".encode())
            for child in item:
                visit(child)
        else:
            digest.update(f"{type(item).__name__} {item!r}\n".encode())

    visit(value)
    return digest.hexdigest()
