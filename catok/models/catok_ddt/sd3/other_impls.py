"""Small attention helpers used by the action-only MMDiT implementation."""

import math
import os

import torch
from torch import nn


DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")
if DEVICE_TYPE in {"ascend", "npu"}:
    import torch_npu


def attention(q, k, v, heads, mask=None):
    """Apply scaled dot-product attention to packed token tensors."""
    batch_size, _, packed_dim = q.shape
    dim_head = packed_dim // heads
    q, k, v = map(
        lambda tensor: tensor.view(batch_size, -1, heads, dim_head).transpose(1, 2),
        (q, k, v),
    )

    if DEVICE_TYPE == "gpu":
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        return out.transpose(1, 2).reshape(batch_size, -1, heads * dim_head)

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        heads,
        input_layout="BNSD",
        pse=None,
        atten_mask=mask if mask is None else torch.logical_not(mask),
        scale=1.0 / math.sqrt(q.shape[-1]),
        pre_tockens=65536,
        next_tockens=65536,
        keep_prob=1.0,
        sync=False,
        inner_precise=0,
    )[0]
    return out.transpose(1, 2).contiguous().reshape(batch_size, -1, heads * dim_head)


class Mlp(nn.Module):
    """Vision Transformer style two-layer MLP."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
