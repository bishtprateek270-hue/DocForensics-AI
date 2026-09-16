"""
DocForensics AI — SegFormer Architecture (Phase 6)
Transformer-based semantic segmentation with Mix Transformer (MiT) hierarchical encoder
and All-MLP lightweight decoder for multi-scale document tampering detection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping convolutions to preserve local continuity."""

    def __init__(self, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Efficient Multi-Head Self-Attention with Spatial Reduction Ratio (SR)."""

    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    """Mix-FeedForward Network with 3x3 depthwise convolution (implicit positional encoding)."""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class MixTransformerEncoder(nn.Module):
    """Hierarchical Mix Transformer Encoder (MiT-B0 / B1)."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: List[int] = [32, 64, 160, 256],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[float] = [4.0, 4.0, 4.0, 4.0],
        sr_ratios: List[int] = [8, 4, 2, 1],
        depths: List[int] = [2, 2, 2, 2],
    ):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stages
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], sr_ratios[0], mlp_ratios[0]) for _ in range(depths[0])
        ])
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], sr_ratios[1], mlp_ratios[1]) for _ in range(depths[1])
        ])
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], sr_ratios[2], mlp_ratios[2]) for _ in range(depths[2])
        ])
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], sr_ratios[3], mlp_ratios[3]) for _ in range(depths[3])
        ])

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H1, W1)
        x1 = self.norm1(x).permute(0, 2, 1).reshape(B, -1, H1, W1)
        outs.append(x1)

        # Stage 2
        x, H2, W2 = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H2, W2)
        x2 = self.norm2(x).permute(0, 2, 1).reshape(B, -1, H2, W2)
        outs.append(x2)

        # Stage 3
        x, H3, W3 = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H3, W3)
        x3 = self.norm3(x).permute(0, 2, 1).reshape(B, -1, H3, W3)
        outs.append(x3)

        # Stage 4
        x, H4, W4 = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H4, W4)
        x4 = self.norm4(x).permute(0, 2, 1).reshape(B, -1, H4, W4)
        outs.append(x4)

        return outs


class SegFormerHead(nn.Module):
    """All-MLP lightweight decoder head."""

    def __init__(self, in_channels: List[int], embedding_dim: int = 256, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.linear_c1 = nn.Conv2d(in_channels[0], embedding_dim, 1)
        self.linear_c2 = nn.Conv2d(in_channels[1], embedding_dim, 1)
        self.linear_c3 = nn.Conv2d(in_channels[2], embedding_dim, 1)
        self.linear_c4 = nn.Conv2d(in_channels[3], embedding_dim, 1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        h1, w1 = c1.shape[-2:]

        _c1 = self.linear_c1(c1)
        _c2 = F.interpolate(self.linear_c2(c2), size=(h1, w1), mode="bilinear", align_corners=True)
        _c3 = F.interpolate(self.linear_c3(c3), size=(h1, w1), mode="bilinear", align_corners=True)
        _c4 = F.interpolate(self.linear_c4(c4), size=(h1, w1), mode="bilinear", align_corners=True)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        logits = self.classifier(_c)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=True)


class SegFormer(nn.Module):
    """
    SegFormer Architecture for Document Tampering Detection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        variant: str = "b0",
    ):
        super().__init__()
        if variant == "b1":
            embed_dims = [64, 128, 320, 512]
            depths = [2, 2, 2, 2]
            head_dim = 256
        else:  # b0 (default, fast and lightweight: ~3.7M params)
            embed_dims = [32, 64, 160, 256]
            depths = [2, 2, 2, 2]
            head_dim = 128

        self.encoder = MixTransformerEncoder(in_chans=in_channels, embed_dims=embed_dims, depths=depths)
        self.decoder = SegFormerHead(in_channels=embed_dims, embedding_dim=head_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decoder(features, target_size=input_size)
        return logits


def get_segformer_model(
    in_channels: int = 3,
    num_classes: int = 1,
    variant: str = "b0",
    device: torch.device = None,
) -> SegFormer:
    """Factory builder for SegFormer."""
    model = SegFormer(in_channels=in_channels, num_classes=num_classes, variant=variant)
    if device is not None:
        model = model.to(device)
    return model
