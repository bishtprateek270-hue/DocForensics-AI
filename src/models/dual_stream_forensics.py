"""
DocForensics AI — Dual-Stream Forensic Segmentation Architecture (Phase 7)

Combines:
1. Stream 1 (RGB Spatial Stream): DeepLabV3+ / ResNet34 backbone extracting contextual and spatial semantics.
2. Stream 2 (Forensic Residual Stream): High-pass Spatial Rich Model (SRM) / noise residual encoder
   exposing subtle manipulation boundaries, inpainting textures, and copy-move noise inconsistencies.
3. Feature Fusion: Clean baseline (concat + conv + residual) and optional gated/attention fusion.
4. ASPP + Multi-Scale Decoder: Produces high-resolution pixel-level tampering segmentation logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, List

from src.forensics.srm_filters import SRMExtractor
from src.models.deeplabv3plus import ASPP


class BaselineFusion(nn.Module):
    """
    Clean baseline fusion: Concatenation + 1x1 Conv + Residual 3x3 Conv Block.
    Fuses high-level RGB spatial features (in_rgb) and forensic residual features (in_for)
    into a unified representation (out_channels).
    """

    def __init__(self, in_rgb: int = 512, in_for: int = 256, out_channels: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_rgb + in_for, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_rgb: torch.Tensor, f_for: torch.Tensor) -> torch.Tensor:
        # Match spatial resolution if needed
        if f_for.shape[-2:] != f_rgb.shape[-2:]:
            f_for = F.interpolate(f_for, size=f_rgb.shape[-2:], mode="bilinear", align_corners=True)
        cat_feat = torch.cat([f_rgb, f_for], dim=1)
        proj_feat = self.proj(cat_feat)
        res = self.res_block(proj_feat)
        return self.relu(proj_feat + res)


class GatedAttentionFusion(nn.Module):
    """
    Gated Attention Fusion: Dynamically weights RGB contextual features against
    forensic high-frequency noise anomalies using spatial and channel attention gates.
    """

    def __init__(self, in_rgb: int = 512, in_for: int = 256, out_channels: int = 512):
        super().__init__()
        self.for_proj = nn.Sequential(
            nn.Conv2d(in_for, in_rgb, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_rgb),
            nn.ReLU(inplace=True),
        )
        # Gate generator
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_rgb * 2, in_rgb // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_rgb // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_rgb // 2, 2, kernel_size=1, bias=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_rgb, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_rgb: torch.Tensor, f_for: torch.Tensor) -> torch.Tensor:
        if f_for.shape[-2:] != f_rgb.shape[-2:]:
            f_for = F.interpolate(f_for, size=f_rgb.shape[-2:], mode="bilinear", align_corners=True)
        f_for_proj = self.for_proj(f_for)

        cat_feat = torch.cat([f_rgb, f_for_proj], dim=1)
        gate_logits = self.gate_conv(cat_feat)  # [B, 2, H, W]
        gate_weights = F.softmax(gate_logits, dim=1)

        w_rgb = gate_weights[:, 0:1, :, :]
        w_for = gate_weights[:, 1:2, :, :]

        fused = w_rgb * f_rgb + w_for * f_for_proj
        return self.out_conv(fused)


class ForensicStreamEncoder(nn.Module):
    """
    Stream 2: Dedicated CNN feature extractor for SRM / noise residual maps.
    Captures high-frequency noise inconsistencies, edge discontinuities, and local texture artifacts.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # 1/4 scale, 64 channels

        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # 1/8 scale, 128 channels

        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )  # 1/16 scale, 256 channels

        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )  # 1/32 scale, 256 channels

    def forward(self, res_in: torch.Tensor):
        x_stem = self.stem(res_in)          # [B, 64, H/4, W/4]
        low_for = self.block1(x_stem)        # [B, 64, H/4, W/4]
        x2 = self.stage2(low_for)           # [B, 128, H/8, W/8]
        x3 = self.stage3(x2)                # [B, 256, H/16, W/16]
        high_for = self.stage4(x3)          # [B, 256, H/32, W/32]
        return low_for, high_for


class DualStreamDecoder(nn.Module):
    """
    Decodes fused high-level ASPP representations and multi-stream low-level features
    back to full-resolution tampering localization map.
    """

    def __init__(self, low_level_rgb_ch: int = 64, low_level_for_ch: int = 64, aspp_ch: int = 256, num_classes: int = 1):
        super().__init__()
        # Fuse low-level features from both streams
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_rgb_ch + low_level_for_ch, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(aspp_ch + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(
        self,
        aspp_feat: torch.Tensor,
        low_rgb: torch.Tensor,
        low_for: torch.Tensor,
        target_size: tuple,
    ) -> torch.Tensor:
        # Concatenate low-level features
        if low_for.shape[-2:] != low_rgb.shape[-2:]:
            low_for = F.interpolate(low_for, size=low_rgb.shape[-2:], mode="bilinear", align_corners=True)
        low_cat = torch.cat([low_rgb, low_for], dim=1)
        low_proj = self.low_level_proj(low_cat)

        aspp_up = F.interpolate(aspp_feat, size=low_proj.shape[-2:], mode="bilinear", align_corners=True)
        fused = torch.cat([aspp_up, low_proj], dim=1)
        logits = self.fusion(fused)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=True)


class DualStreamForensicNet(nn.Module):
    """
    Full Phase 7 Dual-Stream Forensic Tampering Detection & Localization Model.
    
    Streams:
    1. RGB Stream (ResNet34 backbone)
    2. Forensic Residual Stream (SRM high-pass filtering + Forensic CNN)
    Fusion: Baseline (concat+conv) or Gated Attention
    Decoder: ASPP + Dual-Stream Low-Level Feature Fusion Decoder
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained_backbone: bool = False,
        fusion_type: str = "baseline",  # "baseline" or "gated"
    ):
        super().__init__()
        self.fusion_type = fusion_type.lower()

        # Stream 1: RGB Spatial Backbone (ResNet34)
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained_backbone else None)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.rgb_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.rgb_layer1 = resnet.layer1  # [B, 64, H/4, W/4]
        self.rgb_layer2 = resnet.layer2  # [B, 128, H/8, W/8]
        self.rgb_layer3 = resnet.layer3  # [B, 256, H/16, W/16]
        self.rgb_layer4 = resnet.layer4  # [B, 512, H/32, W/32]

        # Stream 2: Forensic Residual Stream
        self.srm_extractor = SRMExtractor(in_channels=in_channels, out_channels=3)
        self.forensic_encoder = ForensicStreamEncoder(in_channels=3)

        # High-Level Feature Fusion
        if self.fusion_type == "gated":
            self.fusion = GatedAttentionFusion(in_rgb=512, in_for=256, out_channels=512)
        else:
            self.fusion = BaselineFusion(in_rgb=512, in_for=256, out_channels=512)

        # Multi-scale contextual pooling (ASPP)
        self.aspp = ASPP(in_channels=512, atrous_rates=[6, 12, 18], out_channels=256)

        # Decoder
        self.decoder = DualStreamDecoder(
            low_level_rgb_ch=64,
            low_level_for_ch=64,
            aspp_ch=256,
            num_classes=num_classes,
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        input_size = rgb.shape[-2:]

        # 1. Stream 1 Forward (RGB)
        x_stem = self.rgb_stem(rgb)
        low_rgb = self.rgb_layer1(x_stem)     # [B, 64, H/4, W/4]
        x_rgb2 = self.rgb_layer2(low_rgb)     # [B, 128, H/8, W/8]
        x_rgb3 = self.rgb_layer3(x_rgb2)      # [B, 256, H/16, W/16]
        high_rgb = self.rgb_layer4(x_rgb3)    # [B, 512, H/32, W/32]

        # 2. Stream 2 Forward (Forensic Residuals)
        srm_residuals = self.srm_extractor(rgb) # [B, 3, H, W]
        low_for, high_for = self.forensic_encoder(srm_residuals)

        # 3. High-Level Fusion
        fused_high = self.fusion(high_rgb, high_for) # [B, 512, H/32, W/32]

        # 4. ASPP
        aspp_out = self.aspp(fused_high)            # [B, 256, H/32, W/32]

        # 5. Dual-Stream Decoder
        logits = self.decoder(aspp_out, low_rgb, low_for, target_size=input_size)
        return logits


def get_dual_stream_model(
    in_channels: int = 3,
    num_classes: int = 1,
    pretrained_backbone: bool = False,
    fusion_type: str = "baseline",
    device: Optional[torch.device] = None,
) -> DualStreamForensicNet:
    """Factory builder for DualStreamForensicNet."""
    model = DualStreamForensicNet(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        fusion_type=fusion_type,
    )
    if device is not None:
        model = model.to(device)
    return model
