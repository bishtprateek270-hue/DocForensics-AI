"""
DocForensics AI — Multi-Stream Forensic Architecture (Phase 10)
Supports flexible multi-stream forensic feature extraction:
- Stream 1: RGB Spatial Semantics (ResNet encoder)
- Stream 2: SRM High-Pass Noise Residuals (30 SRM directional filters)
- Stream 3: Frequency DCT Discontinuity Stream (Block-DCT energy residuals)
- Gated Spatial & Channel Attention Fusion
- Atrous Spatial Pyramid Pooling (ASPP) multi-scale decoder
"""

import math
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.forensics.srm_filters import SRMExtractor
from src.forensics.frequency_extractor import FrequencyDCTExtractor
from src.forensics.noise_extractor import LocalNoiseExtractor
from src.models.deeplabv3plus import ASPP


class GatedMultiStreamFusion(nn.Module):
    """
    Dynamically weights N incoming feature streams using spatial and channel attention.
    """

    def __init__(self, stream_channels: List[int], out_channels: int = 512):
        super().__init__()
        self.num_streams = len(stream_channels)
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for c in stream_channels
        ])

        # Gate generator across concatenated streams
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels * self.num_streams, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, self.num_streams, kernel_size=1, bias=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        target_size = feature_list[0].shape[-2:]
        projected = []
        for feat, proj in zip(feature_list, self.proj_layers):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=True)
            projected.append(proj(feat))

        cat_feat = torch.cat(projected, dim=1)
        gate_logits = self.gate_conv(cat_feat)  # (B, num_streams, H, W)
        gate_weights = F.softmax(gate_logits, dim=1)

        fused = torch.zeros_like(projected[0])
        for idx, p_feat in enumerate(projected):
            w = gate_weights[:, idx:idx + 1, :, :]
            fused = fused + (w * p_feat)

        return self.out_conv(fused)


class ResidualStreamEncoder(nn.Module):
    """Dedicated convolutional encoder for forensic noise / DCT / residual feature maps."""

    def __init__(self, in_channels: int = 30, base_channels: int = 64, out_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x3


class MultiStreamForensicNet(nn.Module):
    """
    Multi-Stream Forensic Neural Network for Document Tampering Localization.
    Supports stream_mode:
    - 'rgb_srm' (2 streams: RGB + SRM 30)
    - 'rgb_srm_dct' (3 streams: RGB + SRM 30 + DCT 3)
    - 'rgb_srm_noise' (3 streams: RGB + SRM 30 + Local Noise 3)
    - 'rgb_srm_dct_noise' (4 streams)
    """

    def __init__(
        self,
        stream_mode: str = "rgb_srm",
        rgb_backbone: str = "resnet34",
        pretrained: bool = True,
    ):
        super().__init__()
        self.stream_mode = stream_mode.lower().strip()

        # Stream 1: RGB Spatial Stream
        if rgb_backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.rgb_encoder_low = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.rgb_encoder_high = nn.Sequential(resnet.layer2, resnet.layer3, resnet.layer4)
            rgb_high_channels = 2048
            self.rgb_low_channels = 256
        else:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.rgb_encoder_low = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.rgb_encoder_high = nn.Sequential(resnet.layer2, resnet.layer3, resnet.layer4)
            rgb_high_channels = 512
            self.rgb_low_channels = 64

        stream_channel_list = [rgb_high_channels]

        # Stream 2: SRM High-Pass Filter Stream
        self.srm_extractor = SRMExtractor()
        self.srm_encoder = ResidualStreamEncoder(in_channels=3, out_channels=256)
        stream_channel_list.append(256)

        # Stream 3: DCT Frequency Stream (if enabled)
        if "dct" in self.stream_mode:
            self.dct_extractor = FrequencyDCTExtractor(block_size=8)
            self.dct_encoder = ResidualStreamEncoder(in_channels=3, out_channels=128)
            stream_channel_list.append(128)
        else:
            self.dct_extractor = None
            self.dct_encoder = None

        # Stream 4: Local Noise Residual Stream (if enabled)
        if "noise" in self.stream_mode:
            self.noise_extractor = LocalNoiseExtractor()
            self.noise_encoder = ResidualStreamEncoder(in_channels=3, out_channels=128)
            stream_channel_list.append(128)
        else:
            self.noise_extractor = None
            self.noise_encoder = None

        # Dynamic Gated Fusion
        self.fusion = GatedMultiStreamFusion(stream_channels=stream_channel_list, out_channels=512)

        # Multi-scale ASPP
        self.aspp = ASPP(in_channels=512, atrous_rates=[6, 12, 18], out_channels=256)

        # Decoder & Low-Level RGB skip connection projection
        self.low_proj = nn.Sequential(
            nn.Conv2d(self.rgb_low_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        input_size = x_rgb.shape[-2:]

        # 1. RGB Feature Extraction
        f_low = self.rgb_encoder_low(x_rgb)
        f_high = self.rgb_encoder_high(f_low)

        active_streams = [f_high]

        # 2. SRM Feature Extraction
        srm_maps = self.srm_extractor(x_rgb)
        f_srm = self.srm_encoder(srm_maps)
        active_streams.append(f_srm)

        # 3. DCT Feature Extraction (if enabled)
        if self.dct_extractor is not None and self.dct_encoder is not None:
            dct_maps = self.dct_extractor(x_rgb)
            f_dct = self.dct_encoder(dct_maps)
            active_streams.append(f_dct)

        # 4. Noise Feature Extraction (if enabled)
        if self.noise_extractor is not None and self.noise_encoder is not None:
            noise_maps = self.noise_extractor(x_rgb)
            f_noise = self.noise_encoder(noise_maps)
            active_streams.append(f_noise)

        # 5. Gated Fusion
        fused = self.fusion(active_streams)

        # 6. ASPP & Decoder
        f_aspp = self.aspp(fused)
        f_aspp_up = F.interpolate(f_aspp, size=f_low.shape[-2:], mode="bilinear", align_corners=True)

        low_feat = self.low_proj(f_low)
        dec_in = torch.cat([f_aspp_up, low_feat], dim=1)
        logits = self.decoder(dec_in)

        # Final full-resolution upsampling
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=True)


def get_multi_stream_model(
    stream_mode: str = "rgb_srm",
    rgb_backbone: str = "resnet34",
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> MultiStreamForensicNet:
    """Factory helper to load multi-stream model."""
    model = MultiStreamForensicNet(
        stream_mode=stream_mode,
        rgb_backbone=rgb_backbone,
        pretrained=pretrained,
    )
    if device is not None:
        model = model.to(device)
    return model
