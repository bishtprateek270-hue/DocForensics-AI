"""
DocForensics AI — DeepLabV3+ Architecture (Phase 6)
CNN-based semantic segmentation with Atrous Spatial Pyramid Pooling (ASPP)
and high/low-level feature fusion decoder for pixel-level tampering localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling with 1x1, 3x3 dilated convolutions and global pooling.
    """

    def __init__(self, in_channels: int, atrous_rates: List[int] = [6, 12, 18], out_channels: int = 256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3PlusDecoder(nn.Module):
    """
    DeepLabV3+ low-level feature fusion decoder.
    Combines low-level features (1/4 scale) with ASPP features.
    """

    def __init__(self, low_level_channels: int = 64, aspp_channels: int = 256, num_classes: int = 1):
        super().__init__()
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, aspp_feat: torch.Tensor, low_level_feat: torch.Tensor, target_size: tuple) -> torch.Tensor:
        low_proj = self.low_level_proj(low_level_feat)
        aspp_up = F.interpolate(aspp_feat, size=low_proj.shape[-2:], mode="bilinear", align_corners=True)
        fused = torch.cat([aspp_up, low_proj], dim=1)
        logits = self.fusion(fused)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=True)


class DeepLabV3Plus(nn.Module):
    """
    Complete DeepLabV3+ model with ResNet34 backbone.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, pretrained: bool = False):
        super().__init__()
        # Backbone: ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # 64 channels (1/4 scale) - Low level
        self.layer2 = resnet.layer2  # 128 channels (1/8 scale)
        self.layer3 = resnet.layer3  # 256 channels (1/16 scale)
        self.layer4 = resnet.layer4  # 512 channels (1/32 scale) - High level

        self.aspp = ASPP(in_channels=512, atrous_rates=[6, 12, 18], out_channels=256)
        self.decoder = DeepLabV3PlusDecoder(low_level_channels=64, aspp_channels=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x_stem = self.stem(x)
        low_level = self.layer1(x_stem)  # [B, 64, H/4, W/4]
        x2 = self.layer2(low_level)      # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)             # [B, 256, H/16, W/16]
        high_level = self.layer4(x3)     # [B, 512, H/32, W/32]

        aspp_out = self.aspp(high_level) # [B, 256, H/32, W/32]
        logits = self.decoder(aspp_out, low_level, target_size=input_size)
        return logits


def get_deeplabv3plus_model(
    in_channels: int = 3,
    num_classes: int = 1,
    pretrained: bool = False,
    device: torch.device = None,
) -> DeepLabV3Plus:
    """Factory builder for DeepLabV3+."""
    model = DeepLabV3Plus(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    if device is not None:
        model = model.to(device)
    return model
