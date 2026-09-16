"""
DocForensics AI — U-Net Segmentation Baseline (Phase 5)
Modular, high-performance U-Net architecture for pixel-level document tampering localization.

Architecture:
- Input: RGB document image [B, 3, H, W]
- Encoder: 4 downsampling stages with DoubleConv (Conv -> BN -> ReLU x 2) + MaxPool
- Bottleneck: High-capacity feature extraction
- Decoder: 4 upsampling stages with skip connections + DoubleConv
- Output: 1-channel logit map [B, 1, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv with skip connection concatenation"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 spatial dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping to class logits"""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net Architecture for Document Tampering Segmentation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        c = base_channels  # e.g., 32 -> stages: 32, 64, 128, 256, 512

        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(c * 8, (c * 16) // factor)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.up1 = Up(c * 16, (c * 8) // factor, bilinear)
        self.up2 = Up(c * 8, (c * 4) // factor, bilinear)
        self.up3 = Up(c * 4, (c * 2) // factor, bilinear)
        self.up4 = Up(c * 2, c, bilinear)
        self.outc = OutConv(c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def get_unet_model(
    in_channels: int = 3,
    num_classes: int = 1,
    base_channels: int = 32,
    device: torch.device = None,
) -> UNet:
    """Factory helper to build and initialize UNet model."""
    model = UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    net = UNet(in_channels=3, num_classes=1, base_channels=32)
    dummy_input = torch.randn(2, 3, 512, 512)
    out = net(dummy_input)
    print(f"UNet output shape: {out.shape}")
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
