"""U-Net predicting non-negative stain concentrations (EY, H, LG, OG) from OD patches."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithReflectionPad(nn.Module):
    """Conv2d preceded by reflection padding."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4, mode='reflect')
        return self.conv(x)


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU -> Dropout) x 2."""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_p=0.0):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            Conv2dWithReflectionPad(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            Conv2dWithReflectionPad(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool -> DoubleConv."""

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample -> concat skip -> DoubleConv."""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_p=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """1x1 output convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv2dWithReflectionPad(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2(nn.Module):
    """
    U-Net with reflection padding and a ReLU output.

    Args:
        n_channels: Input channels (3, optical density).
        n_classes: Output stain channels (4).
        bilinear: Use bilinear upsampling instead of transposed convolution.
        dropout_keep_prob: Keep probability of the Dropout2d layers.

    Example:
        >>> conc = UNet2()(torch.rand(1, 3, 256, 256))  # (1, 4, 256, 256)
    """

    def __init__(self, n_channels=3, n_classes=4, bilinear=True, dropout_keep_prob=0.95):
        super().__init__()
        dropout_p = 1.0 - dropout_keep_prob
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64, dropout_p=dropout_p)
        self.down1 = Down(64, 128, dropout_p=dropout_p)
        self.down2 = Down(128, 256, dropout_p=dropout_p)
        self.down3 = Down(256, 512, dropout_p=dropout_p)
        self.down4 = Down(512, 1024 // factor, dropout_p=dropout_p)
        self.up1 = Up(1024, 512 // factor, bilinear, dropout_p=dropout_p)
        self.up2 = Up(512, 256 // factor, bilinear, dropout_p=dropout_p)
        self.up3 = Up(256, 128 // factor, bilinear, dropout_p=dropout_p)
        self.up4 = Up(128, 64, bilinear, dropout_p=dropout_p)
        self.outc = OutConv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map OD (B, 3, H, W) to concentrations (B, 4, H, W)."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.relu(self.outc(x))
