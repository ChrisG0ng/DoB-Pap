"""Shallow CNN branch providing fine detail features to the decoder."""

from typing import Dict

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),
        nn.GELU(),
    )


class ShallowDetailEncoder(nn.Module):
    """RGB -> features at full, 1/2 and 1/4 resolution."""

    def __init__(self, in_channels: int = 3, full_channels: int = 32,
                 half_channels: int = 48, quarter_channels: int = 64):
        super().__init__()
        self.channels = {'full': full_channels, 'half': half_channels, 'quarter': quarter_channels}
        self.stem = _conv_block(in_channels, full_channels, stride=1)
        self.down1 = _conv_block(full_channels, half_channels, stride=2)
        self.down2 = _conv_block(half_channels, quarter_channels, stride=2)

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return {'full', 'half', 'quarter'} feature maps."""
        full = self.stem(rgb)
        half = self.down1(full)
        return {'full': full, 'half': half, 'quarter': self.down2(half)}
