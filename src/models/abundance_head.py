"""Head mapping decoder features to non-negative stain abundances."""

import torch
import torch.nn as nn


class AbundanceHead(nn.Module):
    """
    3x3 Conv -> GELU -> 1x1 Conv -> ReLU.

    No dropout here: dropping channels right before the final ReLU easily
    collapses all outputs to zero.
    """

    def __init__(self, in_channels: int, mid_channels: int = 64, n_classes: int = 4):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv1x1 = nn.Conv2d(mid_channels, n_classes, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv1x1(self.act(self.conv3x3(x))))
