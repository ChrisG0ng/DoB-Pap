"""Per-block projection of ViT tokens to 2D feature maps."""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class FeatureProjectionBlock(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> reshape -> 3x3 depthwise conv."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.depthwise_conv = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim)

    def forward(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """Project (B, N, in_dim) tokens to a (B, out_dim, H, W) map."""
        h, w = grid_size
        b, n, _ = tokens.shape
        if n != h * w:
            raise ValueError(f"Token count {n} does not match grid {h}x{w}")
        x = self.dropout(self.act(self.linear(self.norm(tokens))))
        x = x.transpose(1, 2).reshape(b, self.out_dim, h, w)
        return self.depthwise_conv(x)


class MultiLevelFeatureProjection(nn.Module):
    """Independent projection for each encoder level."""

    def __init__(self, num_levels: int, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.projections = nn.ModuleList(
            [FeatureProjectionBlock(in_dim, out_dim, dropout=dropout) for _ in range(num_levels)]
        )

    def forward(self, tokens: Sequence[torch.Tensor], grid_size: Tuple[int, int]) -> List[torch.Tensor]:
        """Project each level; output order matches the input order."""
        if len(tokens) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} levels, got {len(tokens)}")
        return [proj(tok, grid_size) for proj, tok in zip(self.projections, tokens)]
