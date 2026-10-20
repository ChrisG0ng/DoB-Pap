"""Progressive fusion decoder for the UniCAS stain unmixing model."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=min(32, num_channels), num_channels=num_channels)
    if norm_type == 'batch':
        return nn.BatchNorm2d(num_channels)
    if norm_type == 'instance':
        return nn.InstanceNorm2d(num_channels, affine=True)
    raise ValueError(f"Unsupported decoder norm type: {norm_type}")


class BottleneckBlock(nn.Module):
    """3x3 Conv -> Norm -> GELU -> Dropout on the deepest feature."""

    def __init__(self, channels: int, norm_type: str = 'group', dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = _make_norm(norm_type, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.norm(self.conv(x))))


class FusionStage(nn.Module):
    """Upsample, concatenate resized skip features, then two 3x3 convolutions."""

    def __init__(self, decoder_channels: int, skip_channels: Sequence[int], out_channels: int,
                 norm_type: str = 'group', dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(decoder_channels + sum(skip_channels), out_channels, kernel_size=3, padding=1)
        self.norm = _make_norm(norm_type, out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, skips: Sequence[torch.Tensor], size: Tuple[int, int]) -> torch.Tensor:
        feats = [F.interpolate(t, size=size, mode='bilinear', align_corners=False) for t in [x, *skips]]
        x = self.dropout(self.act(self.norm(self.conv1(torch.cat(feats, dim=1)))))
        return self.conv2(x)


class UniCASDecoder(nn.Module):
    """
    Decoder fusing four projected ViT levels (shallow -> deep) and optional detail features.

    Stage resolutions relative to img_size: 1/8 (block 18), 1/4 (block 12 + quarter detail),
    1/2 (block 6 + half detail), 1/1 (full detail).
    """

    def __init__(self, proj_dim: int = 256, decoder_channels: int = 256,
                 detail_channels: Optional[Dict[str, int]] = None,
                 norm_type: str = 'group', dropout: float = 0.0):
        super().__init__()
        self.use_detail = detail_channels is not None
        dc = detail_channels or {}

        def extra(key: str) -> List[int]:
            return [dc[key]] if self.use_detail else []

        self.bottleneck = BottleneckBlock(proj_dim, norm_type, dropout)
        self.stage_skip18 = FusionStage(proj_dim, [proj_dim], decoder_channels, norm_type, dropout)
        self.stage_skip12 = FusionStage(decoder_channels, [proj_dim] + extra('quarter'),
                                        decoder_channels, norm_type, dropout)
        self.stage_skip6 = FusionStage(decoder_channels, [proj_dim] + extra('half'),
                                       decoder_channels, norm_type, dropout)
        self.stage_full_res = FusionStage(decoder_channels, extra('full'),
                                          decoder_channels, norm_type, dropout)

    def forward(self, features: Sequence[torch.Tensor], img_size: int,
                detail: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Return a (B, decoder_channels, img_size, img_size) feature map."""
        feat6, feat12, feat18, feat24 = features

        def extra(key: str) -> List[torch.Tensor]:
            return [detail[key]] if self.use_detail else []

        x = self.bottleneck(feat24)
        x = self.stage_skip18(x, [feat18], (img_size // 8,) * 2)
        x = self.stage_skip12(x, [feat12] + extra('quarter'), (img_size // 4,) * 2)
        x = self.stage_skip6(x, [feat6] + extra('half'), (img_size // 2,) * 2)
        return self.stage_full_res(x, extra('full'), (img_size, img_size))
