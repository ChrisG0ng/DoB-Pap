"""
Stain unmixing model built on the UniCAS cytology foundation model.

OD patch -> RGB (10^-OD) resized to 224x224 -> frozen UniCAS ViT-L/16
-> multi-block tokens -> per-block projection -> progressive fusion decoder
(+ optional shallow detail branch) -> abundance head -> resized back to the
input resolution.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abundance_head import AbundanceHead
from .feature_projection import MultiLevelFeatureProjection
from .shallow_detail_encoder import ShallowDetailEncoder
from .unicas_decoder import UniCASDecoder
from .unicas_encoder import UNICAS_IMG_SIZE, UNICAS_MEAN, UNICAS_STD, UniCASEncoder


class UniCASStainUnmixing(nn.Module):
    """
    Args:
        block_indices: 1-indexed UniCAS blocks used as decoder inputs (exactly four).
        proj_dim: Channels after per-block projection.
        decoder_channels: Channel width of the decoder.
        decoder_norm: 'group', 'batch' or 'instance'.
        use_detail_branch: Enable the shallow CNN detail branch.
        dropout: Dropout in projection and decoder (the encoder is frozen).
        pretrained_encoder: Download UniCAS weights (only needed for training from scratch).

    Example:
        >>> model = UniCASStainUnmixing()
        >>> conc = model(torch.rand(2, 3, 256, 256))  # (2, 4, 256, 256)
    """

    def __init__(self,
                 block_indices: Sequence[int] = (6, 12, 18, 24),
                 proj_dim: int = 256,
                 decoder_channels: int = 256,
                 decoder_norm: str = 'group',
                 use_detail_branch: bool = True,
                 dropout: float = 0.0,
                 pretrained_encoder: bool = False,
                 n_classes: int = 4):
        super().__init__()
        if len(block_indices) != 4:
            raise ValueError(f"Exactly four block_indices are required, got {list(block_indices)}")
        self.img_size = UNICAS_IMG_SIZE
        self.use_detail_branch = use_detail_branch

        self.encoder = UniCASEncoder(block_indices=block_indices, pretrained=pretrained_encoder)
        self.projection = MultiLevelFeatureProjection(
            num_levels=len(block_indices), in_dim=self.encoder.embed_dim, out_dim=proj_dim, dropout=dropout,
        )
        self.detail_encoder = ShallowDetailEncoder() if use_detail_branch else None
        self.decoder = UniCASDecoder(
            proj_dim=proj_dim,
            decoder_channels=decoder_channels,
            detail_channels=self.detail_encoder.channels if use_detail_branch else None,
            norm_type=decoder_norm,
            dropout=dropout,
        )
        self.abundance_head = AbundanceHead(decoder_channels, mid_channels=64, n_classes=n_classes)

        self.register_buffer('rgb_mean', torch.tensor(UNICAS_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('rgb_std', torch.tensor(UNICAS_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, optical_density: torch.Tensor) -> torch.Tensor:
        """Map OD (B, 3, H, W) to concentrations (B, 4, H, W)."""
        height, width = optical_density.shape[-2:]
        rgb = torch.clamp(torch.pow(10.0, -optical_density), 0.0, 1.0)
        rgb = F.interpolate(rgb, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        tokens, grid_size = self.encoder((rgb - self.rgb_mean) / self.rgb_std)
        features = self.projection(tokens, grid_size)
        detail = self.detail_encoder(rgb) if self.use_detail_branch else None

        abundance = self.abundance_head(self.decoder(features, self.img_size, detail))
        if abundance.shape[-2:] != (height, width):
            abundance = F.interpolate(abundance, size=(height, width), mode='bilinear', align_corners=False)
        return abundance
