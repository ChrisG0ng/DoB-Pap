"""Frozen UniCAS (ViT-L/16) encoder returning multi-block patch tokens."""

import functools
from typing import List, Sequence, Tuple

import timm
import torch
import torch.nn as nn

UNICAS_HF_REPO = 'jianght/UniCAS'
UNICAS_HF_FILENAME = 'UniCAS.pth'
UNICAS_MEAN = (0.485, 0.456, 0.406)
UNICAS_STD = (0.229, 0.224, 0.225)
UNICAS_IMG_SIZE = 224
UNICAS_PATCH_SIZE = 16
UNICAS_EMBED_DIM = 1024

_VIT_KWARGS = {
    'img_size': UNICAS_IMG_SIZE,
    'patch_size': UNICAS_PATCH_SIZE,
    'depth': 24,
    'num_heads': 16,
    'init_values': 1e-5,
    'embed_dim': UNICAS_EMBED_DIM,
    'mlp_ratio': 2.671875 * 2,
    'num_classes': 0,
    'no_embed_class': False,
    'reg_tokens': 0,
}


class UniCASEncoder(nn.Module):
    """
    Frozen UniCAS backbone exposing the patch tokens of selected blocks.

    Args:
        block_indices: 1-indexed transformer blocks to extract (e.g. 6, 12, 18, 24).
        pretrained: Download the official UniCAS weights from the Hugging Face hub.
            Not needed when the full model is restored from a checkpoint.
    """

    def __init__(self, block_indices: Sequence[int] = (6, 12, 18, 24), pretrained: bool = False):
        super().__init__()
        self.block_indices = [i - 1 for i in block_indices]
        self.embed_dim = UNICAS_EMBED_DIM
        self.grid_size = (UNICAS_IMG_SIZE // UNICAS_PATCH_SIZE,) * 2

        self.backbone = timm.models.VisionTransformer(
            **_VIT_KWARGS,
            mlp_layer=functools.partial(timm.layers.mlp.GluMlp, gate_last=False),
            act_layer=nn.SiLU,
        )
        if pretrained:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(repo_id=UNICAS_HF_REPO, filename=UNICAS_HF_FILENAME)
            self.backbone.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))

        n_blocks = len(self.backbone.blocks)
        if any(i < 0 or i >= n_blocks for i in self.block_indices):
            raise ValueError(f"block_indices must be within 1..{n_blocks}, got {list(block_indices)}")

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        """
        Args:
            x: Normalized RGB of shape (B, 3, 224, 224).

        Returns:
            (tokens, grid_size): one (B, N, 1024) tensor per block, CLS token removed.
        """
        with torch.no_grad():
            tokens = self.backbone.forward_intermediates(
                x,
                indices=self.block_indices,
                output_fmt='NLC',
                intermediates_only=True,
                norm=False,
            )
        return [t.detach() for t in tokens], self.grid_size
