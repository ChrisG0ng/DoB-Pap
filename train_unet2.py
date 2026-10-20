"""
Train the UNet2 stain unmixing model on synthetic data (reference only; data not released).

Usage:
    python train_unet2.py --config config/train_unet2_config.ini
"""

import argparse

from src.models import UNet2
from src.training.supervised import train_supervised
from src.utils.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Train UNet2')
    parser.add_argument('--config', default='config/train_unet2_config.ini')
    cfg = load_config(parser.parse_args().config)
    keep_prob = cfg.getfloat('MODEL', 'dropout_keep_prob', fallback=0.95)
    best = train_supervised(cfg, lambda: UNet2(dropout_keep_prob=keep_prob), 'unet2_stain')
    print(f"Best model: {best}")


if __name__ == '__main__':
    main()
