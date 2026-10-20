"""
Train the UniCAS stain unmixing model on synthetic data (reference only; data not released).
The frozen UniCAS encoder weights are downloaded from the Hugging Face hub.

Usage:
    python train_uniCAS.py --config config/train_uniCAS_config.ini
"""

import argparse

from src.models import UniCASStainUnmixing
from src.training.supervised import train_supervised
from src.utils.common import load_config, unicas_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description='Train UniCAS stain unmixing model')
    parser.add_argument('--config', default='config/train_uniCAS_config.ini')
    cfg = load_config(parser.parse_args().config)
    kwargs = unicas_kwargs(cfg['MODEL'])
    best = train_supervised(cfg, lambda: UniCASStainUnmixing(**kwargs, pretrained_encoder=True), 'uniCAS_stain')
    print(f"Best model: {best}")


if __name__ == '__main__':
    main()
