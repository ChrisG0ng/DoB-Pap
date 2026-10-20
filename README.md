# DoB-Pap: Deep Learning-Based Stain Unmixing for Pap Smear Images

Stain unmixing for Papanicolaou RGB images. A network predicts per-pixel concentrations of
four stains (EY, H, LG, OG) from RGB or optical density (OD), and an unsupervised domain adaptation
pipeline adapts a pre-trained model to new staining conditions.

Two backbones are provided:

- **UNet2**: U-Net with reflection padding and a non-negative (ReLU) output.
- **UniCAS**: the frozen [UniCAS](https://huggingface.co/jianght/UniCAS) cytology foundation model
  (ViT-L/16). Multi-block tokens are projected and fused by a progressive decoder with an
  optional shallow CNN detail branch.

Both are trained on synthetic OD / concentration / stain-matrix triplets with a
reconstruction loss (OD = A·C) and a concentration loss.

Pre-trained checkpoints are hosted on
[Anonymous Hugging Face](https://anonymous-hf.com/a/1a5qgm2lb0wl/) and are not part of
this repository. A sample of real Papanicolaou images from Matias et al. [1] is in
`data/sample/` for running domain adaptation and inference.

## Installation

```bash
pip install -r requirements.txt
```

Download the two source checkpoints into the paths expected by the domain-adaptation
configs ([repository](https://anonymous-hf.com/a/1a5qgm2lb0wl/)):

```bash
mkdir -p models/unet2_synthetic_50000 models/uniCAS_models_synthetic_50000
curl -L -o models/unet2_synthetic_50000/unet2_stain_best.pth \
  https://anonymous-hf.com/api/a/1a5qgm2lb0wl/resolve/unet2_stain_best.pth
curl -L -o models/uniCAS_models_synthetic_50000/uniCAS_stain_best.pth \
  https://anonymous-hf.com/api/a/1a5qgm2lb0wl/resolve/uniCAS_stain_best.pth
```

A CUDA GPU is recommended (CPU is used automatically otherwise).

## Usage

Run the scripts from the repository root.

### Domain adaptation + inference

```bash
# UNet
python Domain_adaptation.py --config config/domain_adaptation_config.ini
python inference.py --config config/inference_config.ini

# UniCAS
python UniCAS_domain_adaptation.py --config config/uniCAS_domain_adaptation_config.ini
python UniCAS_inference.py --config config/uniCAS_inference_config.ini
```

Domain adaptation (results in `output/<backbone>/adaptation/`):

1. **Stain unmixing**: images are split into 256×256 OD patches and unmixed by the source model
   (`patch_output/`, `od_output/`, `background/`).
2. **Stain matrix estimation**: a stain matrix is fitted per patch. For each stain, the matrices
   of the most stain-dominant patches are filtered for outliers and averaged
   (`stain_matrix/overall_stain_matrix.txt`).
3. **Fine-tuning**: the source model is fine-tuned on the target OD patches with the estimated
   stain matrix fixed (`finetuned_model/`).

Inference uses the fine-tuned model and the estimated stain matrix. Results go to
`output/<backbone>/inference/`:

| Folder | Content |
|---|---|
| `concentration/` | `{image}_concentration.npy`, float32 (H, W, 4), channels EY/H/LG/OG |
| `reconstruction/` | per-stain and full reconstructed RGB images |
| `background/` | background intensity I0 and OD_background per image |
| `scaler/` | concentration statistics (mean + 2·std, 99th percentile) |

To process your own images, set `input_dir` / `test_data_dir` and the output paths in the configs.
Reduce `batch_size` if GPU memory is insufficient.

### Synthetic data and training (reference)

The source data is not released, so these scripts and configs document the procedure only:

```bash
python Synthetic_data_generation.py --config config/synthetic_data_generation_config.ini
python train_unet2.py --config config/train_unet2_config.ini
python train_uniCAS.py --config config/train_uniCAS_config.ini
```

`Synthetic_data_generation.py` augments ground-truth concentration patches and mixes them with
perturbed stain vectors. It writes `*_OD_*.npy` (256×256×3), `*_con_*.npy` (256×256×4) and
`generation_log.csv` (per-sample stain matrices). `train_uniCAS.py` downloads the UniCAS
encoder weights from the Hugging Face hub; the provided checkpoints already contain them.

## Project structure

```
├── Domain_adaptation.py            # UNet2 domain adaptation
├── inference.py                    # UNet2 inference
├── UniCAS_domain_adaptation.py     # UniCAS domain adaptation
├── UniCAS_inference.py             # UniCAS inference
├── Synthetic_data_generation.py    # synthetic data generation (reference)
├── train_unet2.py                  # UNet2 training (reference)
├── train_uniCAS.py                 # UniCAS training (reference)
├── config/                         # one .ini per script
├── data/sample/                    # sample RGB images (Matias et al. [1])
├── models/                         # downloaded checkpoints (not in git)
├── src/
│   ├── models/                     # UNet2, UniCAS encoder / projection / decoder / head
│   ├── data/                       # dataset, augmentation
│   ├── training/                   # trainer, fine-tuning, supervised training
│   └── utils/                      # image ops, stain estimation, scaling, loss, metrics
└── tests/                          # python -m unittest discover tests
```

## References

[1] Matias, André Victória, et al. "Segmentation, Detection, and Classification of Cell Nuclei on Oral Cytology Samples Stained with Papanicolaou." *SN Computer Science* 2.4 (2021): 285. https://doi.org/10.1007/s42979-021-00676-8

## License

This code is released for academic research purposes.
