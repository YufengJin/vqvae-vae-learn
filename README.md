# VQ-VAE / VAE Learning Material

This project is a PyTorch implementation of **VQ-VAE (Vector Quantized Variational Autoencoder)** and **VAE (Variational Autoencoder)**, suitable as learning material for studying and experimenting with both generative models.

- **VQ-VAE** paper: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- **DeepMind original** (TensorFlow): [sonnet/vqvae.py](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py) | [Jupyter example](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb)

## Installation

```bash
pip install -r requirements.txt
```

## Training Commands

Uses Hydra configuration. Main config is in `conf/config.yaml`; override via command line.

### Basic Usage

```bash
# VAE training (CIFAR10 default)
python3 main.py model=vae dataset=CIFAR10 batch_size=64 save=true

# VQ-VAE training
python3 main.py model=vqvae dataset=CIFAR10 batch_size=64 save=true

# Quick debug (few steps, CPU, wandb disabled)
python3 main.py model=vqvae debug=true cpu=true n_steps=1000
```

### Multirun

Add `-m` or `--multirun`, use comma-separated values to sweep all combinations:

```bash
# Run both VAE and VQ-VAE
python3 main.py -m model=vae,vqvae n_steps=1000

# Combined sweep: 2 models × 2 embedding_dim = 4 experiments
python3 main.py -m model=vae,vqvae embedding_dim=32,64 n_steps=5000

# Multiple n_embeddings
python3 main.py -m model=vqvae n_embeddings=256,512,1024
```

Output directory: Hydra creates subdirs under `outputs/YYYY-MM-DD/HH-MM-SS/` per run.

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | vqvae | vae / vqvae |
| `dataset` | CIFAR10 | CIFAR10 / BLOCK / IMAGENET |
| `n_steps` | 100000 | Training steps |
| `batch_size` | 512 | Batch size |
| `embedding_dim` | 64 | Embedding dimension |
| `n_embeddings` | 512 | Codebook size (VQ-VAE) |
| `learning_rate` | 0.0003 | Learning rate |
| `save` | false | Save model checkpoints |
| `debug` | false | Debug mode (disables wandb) |
| `cpu` | false | Force CPU |
| `data_root` | null | Data root (required for IMAGENET) |

More parameters: see `conf/config.yaml`.

## ImageNet

Supported datasets: `CIFAR10` (default), `BLOCK`, `IMAGENET`.

### Directory Layout

Standard ILSVRC layout: `data_root/train/` and `data_root/val/` with class subdirs:

```
data_root/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

### Training Example

If ImageNet is at `~/localdisk/imagenet/ILSVRC/Data/CLS-LOC`:

```bash
python3 main.py dataset=IMAGENET \
  data_root=~/localdisk/imagenet/ILSVRC/Data/CLS-LOC \
  save=true batch_size=64 n_steps=50000
```

**Note**: Input is fixed at 32×32; ImageNet images are auto-resized.

## Model Architecture

VQ-VAE consists of:

1. **Encoder**: `x -> z_e`, maps input to continuous latent space
2. **VectorQuantizer**: `z_e -> z_q`, quantizes encoder output to discrete codebook indices
3. **Decoder**: `z_q -> x_hat`, reconstructs image from discrete representation

Encoder and Decoder use convolutional stacks with Residual blocks (see [ResNet](https://arxiv.org/abs/1512.03385)).

```
models/
    - decoder.py   -> Decoder
    - encoder.py   -> Encoder
    - quantizer.py -> VectorQuantizer
    - residual.py  -> ResidualLayer, ResidualStack
    - vqvae.py     -> VQVAE
    - vae.py       -> VAE
```

## PixelCNN: Sampling from Latent Space

VQ-VAE maps images to a latent space with the same spatial structure as a 1-channel image (e.g. 32×32×3 -> 8×8×1). You can fit a PixelCNN over latent indices `z_ij` and sample.

Workflow:

1. Train VQ-VAE
2. Encode your dataset with the trained model; save `min_encoding_indices` (from `quantizer.py`) via `np.save`
3. Point `utils.load_latent_block` to the saved latent dataset
4. Run the PixelCNN script

```bash
python pixelcnn/gated_pixelcnn.py
```

Defaults to `LATENT_BLOCK` dataset; you must first train VQ-VAE and export latent representations.
