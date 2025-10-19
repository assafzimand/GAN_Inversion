# Optimization-Based GAN Inversion (StyleGAN2)

Minimal, modular implementation of optimization-based GAN inversion using pretrained StyleGAN2.

## Features
- Latent spaces: **W** and **W+**
- Loss functions: **L2** (pixel), **LPIPS** (perceptual)
- Initializations: **mean_w**, encoder-based (final stage)
- Quantitative metrics: **PSNR, SSIM, LPIPS**
- Reproducible experiments with configs and seeding

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd GAN_Inversion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained StyleGAN2-FFHQ weights
bash scripts/download_weights.sh
```

## Quickstart

### Basic Usage

Run inversion on a single image:
```bash
python invert.py --input data/samples/face.png --preset combo_01
```

Run on a folder of images:
```bash
python invert.py --input data/samples/ --preset combo_02
```

### Using Experiment Presets

The project includes 4 staged experiment presets (see `PRD.md` for rationale):

#### Preset 1: W • L2 • mean_w • 300 steps
Baseline sanity check with simplest settings:
```bash
python invert.py --input data/samples/face.png --preset combo_01
```

#### Preset 2: W+ • L2 • mean_w • 300 steps
More expressive W+ latent space (per-layer control):
```bash
python invert.py --input data/samples/face.png --preset combo_02
```

#### Preset 3: W+ • LPIPS • mean_w • 600 steps
Perceptual loss for better visual quality:
```bash
python invert.py --input data/samples/face.png --preset combo_03
```

> **Note:** Preset 4 (encoder initialization) is not yet implemented and will be added in the final stage.

### Custom Settings

Override preset parameters or run with fully custom settings:
```bash
python invert.py \
  --input data/samples/face.png \
  --latent_space W+ \
  --loss lpips \
  --init_method mean_w \
  --steps 600 \
  --lr 0.01 \
  --device cuda \
  --seed 42
```

Start from a preset and override specific parameters:
```bash
python invert.py \
  --input data/samples/ \
  --preset combo_02 \
  --steps 500 \
  --lr 0.02
```

### Output Structure

Each run creates a timestamped directory `outputs/run_YYYYMMDD_HHMMSS/` containing:
```
outputs/run_20250119_143052/
├── config.yaml              # Full configuration used
├── originals/               # Original input images
├── reconstructions/         # Generated reconstructions
├── comparisons/             # Side-by-side comparison panels
├── metrics.json             # PSNR, SSIM, LPIPS per image
├── <name>_loss_curve.png    # Loss convergence plot
└── <name>_loss_history.json # Raw loss values per step
```

## Experiments

To reproduce the staged experiments from the PRD:

**Experiment 1:** Baseline (W space, L2 loss)
```bash
python invert.py --input data/samples/ --preset combo_01
```

**Experiment 2:** W+ space comparison
```bash
python invert.py --input data/samples/ --preset combo_02
```

**Experiment 3:** Perceptual loss comparison
```bash
python invert.py --input data/samples/ --preset combo_03
```

Compare metrics across experiments using the generated `metrics.json` files.

## Testing
```bash
pytest tests/ -v
```

## Troubleshooting

### LPIPS issues
- Ensure `lpips` package is installed: `pip install lpips`
- For CPU fallback, set `--device cpu`

### Out of memory
- Reduce batch size in config
- Use W space instead of W+ (fewer parameters)

### Slow convergence
- Increase steps in config
- Try L-BFGS optimizer (experimental)

## Project Structure
See `PRD.md` for detailed requirements and `cursor/rules/` for coding standards.

## License
MIT

