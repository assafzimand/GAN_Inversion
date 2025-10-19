# Optimization-Based GAN Inversion (StyleGAN2)

Minimal, modular implementation of optimization-based GAN inversion using pretrained StyleGAN2.

Uses **HuggingFace StyleGAN2-FFHQ-128** (128×128 resolution) for fast experimentation.

## Features
- Latent spaces: **W** and **W+**
- Loss functions: **L2** (pixel), **LPIPS** (perceptual)
- Initializations: **mean_w**, encoder-based (final stage)
- Quantitative metrics: **PSNR, SSIM, LPIPS**
- Evolution visualization: see optimization progress every 100 steps
- Reproducible experiments with configs and seeding
- Automatic model downloading from HuggingFace Hub

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

# (Optional) Generate synthetic test images
python scripts/prepare_data.py --mode generate --count 3
```

**Note:** Pretrained StyleGAN2 weights are automatically downloaded from HuggingFace Hub on first run.

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

### Model Details

**Generator:** `hajar001/stylegan2-ffhq-128` from HuggingFace Hub  
**Architecture:** StyleGAN2 with 12 layers  
**Resolution:** 128×128 pixels  
**Latent Dimensions:** 
- W space: `[1, 512]`
- W+ space: `[1, 12, 512]` (per-layer control)

Images are automatically resized to 128×128 during loading.

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

Each run creates a unique directory `outputs/{combo}_{image}_{timestamp}/` containing:
```
outputs/combo_01_test_image_01_20251019_143052/
├── config.yaml                     # Full configuration used
├── originals/                      # Original input images
├── reconstructions/                # Final generated reconstructions
├── comparisons/
│   └── <name>_evolution.png        # Evolution panel: original + steps 0,100,200,... with metrics
├── metrics.json                    # PSNR, SSIM, LPIPS per image
├── <name>_loss_curve.png           # Loss convergence plot
└── <name>_loss_history.json        # Raw loss values per step
```

**Evolution Panel:** Shows the original image alongside reconstructions at iterations 0, 100, 200, etc., with final metrics (PSNR, SSIM, LPIPS) displayed in the title.

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

