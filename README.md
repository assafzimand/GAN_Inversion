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

### Invert a single image
```bash
python invert.py \
  --input data/samples/face.png \
  --output outputs/experiment_01 \
  --config configs/experiment.yaml \
  --seed 42
```

### Run with specific settings
```bash
python invert.py \
  --input data/samples/ \
  --latent_space W+ \
  --loss lpips \
  --steps 600 \
  --lr 0.01 \
  --device cuda
```

### Output Structure
Each run produces:
- `reconstructions/` - Generated images
- `diffs/` - Side-by-side comparison panels
- `metrics.json` - PSNR, SSIM, LPIPS per image
- `loss_curve.png` - Convergence visualization
- `loss_history.json` - Raw loss per step

## Experiments

Follow the staged combos in `PRD.md`:
1. **W • L2 • mean_w** (~300 steps)
2. **W+ • L2 • mean_w** (~300 steps)
3. **W+ • LPIPS • mean_w** (~600 steps)
4. **W+ • LPIPS • encoder-init** (~200-300 steps)

See `configs/` for preset configurations.

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

