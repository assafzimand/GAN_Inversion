# Optimization-Based GAN Inversion (StyleGAN2)

Minimal, modular implementation of optimization-based GAN inversion using pretrained StyleGAN2.

Uses **HuggingFace StyleGAN2-FFHQ-128** (128×128 resolution) for fast experimentation.

## Features
- Latent spaces: **W** and **W+**
- Loss functions: **L2** (pixel), **LPIPS** (perceptual)
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
```

**Note:** 
- Pretrained StyleGAN2 weights are automatically downloaded from HuggingFace Hub on first run
- Sample FFHQ images are included in `data/samples/` (ffhq_1.png, ffhq_2.png, ffhq_3.png)

## Usage

Run inversion on a single image:
```bash
python invert.py --input data/samples/ffhq_1.png --preset combo_01 --device cuda
```

Run on all sample images:
```bash
python invert.py --input data/samples/ --preset combo_01 --device cuda
```

## Model Details

**Generator:** `hajar001/stylegan2-ffhq-128` from HuggingFace Hub  
**Architecture:** StyleGAN2 with 12 layers  
**Resolution:** 128×128 pixels  
**Latent Dimensions:** 
- W space: `[1, 512]`
- W+ space: `[1, 12, 512]` (per-layer control)

Images are automatically resized to 128×128 during loading.

## Experiment Presets

The project implements three optimization-based inversion experiments:

### Combo 1: W Space + L2 Loss
**Purpose:** Baseline with simplest latent space and pixel-wise loss  
**Configuration:**
- Latent space: W
- Loss: L2 (MSE)
- Initialization: mean_w
- Steps: 600
- Learning rate: 0.01

```bash
python invert.py --input data/samples/ --preset combo_01 --device cuda
```

### Combo 2: W+ Space + L2 Loss
**Purpose:** More expressive W+ space (per-layer control) with pixel-wise loss  
**Configuration:**
- Latent space: W+
- Loss: L2 (MSE)
- Initialization: mean_w
- Steps: 600
- Learning rate: 0.01

```bash
python invert.py --input data/samples/ --preset combo_02 --device cuda
```

### Combo 3: W+ Space + LPIPS Loss
**Purpose:** Perceptual loss for better visual quality  
**Configuration:**
- Latent space: W+
- Loss: LPIPS (perceptual)
- Initialization: mean_w
- Steps: 600
- Learning rate: 0.01

```bash
python invert.py --input data/samples/ --preset combo_03 --device cuda
```

## Output Structure

Each run creates a unique directory `outputs/{combo}_{image}_{timestamp}/` containing:
```
outputs/combo_01_multi_20251019_143052/
├── config.yaml                     # Full configuration used
├── originals/                      # Original input images
├── reconstructions/                # Final generated reconstructions
├── comparisons/
│   ├── ffhq_1_evolution.png        # Evolution: original + steps 0,100,200,... + metrics
│   ├── ffhq_2_evolution.png
│   └── ffhq_3_evolution.png
├── metrics.json                    # PSNR, SSIM, LPIPS for all images
├── ffhq_1_loss_curve.png           # Individual loss curves
├── ffhq_1_loss_history.json
├── ffhq_2_loss_curve.png
├── ffhq_2_loss_history.json
├── ffhq_3_loss_curve.png
└── ffhq_3_loss_history.json
```

**Evolution Panel:** Shows the original image alongside reconstructions at iterations 0, 100, 200, 300, 400, 500, 600, with final metrics (PSNR, SSIM, LPIPS) displayed in the title.

## Results

### Combo 1: W Space + L2 Loss
![Combo 1 Results - Placeholder](path/to/combo1_results.png)

**Average Metrics:**
- PSNR: XX.XX dB
- SSIM: 0.XXXX
- LPIPS: 0.XXXX

**Observations:** [To be filled with analysis]

---

### Combo 2: W+ Space + L2 Loss
![Combo 2 Results - Placeholder](path/to/combo2_results.png)

**Average Metrics:**
- PSNR: XX.XX dB
- SSIM: 0.XXXX
- LPIPS: 0.XXXX

**Observations:** [To be filled with analysis]

---

### Combo 3: W+ Space + LPIPS Loss
![Combo 3 Results - Placeholder](path/to/combo3_results.png)

**Average Metrics:**
- PSNR: XX.XX dB
- SSIM: 0.XXXX
- LPIPS: 0.XXXX

**Observations:** [To be filled with analysis]

## Project Structure

```
GAN_Inversion/
├── configs/
│   ├── base.yaml              # Base configuration
│   ├── experiment.yaml        # Experiment presets (combo_01, combo_02, combo_03)
│   ├── losses/                # Loss-specific configs
│   └── init/                  # Initialization configs
├── data/
│   └── samples/               # Sample FFHQ images
├── models/
│   └── stylegan2_loader.py    # StyleGAN2 loading and wrapper
├── losses/
│   ├── l2.py                  # L2 (MSE) loss
│   └── lpips_loss.py          # LPIPS perceptual loss
├── engine/
│   ├── inverter.py            # Core optimization loop
│   └── metrics.py             # Evaluation metrics (PSNR, SSIM, LPIPS)
├── utils/
│   ├── image_io.py            # Image loading/saving
│   ├── viz.py                 # Visualization (evolution panels, loss curves)
│   └── seed.py                # Reproducibility utilities
├── tests/                     # Unit tests
├── invert.py                  # Main CLI entry point
├── requirements.txt           # Dependencies
└── README.md
```

## Testing
```bash
pytest tests/ -v
```
