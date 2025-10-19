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

### Combo 4: W Space + LPIPS Loss + Encoder Initialization  
**Purpose:** Use pre-trained encoder (e4e/pSp) for warm-start initialization  
**Configuration:**
- Latent space: W (**compatible with standard 1024×1024 encoders**)
- Loss: LPIPS (perceptual)
- Initialization: encoder (e4e or pSp with automatic upscaling)
- Steps: 300 (fewer steps needed with encoder warm-start)
- Learning rate: 0.01

**Note:** Uses pre-trained e4e or pSp encoders with automatic 128×128 → 1024×1024 upscaling. See [Encoder Setup](#encoder-setup) below for download instructions.

```bash
python invert.py --input data/samples/ --preset combo_04 --device cuda
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

---

### Combo 4: W Space + LPIPS Loss + Encoder Init
![Combo 4 Results - Placeholder](path/to/combo4_results.png)

**Average Metrics:**
- PSNR: XX.XX dB
- SSIM: 0.XXXX
- LPIPS: 0.XXXX

**Observations:** [To be filled with analysis of encoder-based initialization vs optimization-only]

## Encoder Setup

### About Encoder-Based Initialization

Encoder-based initialization uses a neural network to directly predict StyleGAN latent codes from images. This provides a "warm start" for optimization, offering:
- ✅ Faster convergence (fewer optimization steps needed)
- ✅ Better initialization than random or mean_w  
- ✅ Improved final reconstruction quality

### Using Pre-trained Encoders with Upscaling

**Our Approach:** Use pre-trained e4e or pSp encoders (trained on 1024×1024 FFHQ) with our 128×128 model.

**How it works:**
1. **Input**: 128×128 image
2. **Upscale**: Automatically upscale to 1024×1024 (bicubic interpolation)
3. **Encode**: Feed to e4e/pSp encoder → get W+ latent [1, 18, 512]
4. **Convert**: Average W+ to W space → [1, 512] (perfectly compatible!)
5. **Optimize**: Use as initialization for our 128×128 generator

**Why W space?** The W latent space [512] is universal across all StyleGAN2 resolutions, ensuring perfect compatibility.

### Setup Instructions

#### Step 1: Clone e4e Repository

The encoder loader needs the e4e codebase to properly load the model architecture:

```bash
# Clone into external directory (required)
git clone https://github.com/omertov/encoder4editing.git external/encoder4editing

# Install e4e dependencies
pip install -r external/encoder4editing/requirements.txt
```

#### Step 2: Download Encoder Weights

Download the pre-trained FFHQ encoder:

```bash
# Direct download link (Google Drive):
# https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view

# Using gdown (recommended):
pip install gdown
gdown --id 1cUv_reLE6k3604or78EranS7XzuVMWeO -O checkpoints/e4e_ffhq_encode.pt
```

**Note:** The checkpoint should be placed in `checkpoints/e4e_ffhq_encode.pt` (this path is already configured in `configs/init/encoder.yaml`)

### Quick Links
- **e4e GitHub**: https://github.com/omertov/encoder4editing
- **pSp GitHub**: https://github.com/eladrich/pixel2style2pixel
- **Both provide pre-trained weights** for FFHQ 1024×1024

### Running Without Encoder

If you don't want to use encoder initialization, combos 1-3 work excellently with optimization-only:
- **Combo 1**: W + L2 (baseline)
- **Combo 2**: W+ + L2 (more expressive space)
- **Combo 3**: W+ + LPIPS (perceptual loss)

These achieve strong results with 600 optimization steps.

## Project Structure

```
GAN_Inversion/
├── configs/
│   ├── base.yaml              # Base configuration
│   ├── experiment.yaml        # Experiment presets (combo_01-04)
│   ├── losses/                # Loss-specific configs
│   └── init/
│       ├── mean_w.yaml        # Mean W initialization config
│       ├── random_w.yaml      # Random initialization config
│       └── encoder.yaml       # Encoder initialization config
├── data/
│   └── samples/               # Sample FFHQ images
├── models/
│   ├── stylegan2_loader.py    # StyleGAN2 loading and wrapper
│   └── encoder_loader.py      # Encoder loading and architectures
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
├── docs/                      # Documentation
│   └── report.pdf             # Full project report
└── README.md
```

## Documentation

For detailed technical report, methodology, and analysis, see: [Project Report](docs/report.pdf)