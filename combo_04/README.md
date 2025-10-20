# Combo 04: Encoder-Based Inversion (Google Colab)

**Experiment Configuration:**
- **Resolution:** 1024×1024
- **Latent Space:** W+
- **Loss:** LPIPS (perceptual)
- **Initialization:** Encoder (e4e)
- **Steps:** 300
- **Learning Rate:** 0.01

---

## Overview

Combo 04 demonstrates **encoder-based initialization** for GAN inversion using the [Encoder for Editing (e4e)](https://github.com/omertov/encoder4editing) framework. Unlike Combos 1-3 (which run locally on 128×128 images), Combo 04:

- **Runs on Google Colab** (requires CUDA compilation for e4e components)
- Uses **1024×1024 resolution** (full FFHQ quality)
- Initializes latent codes with a **pre-trained encoder** instead of random/mean
- Refines the initial code using **our optimization loop** (Adam + LPIPS)

This is kept **separate from the main project** to avoid dependency conflicts and CUDA compilation requirements on local machines.

---

## Why Separate?

The e4e encoder uses custom CUDA operations that require:
- CUDA Toolkit installation
- C++ compiler
- Compilation during first run

Google Colab provides these out-of-the-box, making it the easiest way to run Combo 04.

---

## Quick Start

### Option 1: Run in Google Colab (Recommended)

1. **Open the notebook in Colab:**
   - Upload `combo_04_colab.ipynb` to your Google Drive
   - Open it with Google Colab
   - Or: Use the direct Colab link (if available)

2. **Run all cells:**
   - The notebook will automatically:
     - Mount your Google Drive
     - Clone the e4e repository
     - Install dependencies
     - Download the e4e checkpoint
     - Upload/load your target images
     - Run encoder-based inversion
     - Save results to your Drive

3. **Results will be saved to:**
   - `MyDrive/GAN_Inversion_Results/combo_04/`

### Option 2: Run Locally (Advanced)

Only if you have:
- CUDA Toolkit installed (`CUDA_HOME` set)
- Visual Studio Build Tools (Windows) or GCC (Linux)
- Compatible PyTorch + CUDA version

**Setup:**
```bash
# Clone e4e repository
cd combo_04
git clone https://github.com/omertov/encoder4editing.git

# Download e4e checkpoint
# Install gdown first: pip install gdown
gdown https://drive.google.com/uc?id=1cUv_reLE6k3604or78EranS7XzuVMWeO -O checkpoints/e4e_ffhq_encode.pt

# Run notebook or Python script
jupyter notebook combo_04_colab.ipynb
```

---

## Configuration

Edit `config.yaml` to customize:

```yaml
# Combo 04 Configuration
experiment_name: "combo_04_wplus_lpips_encoder_300"

# Model settings
image_size: 1024
latent_space: "W+"
num_layers: 18  # For 1024×1024 StyleGAN2

# Optimization
loss_type: "lpips"
lpips_net: "alex"
steps: 300
learning_rate: 0.01
init_method: "encoder"

# Encoder settings
encoder_checkpoint: "checkpoints/e4e_ffhq_encode.pt"
```

---

## How It Works

### 1. Encoder Initialization
The e4e encoder predicts an initial latent code from the target image:
```
Image (1024×1024) → e4e Encoder → W+ code [1, 18, 512]
```

### 2. Optimization Refinement
The initial code is refined using our Adam optimizer:
```
for step in range(300):
    generated = decoder(latent)
    loss = LPIPS(generated, target)
    loss.backward()
    optimizer.step()
```

### 3. Comparison
- **Combos 1-3:** Start from random/mean → slower convergence
- **Combo 4:** Start from encoder prediction → faster convergence, better results

---

## Expected Results

- **Initial (encoder):** Good quality reconstruction (LPIPS ~0.15-0.25)
- **After optimization:** Improved quality (LPIPS ~0.10-0.20)
- **Comparison:** Encoder init typically outperforms random/mean init

---

## Troubleshooting

### CUDA Compilation Errors
- **Solution:** Use Google Colab (has CUDA pre-installed)
- Or install CUDA Toolkit locally

### Out of Memory (OOM)
- **Solution:** Reduce batch size or use Colab Pro (more VRAM)

### Checkpoint Not Found
- **Solution:** Run the download cell or manually download:
  ```bash
  gdown https://drive.google.com/uc?id=1cUv_reLE6k3604or78EranS7XzuVMWeO -O checkpoints/e4e_ffhq_encode.pt
  ```

---

## References

- **e4e Paper:** [Tov et al., "Designing an Encoder for StyleGAN Image Manipulation" (2021)](https://arxiv.org/abs/2102.02766)
- **e4e Repository:** https://github.com/omertov/encoder4editing
- **StyleGAN2:** https://github.com/NVlabs/stylegan2

---

## File Structure

```
combo_04/
├── README.md                      # This file
├── combo_04_colab.ipynb          # Main Colab notebook
├── config.yaml                    # Configuration
├── encoder4editing/               # Cloned e4e repo (auto-generated)
└── checkpoints/                   # Downloaded checkpoints (auto-generated)
    └── e4e_ffhq_encode.pt
```

---

## Notes

- This combo is **independent** from the main project
- Uses e4e's encoder + decoder **directly** (no wrappers)
- Our optimization loop + configs + visualization
- Results are directly comparable to Combos 1-3 (different resolution though)

