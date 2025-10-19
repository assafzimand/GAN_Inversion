# PRD — Optimization-Based GAN Inversion (StyleGAN2)

## 1) Summary
We implement **optimization-based GAN inversion** on a pretrained **StyleGAN2**.  
Given a real image \(x\), we find a latent code \(z^*\) so that the generated image \(G(z^*)\) reconstructs \(x\) by minimizing a chosen loss:
\[
z^* = \arg\min_z \ell(G(z), x)
\]
We compare **loss functions** (L2 vs LPIPS), **latent spaces** (W vs W+), and **initializations** (mean\_w vs encoder-based), with a staged “start small” sequence (see §6).

## 2) Goals & Non-Goals
**Goals**
- Minimal, clean, modular code to invert images with StyleGAN2 (pretrained).
- CLI to run inversion on a single image or folder.
- Side-by-side visualizations and convergence plots.
- Reproducible experiments with configs and seeds.
- Quantitative metrics: **PSNR, SSIM, LPIPS**.

**Non-Goals**
- Training a GAN or encoder from scratch.
- Covering every inversion method; we focus on optimization-based, plus one hybrid (encoder init) in the final stage.
- Heavy hyperparameter tuning; rely on sensible defaults.

## 3) Core Features (Functional Requirements)
- Load pretrained StyleGAN2-FFHQ weights.
- Latent spaces: **W** and **W+**.
- Losses: **L2** (pixel) and **LPIPS** (perceptual).
- Inits: **mean\_w**; **encoder (e4e/pSp)** only in the final stage.
- Optimizer: **Adam** (default); optional **L-BFGS** (nice-to-have).
- CLI: `invert.py` (input image/folder, config overrides, output dir).
- Outputs per run:
  - Reconstruction images
  - Difference panels
  - `metrics.json` (PSNR/SSIM/LPIPS per image)
  - `loss_curve.png` + raw CSV/JSON of loss per step

## 4) Non-Functional Requirements
- **Reproducibility**: global seeds; all params in configs.
- **Modularity**: losses/metrics/engine/utils separated.
- **Logging**: progress every N steps; runtime reporting.
- **Runtime**: single GPU; up to ~1000 steps per image (demo-scale).
- **Docs**: README with install, quickstart, experiments, troubleshooting.
- **Tests**: unit tests for losses, metrics, and a toy inverter convergence test.

## 5) Data & Weights
- **Weights**: Pretrained StyleGAN2-FFHQ (PyTorch).
- **Data**: Small sample set (FFHQ/CelebA-HQ-like) + ability to run on arbitrary images.
- **Storage**: Keep large binaries out of Git (gitignore). Place under `checkpoints/` and `data/`.

## 6) Staged Combos & Order (Start Small)
We will run exactly four combos, **in this order**, changing one main factor at a time:

1) **W • L2 • Adam • mean\_w • ~300 steps** — cheapest sanity check.  
2) **W+ • L2 • Adam • mean\_w • ~300 steps** — flip latent space only.  
3) **W+ • LPIPS • Adam • mean\_w • ~600 steps** — change loss to perceptual.  
4) **W+ • LPIPS • Adam • encoder-init (e4e) • ~200–300 steps** — finally add encoder init to reduce steps.

Each combo should produce: reconstructions, diff panels, metrics, and loss curves.

## 7) Experiments (Acceptance)
- **Exp-01**: Run combo (1) on N≈3–5 images. Verify pipeline, metrics, loss curves.  
- **Exp-02**: Run combo (2) on the same set; compare metrics vs (1).  
- **Exp-03**: Run combo (3); compare LPIPS/visuals vs (2).  
- **Exp-04**: Run combo (4); confirm similar/better quality with fewer steps vs (3).

## 8) Success Criteria
- Code runs end-to-end from CLI with clear outputs.
- All unit tests pass locally.
- Reproducible results using provided configs and seeds.
- Visuals/metrics clearly show trends between (1)–(4).

## 9) Risks & Mitigations
- **LPIPS install/device issues** → pin versions, test CPU/GPU; provide fallbacks.
- **Weights availability** → script to download; document manual alternative.
- **Slow convergence** → allow steps config; show intermediate reconstructions.

## 10) Deliverables
- Source code + tests + configs.
- README and `PRD.md`.
- Outputs for Exp-01..04 (tables/plots/images).
