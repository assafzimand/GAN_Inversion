"""Quick integration test for HuggingFace loading with StyleGAN2Wrapper"""
import torch
import sys

print("="*60)
print("Testing HuggingFace + StyleGAN2Wrapper Integration")
print("="*60)

# Suppress debug logging
import logging
logging.basicConfig(level=logging.INFO)

from models.stylegan2_loader import load_generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n[1/4] Loading generator from HuggingFace...")
try:
    G = load_generator("hf://hajar001/stylegan2-ffhq-128", device)
    print(f"✓ Generator loaded: {type(G).__name__}")
    print(f"  - latent_dim: {G.latent_dim}")
    print(f"  - num_layers: {G.num_layers}")
except Exception as e:
    print(f"✗ Loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[2/4] Testing W space generation...")
try:
    z = torch.randn(2, 512, device=device)
    img_w = G.generate_from_z(z, latent_space='W')
    print(f"✓ W space works: {z.shape} → {img_w.shape}")
    if img_w.shape != (2, 3, 128, 128):
        print(f"✗ Wrong output shape! Expected (2, 3, 128, 128)")
        sys.exit(1)
except Exception as e:
    print(f"✗ W space failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[3/4] Testing W+ space generation...")
try:
    img_wplus = G.generate_from_z(z, latent_space='W+')
    print(f"✓ W+ space works: {z.shape} → {img_wplus.shape}")
    if img_wplus.shape != (2, 3, 128, 128):
        print(f"✗ Wrong output shape! Expected (2, 3, 128, 128)")
        sys.exit(1)
except Exception as e:
    print(f"✗ W+ space failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[4/4] Testing gradient flow...")
try:
    w = torch.randn(1, 512, device=device, requires_grad=True)
    img = G(w, latent_space='W')
    loss = img.mean()
    loss.backward()
    
    if w.grad is not None and w.grad.abs().sum() > 0:
        print(f"✓ Gradients flow correctly (grad norm: {w.grad.norm().item():.6f})")
    else:
        print(f"✗ Gradients not flowing!")
        sys.exit(1)
except Exception as e:
    print(f"✗ Gradient test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL INTEGRATION TESTS PASSED!")
print("="*60)
print("\nHuggingFace model is fully integrated:")
print(f"  - Load via: load_generator('hf://hajar001/stylegan2-ffhq-128', device)")
print(f"  - W space: ✓")
print(f"  - W+ space: ✓")
print(f"  - Gradients: ✓")
print(f"  - Output: 128×128 images")

