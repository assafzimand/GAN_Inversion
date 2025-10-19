"""Check what the HF StyleGAN expects as input."""

import torch
import sys

# Load the model
from stylegan2_pytorch import StyleGAN
print("Loading model...")
generator = StyleGAN.from_pretrained("hajar001/stylegan2-ffhq-128")
generator.eval()

print(f"\nGenerator type: {type(generator)}")
print(f"Generator class: {generator.__class__.__name__}")

# Check if it has synthesis
if hasattr(generator, 'synthesis'):
    print(f"Has synthesis: Yes - {type(generator.synthesis)}")
else:
    print("Has synthesis: No")

# Check forward signature
import inspect
sig = inspect.signature(generator.forward)
print(f"\nForward signature: {sig}")

# Try calling with different inputs
print("\n" + "="*60)
print("Testing different input formats:")
print("="*60)

# Test 1: W space [1, 512]
print("\n1. W space [1, 512]:")
try:
    latent_w = torch.randn(1, 512)
    with torch.no_grad():
        img = generator(latent_w)
    print(f"✓ Success! Output shape: {img.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: W+ space [1, 18, 512]
print("\n2. W+ space [1, 18, 512]:")
try:
    latent_wplus = torch.randn(1, 18, 512)
    with torch.no_grad():
        img = generator(latent_wplus)
    print(f"✓ Success! Output shape: {img.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: W+ space [1, 12, 512] (128x128 has 12 layers, not 18)
print("\n3. W+ space [1, 12, 512]:")
try:
    latent_wplus = torch.randn(1, 12, 512)
    with torch.no_grad():
        img = generator(latent_wplus)
    print(f"✓ Success! Output shape: {img.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 4: Just styles parameter
print("\n4. With styles= keyword:")
try:
    latent_w = torch.randn(1, 512)
    with torch.no_grad():
        img = generator(styles=latent_w)
    print(f"✓ Success! Output shape: {img.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

