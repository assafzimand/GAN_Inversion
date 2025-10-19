"""
Test script to verify W/W+ support for GAN inversion.
This checks that the model structure supports our optimization workflow.
"""
import torch
from huggingface_hub import hf_hub_download
import sys
import os

print("="*60)
print("Testing W/W+ Support for GAN Inversion")
print("="*60)

# Load model
print("\n[1/6] Loading model...")
model_file = hf_hub_download(
    repo_id="hajar001/stylegan2-ffhq-128",
    filename="style_gan.py"
)
sys.path.insert(0, os.path.dirname(model_file))
from style_gan import StyleGAN

model = StyleGAN.from_pretrained("hajar001/stylegan2-ffhq-128")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"✓ Model loaded on {device}")

# Test 1: Check for mapping/synthesis separation
print("\n[2/6] Checking architecture components...")
has_mapping = hasattr(model, 'mapping')
has_synthesis = hasattr(model, 'synthesis')
print(f"  model.mapping: {'✓ Found' if has_mapping else '✗ Missing'}")
print(f"  model.synthesis: {'✓ Found' if has_synthesis else '✗ Missing'}")

if not (has_mapping and has_synthesis):
    print("✗ Model missing required components!")
    sys.exit(1)

# Get number of layers
num_layers = model.synthesis.num_layers
print(f"  Number of synthesis layers: {num_layers}")

# Test 2: Z → W mapping
print("\n[3/6] Testing Z → W mapping...")
z = torch.randn(2, 512, device=device)
try:
    with torch.no_grad():
        w = model.mapping(z)
    print(f"  ✓ Z shape: {z.shape}")
    print(f"  ✓ W shape: {w.shape}")
    if w.shape == (2, 512):
        print(f"  ✓ W space works (B, 512)")
    else:
        print(f"  ✗ Unexpected W shape!")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Z → W failed: {e}")
    sys.exit(1)

# Test 3: W → Image (broadcast to W+)
print("\n[4/6] Testing W → Image generation...")
try:
    # Need to expand W to W+ format for synthesis
    w_expanded = w.unsqueeze(1).expand(-1, num_layers, -1)
    print(f"  W+ shape (expanded): {w_expanded.shape}")
    
    with torch.no_grad():
        img_from_w = model.synthesis(w_expanded)
    print(f"  ✓ Image shape: {img_from_w.shape}")
    if img_from_w.shape == (2, 3, 128, 128):
        print(f"  ✓ W space → Image works")
    else:
        print(f"  ✗ Unexpected image shape!")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ W → Image failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: W+ (per-layer) support
print("\n[5/6] Testing W+ (per-layer) space...")
try:
    # Create W+ latent (different W for each layer)
    w_plus = torch.randn(2, num_layers, 512, device=device)
    print(f"  W+ shape: {w_plus.shape}")
    
    with torch.no_grad():
        img_from_wplus = model.synthesis(w_plus)
    print(f"  ✓ Image shape: {img_from_wplus.shape}")
    if img_from_wplus.shape == (2, 3, 128, 128):
        print(f"  ✓ W+ space → Image works")
    else:
        print(f"  ✗ Unexpected image shape!")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ W+ → Image failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Gradient flow (CRITICAL for optimization)
print("\n[6/6] Testing gradient flow for optimization...")
try:
    # Test W space gradients
    w_opt = torch.randn(1, 512, device=device, requires_grad=True)
    w_opt_expanded = w_opt.unsqueeze(1).expand(-1, num_layers, -1)
    
    img = model.synthesis(w_opt_expanded)
    loss = img.mean()  # Dummy loss
    loss.backward()
    
    if w_opt.grad is not None and w_opt.grad.abs().sum() > 0:
        print(f"  ✓ W space gradients flow (grad norm: {w_opt.grad.norm().item():.6f})")
    else:
        print(f"  ✗ W space gradients not flowing!")
        sys.exit(1)
    
    # Test W+ space gradients
    w_plus_opt = torch.randn(1, num_layers, 512, device=device, requires_grad=True)
    img2 = model.synthesis(w_plus_opt)
    loss2 = img2.mean()
    loss2.backward()
    
    if w_plus_opt.grad is not None and w_plus_opt.grad.abs().sum() > 0:
        print(f"  ✓ W+ space gradients flow (grad norm: {w_plus_opt.grad.norm().item():.6f})")
    else:
        print(f"  ✗ W+ space gradients not flowing!")
        sys.exit(1)
    
except Exception as e:
    print(f"  ✗ Gradient test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL INVERSION REQUIREMENTS VERIFIED!")
print("="*60)
print("\nSummary:")
print(f"  ✓ model.mapping: Z ({z.shape}) → W ({w.shape})")
print(f"  ✓ model.synthesis: W+ ({w_expanded.shape}) → Image ({img_from_w.shape})")
print(f"  ✓ W space: Supported (broadcast to {num_layers} layers)")
print(f"  ✓ W+ space: Supported (per-layer control)")
print(f"  ✓ Gradients: Flow correctly for optimization")
print(f"\n  Model is READY for GAN inversion!")

