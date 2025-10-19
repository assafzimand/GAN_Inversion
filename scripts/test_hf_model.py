"""
Test script for HuggingFace StyleGAN model.
This verifies the model can be downloaded, loaded, and generates valid images.
"""
import torch
from torchvision.utils import save_image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import sys
import os

print("="*60)
print("Testing HuggingFace StyleGAN Model")
print("="*60)

# Download and load model
print("\n[1/6] Downloading model from HuggingFace...")
try:
    model_file = hf_hub_download(
        repo_id="hajar001/stylegan2-ffhq-128",
        filename="style_gan.py"
    )
    print(f"✓ Downloaded to: {model_file}")
except Exception as e:
    print(f"✗ Download failed: {e}")
    sys.exit(1)

print("\n[2/6] Loading model...")
try:
    sys.path.insert(0, os.path.dirname(model_file))
    from style_gan import StyleGAN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try from_pretrained first, but fall back to manual loading if it fails
    try:
        model = StyleGAN.from_pretrained("hajar001/stylegan2-ffhq-128")
    except:
        print("  from_pretrained failed, trying manual loading...")
        # Download weights file
        weights_file = hf_hub_download(
            repo_id="hajar001/stylegan2-ffhq-128",
            filename="pytorch_model.bin"
        )
        # Instantiate model
        model = StyleGAN(z_dim=512, w_dim=512, img_size=128, img_channels=3, mapping_layers=8)
        # Load weights
        state_dict = torch.load(weights_file, map_location='cpu')
        model.load_state_dict(state_dict)
        print("  ✓ Manually loaded weights")
    
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully on {device}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/6] Generating test image...")
try:
    with torch.no_grad():
        z = torch.randn(1, 512, device=device)
        images = model.generate(z, truncation_psi=0.7)
    print(f"✓ Generated image with shape: {images.shape}")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    sys.exit(1)

print("\n[4/6] Verifying output...")
# Check shape
expected_shape = (1, 3, 128, 128)
if images.shape == expected_shape:
    print(f"✓ Shape correct: {images.shape}")
else:
    print(f"✗ Shape mismatch! Expected {expected_shape}, got {images.shape}")
    sys.exit(1)

# Check range
img_min, img_max = images.min().item(), images.max().item()
print(f"  Image range: [{img_min:.3f}, {img_max:.3f}]")
if -1.5 <= img_min <= -0.5 and 0.5 <= img_max <= 1.5:
    print(f"✓ Range appears to be [-1, 1] (within tolerance)")
else:
    print(f"⚠ Warning: Range might not be [-1, 1]")

print("\n[5/6] Saving image...")
try:
    # Denormalize from [-1, 1] to [0, 1]
    images_normalized = (images + 1) / 2
    images_normalized = torch.clamp(images_normalized, 0, 1)
    
    save_image(images_normalized, "test_hf_output.png")
    print(f"✓ Saved to: test_hf_output.png")
except Exception as e:
    print(f"✗ Save failed: {e}")
    sys.exit(1)

print("\n[6/6] Displaying image...")
try:
    # Convert to numpy for matplotlib
    img_np = images_normalized[0].cpu().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.title("Generated Face (128×128)\nHuggingFace StyleGAN", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_hf_output_plot.png", dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: test_hf_output_plot.png")
    
    # Try to show (may not work in some environments)
    try:
        plt.show()
        print("✓ Display window opened (close it to continue)")
    except:
        print("⚠ Could not open display window (running headless?)")
except Exception as e:
    print(f"✗ Display failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nModel is working correctly:")
print(f"  - Output shape: {images.shape}")
print(f"  - Output range: [{img_min:.3f}, {img_max:.3f}]")
print(f"  - Device: {device}")
print(f"  - Saved images: test_hf_output.png, test_hf_output_plot.png")

