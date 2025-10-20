"""Inspect e4e checkpoint to understand decoder structure"""
import torch

ckpt = torch.load('checkpoints/e4e_ffhq_encode.pt', map_location='cpu')

print("=== E4E CHECKPOINT STRUCTURE ===\n")
print(f"Top-level keys: {list(ckpt.keys())}\n")

print("=== STATE_DICT KEYS ===")
print(f"Total parameters: {len(ckpt['state_dict'])}\n")

# Categorize keys
encoder_keys = [k for k in ckpt['state_dict'].keys() if k.startswith('encoder.')]
decoder_keys = [k for k in ckpt['state_dict'].keys() if k.startswith('decoder.')]
other_keys = [k for k in ckpt['state_dict'].keys() if not k.startswith('encoder.') and not k.startswith('decoder.')]

print(f"Encoder keys: {len(encoder_keys)}")
print(f"  First 3: {encoder_keys[:3]}")

print(f"\nDecoder keys: {len(decoder_keys)}")
print(f"  First 10: {decoder_keys[:10]}")

print(f"\nOther keys: {len(other_keys)}")
if other_keys:
    print(f"  All: {other_keys}")

# Check decoder structure
print("\n=== DECODER STRUCTURE ===")
decoder_structure = {}
for key in decoder_keys[:20]:  # Sample first 20
    parts = key.split('.')
    if len(parts) >= 2:
        component = parts[1]  # After 'decoder.'
        if component not in decoder_structure:
            decoder_structure[component] = 0
        decoder_structure[component] += 1

print("Decoder components:")
for comp, count in decoder_structure.items():
    print(f"  {comp}: {count} parameters")

print("\n=== OPTS ===")
opts = ckpt['opts']
if hasattr(opts, '__dict__'):
    print(f"Opts type: {type(opts)}")
    print(f"Opts attributes: {list(opts.__dict__.keys())[:10]}")
    if hasattr(opts, 'stylegan_size'):
        print(f"  stylegan_size: {opts.stylegan_size}")
    if hasattr(opts, 'encoder_type'):
        print(f"  encoder_type: {opts.encoder_type}")
else:
    print(f"Opts type: {type(opts)}")
    print(f"Opts keys: {list(opts.keys())[:10]}")

print("\n=== LATENT_AVG ===")
print(f"latent_avg shape: {ckpt['latent_avg'].shape}")
print(f"latent_avg mean: {ckpt['latent_avg'].mean():.6f}")

