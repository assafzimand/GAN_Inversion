"""
Unit tests for inversion engine.

Tests initialization strategies, optimization loop convergence,
and history tracking using mock generators.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

from engine.inverter import run_inversion, initialize_latent


class MockStyleGAN2Wrapper(nn.Module):
    """Mock StyleGAN2 wrapper for testing inversion."""
    
    def __init__(self, latent_dim=512, num_layers=18, image_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.image_size = image_size
        
        # Simple conv network to generate images from latents
        # This creates a trainable mapping that can be optimized
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, latent, latent_space='W'):
        """
        Forward pass mimicking StyleGAN2Wrapper interface.
        
        Args:
            latent: [B, 512] or [B, num_layers, 512]
            latent_space: 'W' or 'W+'
        
        Returns:
            Generated image [B, 3, H, W] in [-1, 1]
        """
        # Handle both W and W+ inputs
        if latent.dim() == 2:
            # [B, 512]
            x = latent
        elif latent.dim() == 3:
            # [B, num_layers, 512] -> average to [B, 512]
            x = latent.mean(dim=1)
        else:
            raise ValueError(f"Unexpected latent shape: {latent.shape}")
        
        # Generate image
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        x = torch.relu(x)
        
        x = self.upsample(x)  # 8x8
        x = torch.relu(self.conv1(x))
        
        x = self.upsample(x)  # 16x16
        x = torch.relu(self.conv2(x))
        
        x = self.upsample(x)  # 32x32
        x = self.upsample(x)  # 64x64
        x = self.upsample(x)  # 128x128
        x = torch.tanh(self.conv3(x))  # Output: 128x128
        
        return x
    
    def style(self, z):
        """Mock style/mapping network."""
        # Simple identity mapping for testing
        return z
    
    def synthesis(self, w, **kwargs):
        """Mock synthesis network."""
        return self.forward(w, latent_space='W')


class TestLatentInitialization:
    """Test cases for latent initialization."""
    
    def test_mean_w_init_with_provided_mean(self):
        """Test mean_w initialization with precomputed mean."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        mean_w = torch.randn(1, 512)
        
        latent = initialize_latent(
            generator=generator,
            latent_space='W',
            init_method='mean_w',
            device=device,
            mean_w=mean_w
        )
        
        assert latent.shape == torch.Size([1, 512])
        assert torch.allclose(latent, mean_w)
    
    def test_mean_w_init_without_provided_mean(self):
        """Test mean_w initialization computes mean_w if not provided."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        
        config = {
            'mean_w_num_samples': 100,
            'mean_w_batch_size': 50
        }
        
        latent = initialize_latent(
            generator=generator,
            latent_space='W',
            init_method='mean_w',
            device=device,
            mean_w=None,
            config=config
        )
        
        assert latent.shape == torch.Size([1, 512])
        assert latent.requires_grad is False  # Should not have grad by default
    
    def test_random_init(self):
        """Test random initialization."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        
        config = {
            'random_w_std': 1.0,
            'random_w_mean': 0.0
        }
        
        latent = initialize_latent(
            generator=generator,
            latent_space='W',
            init_method='random',
            device=device,
            config=config
        )
        
        assert latent.shape == torch.Size([1, 512])
        # Check that values are approximately normal distributed
        assert latent.mean().abs() < 0.5  # Loose check for mean near 0
        assert 0.5 < latent.std() < 2.0   # Loose check for std near 1
    
    def test_random_init_custom_params(self):
        """Test random initialization with custom mean and std."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        
        config = {
            'random_w_std': 2.0,
            'random_w_mean': 5.0
        }
        
        latent = initialize_latent(
            generator=generator,
            latent_space='W',
            init_method='random',
            device=device,
            config=config
        )
        
        assert latent.shape == torch.Size([1, 512])
        # Values should be roughly around mean=5, std=2
        assert 3.0 < latent.mean() < 7.0
    
    def test_w_plus_space_init(self):
        """Test that W+ space initialization returns correct shape."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        mean_w = torch.randn(1, 512)
        
        # For W+, initialize_latent still returns [1, 512]
        # StyleGAN2Wrapper handles broadcasting
        latent = initialize_latent(
            generator=generator,
            latent_space='W+',
            init_method='mean_w',
            device=device,
            mean_w=mean_w
        )
        
        assert latent.shape == torch.Size([1, 512])
    
    def test_encoder_init_not_implemented(self):
        """Test that encoder initialization raises NotImplementedError."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        
        with pytest.raises(NotImplementedError, match="Encoder initialization not yet implemented"):
            initialize_latent(
                generator=generator,
                latent_space='W',
                init_method='encoder',
                device=device
            )
    
    def test_invalid_init_method(self):
        """Test that invalid init method raises ValueError."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        
        with pytest.raises(ValueError, match="Unknown init_method"):
            initialize_latent(
                generator=generator,
                latent_space='W',
                init_method='invalid_method',
                device=device
            )


class TestInversionEngine:
    """Test cases for inversion engine."""
    
    def test_basic_inversion_l2(self):
        """Test basic inversion with L2 loss."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        generator.eval()
        
        # Create a target image
        target = (torch.rand(1, 3, 128, 128) * 2 - 1) * 0.8  # [-0.8, 0.8]
        
        config = {
            'latent_space': 'W',
            'init_method': 'random',
            'loss_type': 'l2',
            'steps': 50,
            'learning_rate': 0.01,
            'log_interval': 25
        }
        
        z_star, recon, history = run_inversion(generator, target, config)
        
        # Check outputs
        assert z_star.shape == torch.Size([1, 512])
        assert recon.shape == target.shape
        assert len(history['loss']) == 50
        assert history['steps'] == 50
        assert history['time'] > 0
        assert history['early_stopped'] is False
    
    def test_loss_decreases(self):
        """Test that loss decreases over optimization steps."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        
        # Create a target that's generated from the model
        # This ensures we can actually optimize to it
        with torch.no_grad():
            z_true = torch.randn(1, 512) * 0.5
            target = generator(z_true, latent_space='W')
        
        config = {
            'latent_space': 'W',
            'init_method': 'random',
            'loss_type': 'l2',
            'steps': 100,
            'learning_rate': 0.05,
            'log_interval': 50
        }
        
        z_star, recon, history = run_inversion(generator, target, config)
        
        # Loss should decrease
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"
        
        # Loss should converge to near zero for this simple case
        assert final_loss < 0.1, f"Final loss too high: {final_loss:.6f}"
    
    def test_inversion_w_plus_space(self):
        """Test inversion in W+ space."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        target = (torch.rand(1, 3, 128, 128) * 2 - 1) * 0.8  # [-0.8, 0.8]
        
        config = {
            'latent_space': 'W+',
            'init_method': 'random',
            'loss_type': 'l2',
            'steps': 50,
            'learning_rate': 0.01,
            'log_interval': 25
        }
        
        z_star, recon, history = run_inversion(generator, target, config)
        
        # For W+, latent stays [1, 512] (StyleGAN2Wrapper handles broadcast)
        assert z_star.shape == torch.Size([1, 512])
        assert recon.shape == target.shape
    
    def test_inversion_lpips_loss(self):
        """Test inversion with LPIPS loss."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        target = torch.randn(1, 3, 128, 128).clamp(-1, 1)
        
        config = {
            'latent_space': 'W',
            'init_method': 'random',
            'loss_type': 'lpips',
            'lpips_net': 'alex',
            'steps': 30,
            'learning_rate': 0.01,
            'log_interval': 15
        }
        
        z_star, recon, history = run_inversion(generator, target, config)
        
        assert z_star.shape == torch.Size([1, 512])
        assert recon.shape == target.shape
        assert len(history['loss']) == 30
    
    def test_mean_w_initialization(self):
        """Test inversion with mean_w initialization."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        target = (torch.rand(1, 3, 128, 128) * 2 - 1) * 0.8  # [-0.8, 0.8]
        mean_w = torch.zeros(1, 512)  # Use zero mean for determinism
        
        config = {
            'latent_space': 'W',
            'init_method': 'mean_w',
            'loss_type': 'l2',
            'steps': 50,
            'learning_rate': 0.01,
            'log_interval': 25
        }
        
        z_star, recon, history = run_inversion(generator, target, config, mean_w=mean_w)
        
        assert z_star.shape == torch.Size([1, 512])
        assert len(history['loss']) == 50
    
    def test_early_stopping(self):
        """Test that early stopping works when loss plateaus."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        
        # Use a target that converges quickly
        with torch.no_grad():
            z_true = torch.randn(1, 512) * 0.5
            target = generator(z_true, latent_space='W')
        
        config = {
            'latent_space': 'W',
            'init_method': 'random',
            'loss_type': 'l2',
            'steps': 1000,  # Set high to ensure early stop triggers
            'learning_rate': 0.1,
            'log_interval': 100,
            'enable_early_stop': True,
            'early_stop_patience': 10,
            'early_stop_threshold': 1e-5
        }
        
        z_star, recon, history = run_inversion(generator, target, config)
        
        # Should stop early
        assert history['steps'] < 1000, "Early stopping should have triggered"
        assert history['early_stopped'] is True
    
    def test_history_tracking(self):
        """Test that history is properly tracked."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        target = (torch.rand(1, 3, 128, 128) * 2 - 1) * 0.8  # [-0.8, 0.8]
        
        config = {
            'latent_space': 'W',
            'init_method': 'random',
            'loss_type': 'l2',
            'steps': 25,
            'learning_rate': 0.01,
            'log_interval': 10
        }
        
        z_star, recon, history = run_inversion(generator, target, config)
        
        assert 'loss' in history
        assert 'steps' in history
        assert 'time' in history
        assert 'early_stopped' in history
        
        assert len(history['loss']) == 25
        assert history['steps'] == 25
        assert isinstance(history['time'], float)
        assert history['time'] > 0
        
        # All losses should be finite
        for loss_val in history['loss']:
            assert not torch.isnan(torch.tensor(loss_val))
            assert not torch.isinf(torch.tensor(loss_val))
    
    def test_invalid_latent_space(self):
        """Test that invalid latent_space raises ValueError."""
        generator = MockStyleGAN2Wrapper()
        target = torch.randn(1, 3, 128, 128)
        
        config = {
            'latent_space': 'Z',  # Invalid
            'init_method': 'random',
            'loss_type': 'l2',
            'steps': 10
        }
        
        with pytest.raises(ValueError, match="latent_space must be 'W' or 'W\\+'"):
            run_inversion(generator, target, config)
    
    def test_invalid_init_method(self):
        """Test that invalid init_method raises ValueError."""
        generator = MockStyleGAN2Wrapper()
        target = torch.randn(1, 3, 128, 128)
        
        config = {
            'latent_space': 'W',
            'init_method': 'encoder',  # Not yet supported
            'loss_type': 'l2',
            'steps': 10
        }
        
        with pytest.raises(ValueError, match="encoder not yet supported"):
            run_inversion(generator, target, config)
    
    def test_invalid_loss_type(self):
        """Test that invalid loss_type raises ValueError."""
        generator = MockStyleGAN2Wrapper()
        target = torch.randn(1, 3, 128, 128)
        
        config = {
            'latent_space': 'W',
            'init_method': 'random',
            'loss_type': 'invalid_loss',
            'steps': 10
        }
        
        with pytest.raises(ValueError, match="loss_type must be 'l2' or 'lpips'"):
            run_inversion(generator, target, config)
    
    def test_target_shape_validation(self):
        """Test that target image shape is validated."""
        generator = MockStyleGAN2Wrapper()
        
        # 3D tensor (missing batch dimension)
        target_3d = torch.randn(3, 128, 128)
        config = {'latent_space': 'W', 'init_method': 'random', 'loss_type': 'l2', 'steps': 10}
        
        with pytest.raises(ValueError, match="target_image must be 4D"):
            run_inversion(generator, target_3d, config)
        
        # Batch size > 1
        target_batch = torch.randn(2, 3, 128, 128)
        
        with pytest.raises(ValueError, match="batch size must be 1"):
            run_inversion(generator, target_batch, config)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_execution(self):
        """Test that inversion works on GPU."""
        device = torch.device('cuda')
        generator = MockStyleGAN2Wrapper().to(device)
        target = ((torch.rand(1, 3, 128, 128) * 2 - 1) * 0.8).to(device)  # [-0.8, 0.8]
        
        config = {
            'latent_space': 'W',
            'init_method': 'random',
            'loss_type': 'l2',
            'steps': 30,
            'learning_rate': 0.01,
            'log_interval': 15
        }
        
        z_star, recon, history = run_inversion(generator, target, config)
        
        # All outputs should be on GPU
        assert z_star.device.type == 'cuda'
        assert recon.device.type == 'cuda'
        assert len(history['loss']) == 30
    
    def test_gradient_flow(self):
        """Test that gradients flow properly during optimization."""
        device = torch.device('cpu')
        generator = MockStyleGAN2Wrapper()
        target = (torch.rand(1, 3, 128, 128) * 2 - 1) * 0.8  # [-0.8, 0.8]
        
        config = {
            'latent_space': 'W',
            'init_method': 'random',
            'loss_type': 'l2',
            'steps': 5,
            'learning_rate': 0.01,
            'log_interval': 5
        }
        
        # This test just ensures no gradient errors occur
        z_star, recon, history = run_inversion(generator, target, config)
        
        # z_star should be detached (no grad after optimization)
        assert z_star.requires_grad is False
        assert recon.requires_grad is False

