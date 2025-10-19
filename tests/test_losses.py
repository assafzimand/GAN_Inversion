"""
Unit tests for loss functions.

Tests L2 and LPIPS losses with deterministic checks on small tensors.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from losses.l2 import L2Loss
from losses.lpips_loss import LPIPSLoss, LPIPS_AVAILABLE


class TestL2Loss:
    """Test cases for L2 pixel loss."""

    def test_identical_images(self):
        """Loss should be 0 for identical images."""
        loss_fn = L2Loss(reduction="mean")
        # Generate images in [-1, 1] range
        img = torch.randn(2, 3, 64, 64).clamp(-1, 1)
        loss = loss_fn(img, img)
        assert loss.item() == 0.0, "Loss should be exactly 0 for identical images"

    def test_known_values(self):
        """Test with known MSE values."""
        loss_fn = L2Loss(reduction="mean")

        # Simple case: all zeros vs all ones (scaled to [-1, 1])
        generated = torch.ones(1, 3, 2, 2) * 1.0   # All 1s
        target = torch.ones(1, 3, 2, 2) * -1.0      # All -1s

        loss = loss_fn(generated, target)
        expected = ((1.0 - (-1.0)) ** 2)  # (2.0)^2 = 4.0
        assert abs(loss.item() - expected) < 1e-6, f"Expected {expected}, got {loss.item()}"

    def test_shape_validation(self):
        """Should raise error for mismatched shapes."""
        loss_fn = L2Loss()
        img1 = torch.randn(1, 3, 64, 64)
        img2 = torch.randn(1, 3, 32, 32)  # Different size

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(img1, img2)

    def test_dimension_validation(self):
        """Should raise error for wrong number of dimensions."""
        loss_fn = L2Loss()
        img1 = torch.randn(3, 64, 64)  # 3D instead of 4D
        img2 = torch.randn(3, 64, 64)

        with pytest.raises(ValueError, match="Expected 4D tensor"):
            loss_fn(img1, img2)

    def test_range_validation(self):
        """Should raise error for values outside expected range."""
        loss_fn = L2Loss(validate_range=True)
        
        # Values way outside [-1, 1]
        generated = torch.randn(1, 3, 8, 8) * 10.0  # Large values
        target = torch.randn(1, 3, 8, 8)

        with pytest.raises(ValueError, match="out of expected range"):
            loss_fn(generated, target)

    def test_range_validation_disabled(self):
        """Should not raise error when validation is disabled."""
        loss_fn = L2Loss(validate_range=False)
        
        # Values outside [-1, 1] but validation disabled
        generated = torch.randn(1, 3, 8, 8) * 10.0
        target = torch.randn(1, 3, 8, 8) * 10.0

        # Should not raise
        loss = loss_fn(generated, target)
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_batch_processing(self):
        """Should handle batched inputs correctly."""
        loss_fn = L2Loss(reduction="mean")
        
        # Batch of 4 images in [-1, 1] range
        generated = torch.randn(4, 3, 32, 32).clamp(-1, 1)
        target = torch.randn(4, 3, 32, 32).clamp(-1, 1)

        loss = loss_fn(generated, target)
        assert loss.ndim == 0, "Should return scalar for reduction='mean'"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_reduction_modes(self):
        """Test different reduction modes."""
        # Generate images in [-1, 1] range
        img1 = torch.randn(2, 3, 16, 16).clamp(-1, 1)
        img2 = torch.randn(2, 3, 16, 16).clamp(-1, 1)

        # Mean reduction
        loss_mean = L2Loss(reduction="mean")
        result_mean = loss_mean(img1, img2)
        assert result_mean.ndim == 0, "Mean reduction should return scalar"

        # Sum reduction
        loss_sum = L2Loss(reduction="sum")
        result_sum = loss_sum(img1, img2)
        assert result_sum.ndim == 0, "Sum reduction should return scalar"
        assert result_sum > result_mean, "Sum should be larger than mean"

        # None reduction (per-sample)
        loss_none = L2Loss(reduction="none")
        result_none = loss_none(img1, img2)
        assert result_none.shape == (2,), "None reduction should return per-sample loss"

    def test_invalid_reduction(self):
        """Should raise error for invalid reduction."""
        with pytest.raises(ValueError, match="reduction must be"):
            L2Loss(reduction="invalid")

    def test_gradient_flow(self):
        """Gradients should flow through the loss."""
        loss_fn = L2Loss(reduction="mean")
        
        # Generate images in [-1, 1] range as leaf tensors
        generated = torch.rand(1, 3, 16, 16) * 2 - 1
        generated.requires_grad = True
        target = torch.rand(1, 3, 16, 16) * 2 - 1

        loss = loss_fn(generated, target)
        loss.backward()

        assert generated.grad is not None, "Gradients should be computed"
        assert generated.grad.shape == generated.shape, "Gradient shape should match input"


@pytest.mark.skipif(not LPIPS_AVAILABLE, reason="lpips package not installed")
class TestLPIPSLoss:
    """Test cases for LPIPS perceptual loss."""

    def test_identical_images(self):
        """Loss should be near 0 for identical images."""
        loss_fn = LPIPSLoss(net="alex", device=torch.device("cpu"))
        
        # Use small images for speed, in [-1, 1] range
        img = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        loss = loss_fn(img, img)
        
        assert loss.item() < 1e-6, f"Loss should be near 0 for identical images, got {loss.item()}"

    def test_different_images(self):
        """Loss should be positive for different images."""
        loss_fn = LPIPSLoss(net="alex", device=torch.device("cpu"))
        
        # Generate images in [-1, 1] range
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        
        loss = loss_fn(img1, img2)
        assert loss.item() > 0, "Loss should be positive for different images"

    def test_cpu_execution(self):
        """Should work on CPU."""
        loss_fn = LPIPSLoss(net="alex", device=torch.device("cpu"))
        
        # Generate images in [-1, 1] range
        img1 = torch.randn(2, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(2, 3, 64, 64).clamp(-1, 1)
        
        loss = loss_fn(img1, img2)
        assert loss.device.type == "cpu", "Loss should be on CPU"
        assert loss.item() >= 0, "Loss should be non-negative"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_execution(self):
        """Should work on GPU."""
        device = torch.device("cuda")
        loss_fn = LPIPSLoss(net="alex", device=device)
        
        # Generate images in [-1, 1] range
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1).cuda()
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1).cuda()
        
        loss = loss_fn(img1, img2)
        assert loss.device.type == "cuda", "Loss should be on CUDA"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_shape_validation(self):
        """Should raise error for mismatched shapes."""
        loss_fn = LPIPSLoss(net="alex", device=torch.device("cpu"))
        
        img1 = torch.randn(1, 3, 64, 64)
        img2 = torch.randn(1, 3, 32, 32)  # Different size

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(img1, img2)

    def test_channel_validation(self):
        """Should raise error for non-RGB images."""
        loss_fn = LPIPSLoss(net="alex", device=torch.device("cpu"))
        
        img1 = torch.randn(1, 1, 64, 64)  # Grayscale
        img2 = torch.randn(1, 1, 64, 64)

        with pytest.raises(ValueError, match="Expected 3 channels"):
            loss_fn(img1, img2)

    def test_range_validation(self):
        """Should raise error for values outside expected range."""
        loss_fn = LPIPSLoss(net="alex", device=torch.device("cpu"))
        
        # Values way outside [-1, 1]
        img1 = torch.randn(1, 3, 64, 64) * 10.0
        img2 = torch.randn(1, 3, 64, 64)

        with pytest.raises(ValueError, match="out of expected range"):
            loss_fn(img1, img2)

    def test_batch_processing(self):
        """Should handle batched inputs correctly."""
        loss_fn = LPIPSLoss(net="alex", device=torch.device("cpu"))
        
        # Batch of 3 images in [-1, 1] range
        img1 = torch.randn(3, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(3, 3, 64, 64).clamp(-1, 1)

        loss = loss_fn(img1, img2)
        assert loss.ndim == 0, "Should return scalar (averaged over batch)"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_invalid_net(self):
        """Should raise error for invalid network."""
        with pytest.raises(ValueError, match="net must be"):
            LPIPSLoss(net="invalid", device=torch.device("cpu"))

    def test_spatial_mode(self):
        """Test spatial loss map mode."""
        loss_fn = LPIPSLoss(net="alex", device=torch.device("cpu"), spatial=True)
        
        # Generate images in [-1, 1] range
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)

        loss_map = loss_fn(img1, img2)
        # Spatial mode returns per-pixel loss map [B, 1, H, W]
        assert loss_map.ndim == 4, "Spatial mode should return 4D tensor"
        assert loss_map.shape[0] == 1, "Batch size should be 1"
        assert loss_map.shape[1] == 1, "Should have 1 channel"

    def test_determinism(self):
        """LPIPS should be deterministic for same inputs."""
        loss_fn = LPIPSLoss(net="alex", device=torch.device("cpu"))
        
        # Generate images in [-1, 1] range
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)

        loss1 = loss_fn(img1, img2)
        loss2 = loss_fn(img1, img2)

        assert abs(loss1.item() - loss2.item()) < 1e-6, "LPIPS should be deterministic"


@pytest.mark.skipif(not LPIPS_AVAILABLE, reason="lpips package not installed")
class TestLPIPSNetworks:
    """Test different LPIPS network backbones."""

    @pytest.mark.parametrize("net", ["alex", "vgg", "squeeze"])
    def test_different_networks(self, net):
        """Test that all network backbones work."""
        loss_fn = LPIPSLoss(net=net, device=torch.device("cpu"))
        
        # Generate images in [-1, 1] range
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)

        loss = loss_fn(img1, img2)
        assert loss.item() >= 0, f"Loss with {net} should be non-negative"

