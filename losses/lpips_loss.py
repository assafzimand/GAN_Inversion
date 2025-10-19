"""
LPIPS (Learned Perceptual Image Patch Similarity) loss.

Wraps the `lpips` package for perceptual similarity loss using
pretrained deep features (AlexNet, VGG, or SqueezeNet).
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual loss using pretrained deep networks.

    Measures perceptual similarity between images using deep features.
    Lower values = more perceptually similar.

    Args:
        net: Network backbone - 'alex' (default), 'vgg', or 'squeeze'
        device: Device to place model on
        spatial: If True, return per-pixel loss map; if False, return scalar

    Example:
        >>> loss_fn = LPIPSLoss(net='alex', device='cuda')
        >>> generated = torch.randn(1, 3, 256, 256).cuda()
        >>> target = torch.randn(1, 3, 256, 256).cuda()
        >>> loss = loss_fn(generated, target)

    Note:
        LPIPS expects inputs in [-1, 1] range (ImageNet normalization applied internally).
    """

    def __init__(
        self,
        net: str = "alex",
        device: Optional[torch.device] = None,
        spatial: bool = False
    ):
        """
        Initialize LPIPS loss.

        Args:
            net: Network backbone ('alex', 'vgg', or 'squeeze')
            device: Device to place model on
            spatial: Return spatial loss map if True

        Raises:
            ImportError: If lpips package is not installed
            ValueError: If net is not valid
        """
        super().__init__()

        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips package not found. Install with: pip install lpips"
            )

        if net not in ["alex", "vgg", "squeeze"]:
            raise ValueError(
                f"net must be 'alex', 'vgg', or 'squeeze', got '{net}'"
            )

        self.net_name = net
        self.device = device or torch.device("cpu")
        self.spatial = spatial

        # Initialize LPIPS model
        # lpips.LPIPS automatically downloads pretrained weights on first use
        self.model = lpips.LPIPS(
            net=net,
            spatial=spatial,
            verbose=False
        ).to(self.device)

        # Set to eval mode (no training)
        self.model.eval()

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LPIPS loss between generated and target images.

        Args:
            generated: Generated images [B, C, H, W] in [-1, 1]
            target: Target images [B, C, H, W] in [-1, 1]

        Returns:
            Perceptual loss scalar (averaged over batch)
            or spatial loss map [B, 1, H, W] if spatial=True

        Raises:
            ValueError: If shapes don't match or values out of expected range
        """
        # Validate shapes
        if generated.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: generated {generated.shape} vs target {target.shape}"
            )

        # Validate dimensions
        if generated.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor [B, C, H, W], got shape {generated.shape}"
            )

        if generated.shape[1] != 3:
            raise ValueError(
                f"Expected 3 channels (RGB), got {generated.shape[1]}"
            )

        # Validate value ranges (expect [-1, 1] with some tolerance)
        gen_min, gen_max = generated.min(), generated.max()
        tgt_min, tgt_max = target.min(), target.max()

        if gen_min < -1.5 or gen_max > 1.5:
            raise ValueError(
                f"Generated image values out of expected range [-1, 1]: "
                f"[{gen_min:.3f}, {gen_max:.3f}]"
            )
        if tgt_min < -1.5 or tgt_max > 1.5:
            raise ValueError(
                f"Target image values out of expected range [-1, 1]: "
                f"[{tgt_min:.3f}, {tgt_max:.3f}]"
            )

        # Ensure tensors are on correct device
        generated = generated.to(self.device)
        target = target.to(self.device)

        # Compute LPIPS
        # Note: Model is in eval() mode with frozen parameters,
        # but we allow gradients to flow for optimization
        loss = self.model(generated, target)

        # loss shape: [B, 1, 1, 1] for non-spatial or [B, 1, H, W] for spatial
        if not self.spatial:
            # Return mean over batch
            return loss.mean()
        else:
            return loss

