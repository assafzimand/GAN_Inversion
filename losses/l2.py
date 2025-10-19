"""
L2 (MSE) pixel-wise loss for image reconstruction.

Implements Mean Squared Error loss with proper shape validation
and normalization checks. Expects inputs in [-1, 1] range.
"""

import torch
import torch.nn as nn
from typing import Optional


class L2Loss(nn.Module):
    """
    Mean Squared Error loss for image reconstruction.

    Computes pixel-wise MSE between generated and target images.
    Assumes inputs are normalized to [-1, 1] range.

    Args:
        reduction: Reduction mode - 'mean', 'sum', or 'none'
        validate_range: If True, check inputs are in reasonable range

    Example:
        >>> loss_fn = L2Loss(reduction='mean')
        >>> generated = torch.randn(1, 3, 256, 256)
        >>> target = torch.randn(1, 3, 256, 256)
        >>> loss = loss_fn(generated, target)
    """

    def __init__(
        self,
        reduction: str = "mean",
        validate_range: bool = True
    ):
        """
        Initialize L2 loss.

        Args:
            reduction: 'mean', 'sum', or 'none'
            validate_range: Whether to validate input ranges

        Raises:
            ValueError: If reduction is not valid
        """
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction
        self.validate_range = validate_range

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L2 loss between generated and target images.

        Args:
            generated: Generated images [B, C, H, W] in [-1, 1]
            target: Target images [B, C, H, W] in [-1, 1]

        Returns:
            Loss scalar (if reduction='mean' or 'sum')
            Loss tensor [B] (if reduction='none')

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

        # Validate value ranges (expect [-1, 1] with some tolerance)
        if self.validate_range:
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

        # Compute MSE
        mse = (generated - target) ** 2

        # Apply reduction
        if self.reduction == "mean":
            return mse.mean()
        elif self.reduction == "sum":
            return mse.sum()
        else:  # none
            return mse.mean(dim=[1, 2, 3])  # Per-sample loss

