"""
Global seeding utilities for reproducibility.

Sets all random number generator seeds for Python, NumPy, and PyTorch
to ensure reproducible experiments.
"""

import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - cuDNN (deterministic mode if requested)

    Args:
        seed: Random seed value
        deterministic: If True, enable cudnn deterministic mode
                      (may impact performance but ensures reproducibility)

    Example:
        >>> set_seed(42)
        >>> # All random operations are now deterministic
    """
    logger.info(f"Setting global seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # cuDNN deterministic mode
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("cuDNN deterministic mode enabled")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark mode enabled (non-deterministic)")
    
    logger.info(f"Global seed {seed} set successfully")

