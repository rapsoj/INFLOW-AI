import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set seeds for deterministic experiment behavior where possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow is optional for this ablation framework.
        pass
