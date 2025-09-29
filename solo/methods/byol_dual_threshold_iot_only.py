import torch
import torch.nn as nn
from typing import Any, Dict, List, Sequence

from solo.losses.byol_dual_threshold_iot_onlt import dual_threshold_iot_loss
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select

class dual_threshold_BYOL(BaseMethod):
    """
        BYOL with IOT-Loss using Dual-Threshold Smooth Mapping.
        Implements Framework III (Asymmetric Structural Alignment) from the IOT paper.
    """

    def __init__(
            self,
            cfg,
            # BYOL architecture
            pred_hidden_dim: int = 4096,
            proj_hidden_dim: int = 4096,
            proj_output_dim: int = 256,
            # IOT Dual-Threshold parameters
            gamma: float = 1.0,
            epsilon: float = 1e-4,
            e0: float = 0.1,
            e1: float = 0.9,
            # EMA
            base_tau_momentum: float = 0.99,
            final_tau_momentum: float = 1.0,
            momentum_classifier: bool = False,
            **kwargs
    ):
        """

        :param cfg:
        :param pred_hidden_dim: Hidden dimension of predictor
        :param proj_hidden_dim: Hidden dimension of projector
        :param proj_output_dim: Output dimension of projector
        :param gamma: IOT regularization coefficient
        :param epsilon: Minimal affinity (ε₀)
        :param e0: Low similarity threshold for smoothstep
        :param e1: High similarity threshold for smoothstep
        :param base_tau_momentum: Base EMA coefficient
        :param final_tau_momentum: Final EMA coefficient
        :param momentum_classifier:
        :param kwargs:
        """