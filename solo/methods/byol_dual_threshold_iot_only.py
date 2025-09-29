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
        super().__init__(cfg, **kwargs)
        # Projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # Predictor (only for online network)
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        # Momentum networks
        self._build_momentum_network(proj_hidden_dim, proj_output_dim)
        # IOT Loss with Dual-Threshold
        self.iot_loss =dual_threshold_iot_loss(
            gamma=gamma,
            epsilon=epsilon,
            e0=e0,
            e1=e1
        )

        # EMA parameters
        self.base_tau_momentum = base_tau_momentum
        self.final_tau_momentum = final_tau_momentum
        self.momentum_classifier = momentum_classifier

    def _build_momentum_network(self, proj_hidden_dim: int, proj_output_dim: int):
        """Build momentum encoder and projector."""
        # Momentum encoder
        self.momentum_encoder = self._build_encoder()
        # Momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        # Initialize and freeze momentum networks
        for param_b, param_m in zip(self.encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
        for param_b, param_m in zip(self.projector.parameters(),
                                    self.momentum_projector.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def update_momentum_network(self, tau: float):
        """Update momentum networks with EMA."""
        for param_b, param_m in zip(self.encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data = tau * param_m.data + (1 - tau) * param_b.data

        for param_b, param_m in zip(self.projector.parameters(),
                                    self.momentum_projector.parameters()):
            param_m.data = tau * param_m.data + (1 - tau) * param_b.data

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Forward pass through both networks."""
        if not self.training:
            return super().forward(X)
        # Process two augmented views
        aug1, aug2 = X[:2]
        # Online network
        f1 = self.encoder(aug1)
        z1 = self.projector(f1)
        p1 = self.predictor(z1)

        f2 = self.encoder(aug2)
        z2 = self.projector(f2)
        p2 = self.predictor(z2)

        # Momentum network (no gradients)
        with torch.no_grad():
            f1_m = self.momentum_encoder(aug1)
            z1_m = self.momentum_projector(f1_m)
            f2_m = self.momentum_encoder(aug2)
            z2_m = self.momentum_projector(f2_m)
        # Combine for loss computation
        # p1 -> z2_m and p2 -> z1_m (cross-prediction)
        return {
            "p_online": torch.cat([p1, p2], dim=0),
            "z_momentum": torch.cat([z2_m, z1_m], dim=0)
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with IOT loss."""
        # Remove indices if present
        indexes = batch.pop("index", None)
        # Forward pass
        outputs = self.forward(batch["X"])
        # Compute IOT loss with dual-threshold mapping
        loss = self.iot_loss(outputs["p_online"], outputs["z_momentum"])
        # Logging
        self.log("train_iot_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        # Update momentum network
        tau = self._get_momentum_tau()
        self.update_momentum_network(tau)
        self.log("tau", tau, on_step=False, on_epoch=True, logger=True)

        return loss

    def _get_momentum_tau(self) -> float:
        """Cosine schedule for momentum coefficient."""
        max_steps = self.trainer.estimated_stepping_batches
        current_step = self.trainer.global_step
        base = self.base_tau_momentum
        final = self.final_tau_momentum

        tau = final - (final - base) * (
                (torch.cos(torch.tensor(torch.pi * current_step / max_steps)) + 1) / 2
        )
        return tau

    def configure_optimizers(self):
        """Configure optimizer for online network only."""
        params = []
        for module in [self.encoder, self.projector, self.predictor]:
            params.extend(list(module.parameters()))

        return super().configure_optimizers(params)
