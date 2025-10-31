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

    @staticmethod
    def add_and_assert_specific_cfg(cfg):
        """Adds method specific default values/checks for config."""
        cfg = BaseMethod.add_and_assert_specific_cfg(cfg)
        
        # Set defaults for method_kwargs if not present
        cfg.method_kwargs.proj_hidden_dim = cfg.method_kwargs.get("proj_hidden_dim", 4096)
        cfg.method_kwargs.proj_output_dim = cfg.method_kwargs.get("proj_output_dim", 256)
        cfg.method_kwargs.pred_hidden_dim = cfg.method_kwargs.get("pred_hidden_dim", 4096)
        cfg.method_kwargs.gamma = cfg.method_kwargs.get("gamma", 1.0)
        cfg.method_kwargs.epsilon = cfg.method_kwargs.get("epsilon", 1e-4)
        cfg.method_kwargs.e0 = cfg.method_kwargs.get("e0", 0.1)
        cfg.method_kwargs.e1 = cfg.method_kwargs.get("e1", 0.9)
        cfg.method_kwargs.base_tau_momentum = cfg.method_kwargs.get("base_tau_momentum", 0.99)
        cfg.method_kwargs.final_tau_momentum = cfg.method_kwargs.get("final_tau_momentum", 1.0)
        cfg.method_kwargs.momentum_classifier = cfg.method_kwargs.get("momentum_classifier", False)
        
        return cfg

    def __init__(self, cfg, **kwargs):
        """
        BYOL with IOT-Loss using Dual-Threshold Smooth Mapping.
        
        Args:
            cfg: Configuration object containing all parameters
        """
        super().__init__(cfg, **kwargs)
        
        # Extract parameters from config
        pred_hidden_dim = cfg.method_kwargs.pred_hidden_dim
        proj_hidden_dim = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim = cfg.method_kwargs.proj_output_dim
        gamma = cfg.method_kwargs.gamma
        epsilon = cfg.method_kwargs.epsilon
        e0 = cfg.method_kwargs.e0
        e1 = cfg.method_kwargs.e1
        self.base_tau_momentum = cfg.method_kwargs.base_tau_momentum
        self.final_tau_momentum = cfg.method_kwargs.final_tau_momentum
        self.momentum_classifier = cfg.method_kwargs.momentum_classifier
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

        # Momentum networks - build after parent init so self.encoder exists
        self._build_momentum_network(proj_hidden_dim, proj_output_dim)
        # IOT Loss with Dual-Threshold
        self.iot_loss = dual_threshold_iot_loss(
            gamma=gamma,
            epsilon=epsilon,
            e0=e0,
            e1=e1
        )
        
        # Track last step for momentum update
        self.last_step = 0

    def _build_momentum_network(self, proj_hidden_dim: int, proj_output_dim: int):
        """Build momentum encoder and projector."""
        # Momentum encoder - copy from backbone
        kwargs = self.backbone_args.copy()
        method = self.cfg.method
        self.momentum_encoder = self.base_model(method, **kwargs)
        if self.backbone_name.startswith("resnet"):
            self.momentum_encoder.fc = nn.Identity()
            cifar = self.cfg.data.dataset in ["cifar10", "cifar100"]
            if cifar:
                self.momentum_encoder.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.momentum_encoder.maxpool = nn.Identity()
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
        for param_b, param_m in zip(self.backbone.parameters(),
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
        for param_b, param_m in zip(self.backbone.parameters(),
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
        f1 = self.backbone(aug1)
        z1 = self.projector(f1)
        p1 = self.predictor(z1)

        f2 = self.backbone(aug2)
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
        # Unpack batch - format is [indexes, X, targets]
        indexes, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X
        
        # Forward pass with first two views
        outputs = self.forward(X)
        # Compute IOT loss with dual-threshold mapping
        loss = self.iot_loss(outputs["p_online"], outputs["z_momentum"])
        # Logging
        self.log("train_iot_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update momentum networks at the end of each training batch."""
        if self.trainer.global_step > self.last_step:
            tau = self._get_momentum_tau()
            self.update_momentum_network(tau)
            self.last_step = self.trainer.global_step

    @property
    def learnable_params(self):
        """Adds projector and predictor parameters to the parent's learnable parameters."""
        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params
