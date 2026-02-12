import torch
import torch.nn as nn
import torch.nn.functional as F


class dual_threshold_iot_loss(nn.Module):
    """
        Inverse Optimal Transport Loss for self-supervised learning.
        dual threshold iot loss
    """

    def __init__(
            self,
            gamma: float = 1.0,
            epsilon: float = 1e-4,
            e0: float = 0.1,
            e1: float = 0.9,
    ):
        """

        :param gamma:  Regularization coefficient for entropy term
        :param epsilon: Minimal affinity to ensure numerical stability
        :param e0: Low similarity threshold for smoothstep transition
        :param e1: High similarity threshold for smoothstep transition
        """
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.e0 = e0
        self.e1 = e1

    def compute_target_plan(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute target transport plan X using Dual-Threshold Smooth Mapping.

        :param z: Embeddings [batch_size, dim]
        :return: Target transport plan [batch_size, batch_size]
        """
        # Normalize embeddings for cosine similarity
        z_norm = F.normalize(z, dim=-1, p=2)
        # Compute cosine similarity matrix
        sim_matrix = torch.mm(z_norm, z_norm.t())
        # Apply dual-threshold smooth mapping (Form 1 from paper)
        # X_ij = ε₀ + (1 - ε₀)S(s_ij; e₀, e₁)
        X = self.epsilon + (1 - self.epsilon) * self.smoothstep(sim_matrix)

        return X

    def smoothstep(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:  Cosine similarity values (typically in range [0, 1])
        :return: Smooth transition function S(s; e0, e1) that transitions from 0 to 1
        as similarity varies between e0 and e1.
        """
        # x is the similarity matrix (sim_matrix from above)
        # Each element x[i,j] represents cosine similarity between embeddings i and j

        # Map similarities to [0, 1] based on thresholds:
        # - If x[i,j] <= e0 (low similarity): output 0
        # - If x[i,j] >= e1 (high similarity): output 1
        # - If e0 < x[i,j] < e1: smooth transition from 0 to 1

        x_clamped = torch.clamp((x - self.e0) / (self.e1 - self.e0 + 1e-8), 0, 1)
        # Smoothstep formula: 3x^2 - 2x^3 for smooth transition
        return x_clamped * x_clamped * (3.0 - 2.0 * x_clamped)

    def compute_cost_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
         Compute pairwise squared Euclidean distances (cost matrix C).
        :param z: Embeddings [batch_size, dim]
        :return: C: Cost matrix [batch_size, batch_size]
        """
        # Efficient computation of squared Euclidean distances
        # ||z_i - z_j||^2 = ||z_i||^2 - 2<z_i, z_j> + ||z_j||^2
        z_sqnorms = torch.sum(z ** 2, dim=1, keepdim=True)
        z_inner = torch.mm(z, z.t())
        C = z_sqnorms - 2 * z_inner + z_sqnorms.t()
        # Ensure non-negative distances (numerical stability)
        C = torch.clamp(C, min=0)

        return C

    def forward(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute IOT loss between online predictions and target projections.
        :param p: Online network predictions [batch_size, dim]
        :param z: Target network projections [batch_size, dim] (should be detached)
        :return:  loss: IOT loss value
        """
        # Normalize inputs (important for stable training)
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)

        # Compute learned cost matrix C(θ) from online network
        C_theta = self.compute_cost_matrix(p)

        # Compute target transport plan from target network
        # This defines the desired geometric structure
        X_target = self.compute_target_plan(z.detach())

        # Compute target cost matrix G_ent from IOT formula
        # G_ent(X)_ij = -γ log(X_ij)
        G_ent = -self.gamma * torch.log(X_target )

        # IOT loss: minimize discrepancy between learned and target cost matrices
        # L_IOT(θ) = 1/2 ||C(θ) - G_ent(X)||²_F
        loss = 0.5 * torch.mean((C_theta - G_ent) ** 2)

        return loss
