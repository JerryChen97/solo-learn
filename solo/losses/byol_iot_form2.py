import torch
import torch.nn as nn
import torch.nn.functional as F


class iot_loss_form2(nn.Module):
    """
    Inverse Optimal Transport Loss - Form 2: Single-Threshold Simplified Mapping.
    
    This is a simplified version of the IOT loss with fewer hyperparameters.
    Instead of dual thresholds with smooth transition, it uses direct clamping:
    
    X_ij = max(s_ij, ε)
    
    where s_ij is the cosine similarity between embeddings i and j.
    """

    def __init__(
            self,
            gamma: float = 1.0,
            epsilon: float = 1e-4,
    ):
        """
        Initialize IOT Loss Form 2.

        Args:
            gamma: Regularization coefficient for entropy term (controls cost scale)
            epsilon: Minimal affinity for numerical stability and similarity floor
        """
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def compute_target_plan(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute target transport plan X using Single-Threshold Simplified Mapping.
        
        Form 2 from paper: X_ij = max(s_ij, ε)
        
        This is simpler than Form 1 (dual-threshold) but still captures
        the key idea: similar pairs have high affinity, dissimilar pairs
        have low (but non-zero) affinity.

        Args:
            z: Embeddings [batch_size, dim]
            
        Returns:
            X: Target transport plan [batch_size, batch_size]
        """
        # Normalize embeddings for cosine similarity
        z_norm = F.normalize(z, dim=-1, p=2)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.mm(z_norm, z_norm.t())
        
        # Apply single-threshold clamping: X_ij = max(s_ij, ε)
        # This ensures all similarities are at least epsilon
        X = torch.clamp(sim_matrix, min=self.epsilon)
        
        return X

    def compute_cost_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise squared Euclidean distances (cost matrix C).
        
        Args:
            z: Embeddings [batch_size, dim]
            
        Returns:
            C: Cost matrix [batch_size, batch_size]
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
        
        Args:
            p: Online network predictions [batch_size, dim]
            z: Target network projections [batch_size, dim] (should be detached)
            
        Returns:
            loss: IOT loss value
        """
        # Normalize inputs for stable training
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        
        # Compute learned cost matrix C(θ) from online network
        C_theta = self.compute_cost_matrix(p)
        
        # Compute target transport plan from target network
        # This defines the desired geometric structure
        X_target = self.compute_target_plan(z.detach())
        
        # Compute target cost matrix G_ent from IOT formula
        # G_ent(X)_ij = -γ log(X_ij)
        G_ent = -self.gamma * torch.log(X_target)
        
        # IOT loss: minimize discrepancy between learned and target cost matrices
        # L_IOT(θ) = 1/2 ||C(θ) - G_ent(X)||²_F
        loss = 0.5 * torch.mean((C_theta - G_ent) ** 2)
        
        return loss
