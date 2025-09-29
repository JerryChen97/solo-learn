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
