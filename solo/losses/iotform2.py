
import torch
import torch.nn.functional as F

def my_nt_xent_loss(z1, z2, temperature: float = 0.2, lambda_reg: float = 0.0):
	"""
	SimCLR-style InfoNCE loss + small cosine regularizer.
	"""
	z1 = F.normalize(z1, dim=-1)
	z2 = F.normalize(z2, dim=-1)
	b = z1.size(0)

	reps = torch.cat([z1, z2], dim=0)
	sim = (reps @ reps.t()) / temperature
	sim = sim.masked_fill(torch.eye(2*b, device=sim.device, dtype=torch.bool), -1e9)

	logits_i = sim[:b, b:]
	logits_j = sim[b:, :b]
	targets = torch.arange(b, device=z1.device)

	loss_i = F.cross_entropy(logits_i, targets)
	loss_j = F.cross_entropy(logits_j, targets)
	reg = (1.0 - F.cosine_similarity(z1, z2, dim=-1)).mean()

	return 0.5 * (loss_i + loss_j) + lambda_reg * reg