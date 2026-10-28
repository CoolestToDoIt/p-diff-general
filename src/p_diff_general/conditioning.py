"""Task conditioning accepts labeled examples, never task IDs or query labels."""
import torch
from torch import nn
from .tasks import Context


class ContextEncoder(nn.Module):
    """Permutation-invariant class pooling; not trained in the foundation phase."""
    def __init__(self, width=32, embedding_size=64):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width))
        self.output = nn.Sequential(nn.Linear(width*2, embedding_size), nn.SiLU())

    def forward(self, context):
        if not isinstance(context, Context):
            raise ValueError("Expected labeled task context")
        points = self.point(context.x)
        pooled = torch.cat([points[context.y == label].mean(0) for label in (0, 1)])
        return self.output(pooled)


@torch.no_grad()
def centroid_parameters(context):
    """Emit equivalent linear logits for a shared-diagonal Gaussian baseline.

    This is an analytic baseline, not a trained diffusion or hypernetwork model.
    Statistics are fitted only to the given context examples.
    """
    means = torch.stack([context.x[context.y == label].mean(0) for label in (0, 1)])
    residuals = context.x-means[context.y]
    variance = residuals.square().mean(0).clamp_min(.05)
    weights = means/variance
    bias = -.5*(means.square()/variance).sum(1)
    # Canonical zero-sum logits remove softmax's shared-logit ambiguity.
    weights = weights-weights.mean(0)
    bias = bias-bias.mean()
    return torch.cat([weights.reshape(-1), bias]).float()
