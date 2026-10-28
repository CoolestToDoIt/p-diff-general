"""Strict, versioned codec for the first fixed 8-to-2 linear classifier."""
import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 2)

    def forward(self, x):
        return self.linear(x)


def manifest():
    return {"version": 1, "architecture": "linear_8_2", "parameter_count": 18,
            "dtype": "float32", "tensors": [
                {"name": "linear.weight", "shape": [2, 8], "offset": 0, "length": 16},
                {"name": "linear.bias", "shape": [2], "offset": 16, "length": 2}]}


def flatten(model):
    if type(model) is not Classifier or model.linear.weight.shape != (2, 8):
        raise ValueError("Unsupported classifier architecture")
    values = torch.cat([model.linear.weight.detach().cpu().reshape(-1), model.linear.bias.detach().cpu()])
    if values.dtype != torch.float32 or not torch.isfinite(values).all():
        raise ValueError("Need finite float32 parameters")
    return values.clone()


def restore(vector):
    if vector.shape != (18,) or vector.dtype != torch.float32 or not torch.isfinite(vector).all():
        raise ValueError("Need a finite float32 vector of length 18")
    # Construction must not advance the caller's training/noise random stream.
    with torch.random.fork_rng(devices=[]):
        model = Classifier()
    with torch.no_grad():
        model.linear.weight.copy_(vector[:16].reshape(2, 8))
        model.linear.bias.copy_(vector[16:])
    return model.eval().requires_grad_(False)


def save_classifier(path, vector, metadata):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    restore(vector)
    with path.open("xb") as handle:
        torch.save({"manifest": manifest(), "vector": vector.detach().cpu(), "metadata": metadata}, handle)


def load_classifier(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["manifest"] != manifest():
        raise ValueError("Classifier manifest mismatch")
    return restore(payload["vector"])
