"""Deterministic task identities and independent context/source/query streams."""
from dataclasses import dataclass
import hashlib
import struct
import torch


def seed_for(*parts):
    digest = hashlib.sha256(": ".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63-1)


@dataclass(frozen=True)
class TaskSpec:
    split: str
    index: int
    root_seed: int = 42
    family: str = "diagonal_gaussian_binary_v1"

    def __post_init__(self):
        if self.split not in {"train", "validation", "test"} or self.index < 0:
            raise ValueError("Invalid task split or index")
        if self.family != "diagonal_gaussian_binary_v1":
            raise ValueError("Unsupported task family")

    @property
    def task_id(self):
        return f"{self.family}/{self.root_seed}/{self.split}/{self.index:05d}"


@dataclass(frozen=True)
class Context:
    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self):
        if (self.x.ndim != 2 or self.x.shape[1] != 8 or self.y.shape != (len(self.x),)
                or len(self.x) < 2 or self.x.dtype != torch.float32 or self.y.dtype != torch.long
                or not torch.isfinite(self.x).all() or set(self.y.tolist()) != {0, 1}):
            raise ValueError("Need finite float32 [N,8] context with both binary int64 labels")


def data_hash(data):
    """Stable little-endian float32/int64 stream hash without requiring NumPy."""
    x = data.x.detach().cpu().reshape(-1).tolist()
    y = data.y.detach().cpu().tolist()
    return hashlib.sha256(struct.pack(f"<{len(x)}f", *x)
                          + struct.pack(f"<{len(y)}q", *y)).hexdigest()


def task_specs(split, count, root_seed=42):
    if count < 1:
        raise ValueError("Need a positive task count")
    return [TaskSpec(split, index, root_seed) for index in range(count)]


def sample_task(spec, role, count, *, allow_test=False):
    """Test use is explicit; ordinary development commands never enable it."""
    if spec.split == "test" and not allow_test:
        raise ValueError("Test tasks are sealed until a locked evaluation")
    if role not in {"context", "source", "query"} or count < 2 or count % 2:
        raise ValueError("Use a known role and an even, positive sample count")
    task_rng = torch.Generator().manual_seed(seed_for(spec.task_id, "parameters"))
    direction = torch.randn(8, generator=task_rng)
    direction = direction/direction.norm()
    separation = .8+1.2*torch.rand((), generator=task_rng)
    shift = .5*torch.randn(8, generator=task_rng)
    scale = .5+torch.rand(8, generator=task_rng)
    # Random label orientation is stable within a task and absent from the API output.
    orientation = 1 if torch.rand((), generator=task_rng) > .5 else -1
    rng = torch.Generator().manual_seed(seed_for(spec.task_id, role))
    y = torch.arange(count) % 2
    noise = torch.randn(count, 8, generator=rng)
    x = shift+orientation*(2*y[:, None]-1)*separation*direction+noise*scale
    order = torch.randperm(count, generator=rng)
    return Context(x[order].float(), y[order].long())
