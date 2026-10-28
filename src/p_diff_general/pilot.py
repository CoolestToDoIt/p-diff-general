"""Bounded source/context classifier pilot; no generator fitting or test access."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from .conditioning import centroid_parameters
from .parameters import Classifier, flatten, manifest, restore, save_classifier
from .tasks import data_hash, sample_task, task_specs


def canonicalize(vector):
    weights = vector[:16].reshape(2, 8)
    bias = vector[16:]
    return torch.cat([(weights - weights.mean(0)).reshape(-1), bias - bias.mean()])


def fit(data, deadline):
    """Fit only the supplied optimization stream, with a fixed update budget."""
    with torch.random.fork_rng(devices=[]):
        model = Classifier()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    started = time.perf_counter()
    for _ in range(200):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Pilot CPU wall-time cap reached")
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data.x), data.y)
        loss.backward()
        optimizer.step()
    vector = canonicalize(flatten(model))
    with torch.no_grad():
        before, after = model(data.x), restore(vector)(data.x)
        torch.testing.assert_close(before.softmax(1), after.softmax(1), atol=1e-6, rtol=1e-5)
        if not torch.equal(before.argmax(1), after.argmax(1)):
            raise RuntimeError("Canonicalization changed predictions")
        final_loss = F.cross_entropy(after, data.y).item()
    return vector, final_loss, time.perf_counter() - started


def run(output, *, full_collection=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    deadline = started + 30 * 60
    counts = {"train": 256, "validation": 64} if full_collection else {"train": 32, "validation": 8}
    specs = task_specs("train", counts["train"]) + task_specs("validation", counts["validation"])
    # Persist identities, configuration, and stream hashes before fitting anything.
    records = []
    for spec in specs:
        records.append({"spec": asdict(spec), "task_id": spec.task_id,
                        "hashes": {role: data_hash(sample_task(spec, role, count))
                                   for role, count in (("context", 64), ("source", 256), ("query", 256))}})
    hashes = [h for record in records for h in record["hashes"].values()]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Repeated task/role stream")
    configuration = {"phase": "full_source_collection" if full_collection else "source_baseline_pilot", "root_seed": 42,
                     "counts": counts,
                     "examples": {"context": 64, "source": 256, "query": 256},
                     "optimizer": "Adam", "learning_rate": .03, "updates_per_fit": 200,
                     "initialization": "zero", "wall_time_cap_seconds": 1800,
                     "manifest": manifest(), "tasks": records,
                     "python": platform.python_version(), "torch": str(torch.__version__),
                     "platform": platform.platform(), "threads": 1}
    (output / "manifest.json").write_text(json.dumps(configuration, indent=2) + "\n")
    rows, targets = [], []
    status = "complete"
    fields = ["task_id", "split", "source_accuracy", "context_accuracy", "analytic_accuracy",
              "constant_accuracy", "source_loss", "context_loss", "source_seconds",
              "context_seconds", "source_context_disagreement", "source_below_70_percent"]
    with (output / "tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for spec in specs:
            try:
                context = sample_task(spec, "context", 64)
                source = sample_task(spec, "source", 256)
                source_vector, source_loss, source_seconds = fit(source, deadline)
                save_classifier(output / "source" / spec.split / f"{spec.index:03d}.pt",
                                source_vector, {"task_id": spec.task_id, "method": "source_adam"})
                context_vector, context_loss, context_seconds = fit(context, deadline)
                save_classifier(output / "context" / spec.split / f"{spec.index:03d}.pt",
                                context_vector, {"task_id": spec.task_id, "method": "context_adam"})
            except TimeoutError:
                status = "capped"
                break
            query = sample_task(spec, "query", 256)
            with torch.no_grad():
                predictions = [restore(v)(query.x).argmax(1) for v in
                               (source_vector, context_vector, centroid_parameters(context))]
                accuracies = [(p == query.y).float().mean().item() for p in predictions]
            row = dict(zip(fields, [spec.task_id, spec.split, *accuracies, .5,
                                   source_loss, context_loss, source_seconds, context_seconds,
                                   (predictions[0] != predictions[1]).float().mean().item(),
                                   accuracies[0] < .7]))
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if spec.split == "train":
                targets.append(source_vector)
    if targets:
        values = torch.stack(targets)
        torch.save({"mean": values.mean(0), "std": values.std(0, unbiased=False),
                    "count": len(values), "split": "train", "manifest": manifest()},
                   output / "train_statistics.pt")
    summary = {"status": status, "completed_tasks": len(rows),
               "elapsed_seconds": time.perf_counter() - started,
               "generator_trained": False, "test_tasks_accessed": False,
               "source_below_70_percent": [r["task_id"] for r in rows if r["source_below_70_percent"]],
               "splits": {}}
    for split in ("train", "validation"):
        subset = [r for r in rows if r["split"] == split]
        if subset:
            summary["splits"][split] = {key: sum(r[key] for r in subset) / len(subset)
                for key in ("source_accuracy", "context_accuracy", "analytic_accuracy",
                            "source_context_disagreement")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-collection", action="store_true",
                        help="Collect the bounded 256/64 conditional-baseline target set")
    args = parser.parse_args()
    print(json.dumps(run(args.output, full_collection=args.full_collection), indent=2))


if __name__ == "__main__":
    main()
