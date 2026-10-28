"""Offline foundation smoke check; never train a generator or open test tasks."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time
import torch
from .conditioning import centroid_parameters
from .parameters import load_classifier, save_classifier, restore
from .tasks import data_hash, sample_task, task_specs


def run(output, tasks_per_split=8):
    if not 1 <= tasks_per_split <= 8:
        raise ValueError("Foundation smoke is limited to 1-8 tasks per split")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    rows = []
    for split in ("train", "validation"):
        for spec in task_specs(split, tasks_per_split):
            context = sample_task(spec, "context", 64)
            query = sample_task(spec, "query", 256)
            vector = centroid_parameters(context)
            file = output/"models"/f"{split}-{spec.index:03d}.pt"
            save_classifier(file, vector, {"task_id": spec.task_id, "method": "context_diagonal_centroid"})
            model = load_classifier(file)
            with torch.no_grad():
                predictions = model(query.x).argmax(1)
                assert torch.equal(predictions, restore(vector)(query.x).argmax(1))
            rows.append({"task_id": spec.task_id, "split": split, "context_count": 64,
                         "query_count": 256, "accuracy": (predictions == query.y).double().mean().item(),
                         "constant_class_accuracy": .5,
                         "context_sha256": data_hash(context),
                         "query_sha256": data_hash(query)})
    with (output/"tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {"phase": "foundation_smoke", "root_seed": 42, "generator_trained": False,
               "test_tasks_accessed": False, "training_updates": 0,
               "task_family": "diagonal_gaussian_binary_v1", "tasks_per_split": tasks_per_split,
               "elapsed_seconds": time.perf_counter()-started, "torch": str(torch.__version__),
               "splits": {split: {"count": tasks_per_split,
                   "mean_accuracy": statistics.mean(r["accuracy"] for r in rows if r["split"] == split),
                   "worst_accuracy": min(r["accuracy"] for r in rows if r["split"] == split)}
                   for split in ("train", "validation")}}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/foundation-smoke")
    parser.add_argument("--tasks-per-split", default=8, type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.tasks_per_split), indent=2))


if __name__ == "__main__":
    main()
