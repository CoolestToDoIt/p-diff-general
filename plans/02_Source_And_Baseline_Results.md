# Source and baseline pilot results

Completed October 27, 2026, following the user's request to continue the build.
The [bounded pilot](02_Source_And_Baseline_Pilot.md) ran once on all 32 training
and eight validation tasks. No test tasks were opened and no learned generator was
trained. All 40 source models cleared the predeclared 70% query-accuracy gate.

| Split | Source fitting | Context fitting | Analytic context baseline |
|---|---:|---:|---:|
| Training (32 tasks) | 91.88% | 89.45% | 91.22% |
| Validation (8 tasks) | 88.62% | 86.77% | 89.26% |

The constant-class reference scores 50%. Source and context fits used zero
initialization, full-batch cross-entropy, Adam at 0.03, and exactly 200 updates.
Source fits saw 256 examples; context fits and the analytic baseline saw 64.
Query streams contained 256 examples and were used only for scoring.

The analytic method remains a strong baseline: it exceeded both optimized methods
on the eight validation tasks. These results validate the target pipeline, not
weight-generator transfer. There was no query-based checkpoint selection, task
filtering, tuning, or retry. The small validation set limits conclusions.

The run took 2.44 seconds with one CPU thread, Python 3.12.4 and PyTorch 2.14.1.
The artifact manifest records the exact runtime platform; elapsed time is not a
portable performance guarantee. Integrity checks covered distinct role hashes,
canonicalization preserving source probabilities and predictions, finite parameter
vectors, and training-only parameter statistics. Integration tests additionally
check all exports, standalone reload, sealed test tasks, and statistics recomputed
from the 32 training targets.

Task-level results and the unrounded summary are preserved in
[CSV](results/02_source_baseline_tasks.csv) and
[JSON](results/02_source_baseline_summary.json). The original foundation evidence
is also preserved in `plans/results/01_foundation_*`.

Ignored `artifacts/source-baseline-pilot/` contains the pre-fit task/stream manifest,
40 source classifiers, 40 context classifiers, training-only mean/std, and full
metrics. Training and validation targets are kept in separate directories.
The CLI refuses to overwrite an existing output directory and flushes completed
task rows; on its time cap it preserves exports already written and marks the
summary as capped.

Reproduce with an installed package:

```sh
python -m p_diff_general.pilot --output artifacts/source-baseline-pilot-repeat
```

The executed run used the existing earlier project's Python with `PYTHONPATH=src`.
A project-local dependency installation is a separate environment check.
