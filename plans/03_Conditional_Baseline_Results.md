# Deterministic conditional baseline results

Completed October 27, 2026 under the [fixed build plan](03_Conditional_Baseline.md).
The source collection completed all 256 training and 64 validation tasks in 16.30
seconds. Every source classifier cleared the 70% gate. The hypernetwork then
completed exactly 1,000 updates in 6.12 seconds. There was one seed, no checkpoint
search, and no test-task access.

| Method | Training mean accuracy | Validation mean accuracy |
|---|---:|---:|
| Source-fitted reference (256 examples) | 90.92% | 91.39% |
| Context-only Adam (64 examples) | 88.72% | 89.82% |
| Analytic context baseline (64 examples) | 90.51% | 91.31% |
| Deterministic hypernetwork (64 examples) | 88.37% | 85.73% |
| Hypernetwork with rotated task contexts | 50.44% | 51.12% |
| Pooled training-target mean | 52.73% | 50.01% |

The learned model uses the context: rotating contexts reduces performance toward
chance. It nevertheless trails context-only optimization by 4.09 percentage
points and the analytic baseline by 5.58 points on validation tasks. It misses the
master plan's provisional mean-accuracy target of being within two points of
context-only optimization. This is a negative baseline result; it does not
establish a benefit for diffusion or parameter generation over analytic inference.

No automatic tuning, extra training seed, or diffusion run followed this failed
gate. The next decision is whether to revise the bounded baseline plan, diagnose
weight-target loss versus classifier behavior, or retain this baseline and specify
a separate diffusion experiment. Final test tasks remain sealed. All task outcomes
are retained, with no filtering. These are within-family synthetic results only.

The exported hypernetwork contains the encoder, output head, training-only mean
and scale, classifier manifest, seed, update count, PyTorch version, and source
manifest checksum. Reload requires only this package and PyTorch. Its context API
accepts labeled examples without task IDs or source/query inputs. Every generated
classifier is exported before that task's query stream is sampled for scoring.

Durable evidence:

- [Full source collection CSV](results/03_source_collection_tasks.csv) and
  [summary](results/03_source_collection_summary.json).
- [All generated task scores](results/03_hypernetwork_tasks.csv) and
  [summary](results/03_hypernetwork_summary.json).
- Ignored `artifacts/full-source-collection/`: target manifests and classifiers.
- Ignored `artifacts/conditional-baseline/`: hypernetwork checkpoint and generated
  classifiers for all four methods, with separate training/validation directories.

## Environment and validation

A clean project-local `.venv` was created and the editable package plus development
dependencies installed successfully. `pip check` passes. The installed package's
smoke run reproduces the original foundation accuracies. Ten tests pass in that
local environment, covering foundation boundaries, canonicalization, repeatable
fixed-stream fitting, time caps, all pilot exports and training-only statistics,
and standalone hypernetwork reload. An undeclared NumPy dependency in the original
stream hashing was removed; hashes retain the same little-endian float32/int64
representation. PyTorch itself emits an optional NumPy initialization warning in
this environment; package operations and tests complete without NumPy.

`requirements-lock.txt` records the installed dependency versions for Python 3.12
on macOS ARM64. It is a version snapshot, without wheel hashes or a cross-platform
resolution guarantee. Training ran in the earlier existing Python environment;
the new local environment verified package installation, tests, smoke behavior,
and all 320 context hashes and hypernetwork exports against the saved run.
