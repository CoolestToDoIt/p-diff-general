# Architecture and implementation plan

Updated October 27, 2026.

## Purpose

Build one task-conditioned diffusion model that generates classifier weights for
new classification problems, including tasks excluded from generator training.
A new task is described by labeled examples. Its generated classifier should
predict immediately, without classifier fine-tuning or access to query labels.

The first milestone is unseen-task transfer within a synthetic task family.
Broader transfer, image tasks, and compression are later experiments, not current
capabilities. The [master plan](Project_Plan.md) defines the full research roadmap;
this document explains the architecture, current implementation, and next work.

## System design

The planned learned generation path is:

```text
labeled context -> set encoder -> conditional diffusion -> parameter vector
                                                               |
                                                       classifier codec
                                                               |
                                                    standalone classifier
                                                               |
                                                        query predictions
```

The first classifier is a fixed linear layer with eight inputs and two outputs:
16 weights and two biases, stored as 18 float32 parameters. Generating these
parameters directly avoids introducing compression or hidden-neuron alignment
before task conditioning has been demonstrated. Subtracting the classwise mean
of weights and biases removes the shared-logit ambiguity while preserving class
probabilities and predictions.

The later compressed design replaces the direct vector with a compact residual
code and a separate, frozen parameter decoder. That decoder must first preserve
classifier behavior and source variation against direct-weight, mean, and PCA
baselines. A fixed 8-to-32-to-2 MLP with 354 parameters is the proposed next target.

## Data flow and evaluation boundaries

Benchmark version 1 uses binary, eight-dimensional synthetic tasks. The declared
split has 256 training, 64 validation, and 64 test task identities, with root seed
42. Each task has independently seeded streams for 64 context examples, 256 source
optimization examples, and 256 query examples. Entire tasks belong to one split.
The configuration lives in [benchmark.json](../configs/benchmark.json).

During future training, source examples produce classifier targets. Only training
tasks may fit the encoder, generator, normalization, or compression. Validation
tasks support declared model selection. Source targets use more labeled examples
than context-only inference, so this supervision advantage must be reported.

During generation, labeled context is the only task input. Task IDs, procedural
teacher parameters, source weights, source optimizer state, and query labels must
never enter conditioning. Exported classifiers load without the generator,
context, source data, or task specification. Query labels score predictions; they
cannot select the best generated sample. Any future candidate selection requires
a separate selection stream. Test tasks remain sealed until locked evaluation.

## Foundation components (initial snapshot)

| Component | Responsibility | Current state |
|---|---|---|
| [tasks.py](../src/p_diff_general/tasks.py) | Deterministic tasks, whole-task split IDs, independent data streams, test access guard | Implemented |
| [parameters.py](../src/p_diff_general/parameters.py) | Strict 18-parameter manifest, flatten/restore, standalone export/load | Implemented |
| [conditioning.py](../src/p_diff_general/conditioning.py) | Permutation-invariant, label-sensitive context encoder; analytic diagonal-centroid weights | Encoder exists but is untrained; analytic baseline works |
| [smoke.py](../src/p_diff_general/smoke.py) | Context-only generation, query scoring, classifier export | Implemented; no training or test-task access |
| Conditional diffusion and parameter decoder | Learned weight/code generation and optional expansion | Planned; not implemented or trained |

The working foundation path is context → analytic diagonal-centroid parameters →
classifier codec → predictions. It does not use the untrained context encoder.
No source collection, deterministic hypernetwork, or learned weight generator has
been trained.

## Foundation evidence and environment

The original foundation verification passed five tests. The saved smoke summary
in `artifacts/foundation-smoke/summary.json` records eight training and eight
validation tasks, zero training updates, and no test-task access:

| Split | Mean accuracy | Worst task accuracy |
|---|---:|---:|
| Training | 91.65% | 77.73% |
| Validation | 89.26% | 75.39% |

These are analytic baseline results, not diffusion results. The ignored output
directory contains task-level metrics and 16 standalone classifiers. Full binaries
remain under `artifacts/`; durable reports and selected metrics belong in `plans/`.

The initial checks used Python from the earlier project's existing environment
with PyTorch 2.14.1 and `PYTHONPATH=src`, without installing this package. The earlier
project is not a runtime dependency. At handoff there was no local `.venv`; a fresh
environment installation and dependency lock had not been verified. Use the
[README](../README.md) for the normal installation and smoke commands, and always
choose a fresh output directory.

## Initial implementation sequence

1. Review the foundation against the benchmark and leakage boundaries. Preserve
   task-level smoke evidence and a concise result report in `plans/`.
2. Verify a project-local environment, package installation, and configuration
   consistency. Record runtime versions and resolve dependency reproducibility
   before release.
3. Review the initial repository contents for a coherent first commit. At handoff,
   project files were untracked and no initial commit or push had been made;
   this is historical state, not a guarantee of the current Git status.
4. Obtain the separate authorization required by the
   [source and baseline pilot](02_Source_And_Baseline_Pilot.md) before running it.
   The proposal covers 32 training and eight validation tasks, source and
   context-only linear classifiers with exactly 200 updates, analytic baselines,
   and a 30-minute CPU cap. It excludes generator training and test-task access.
5. Use that pilot's evidence to plan deterministic conditional baselines and then
   conditional diffusion. Compare against analytic inference, context-only
   optimization, and unconditional controls; preserve negative outcomes.

The foundation was the initially authorized phase. Later substantial training
phases retain their separate bounded plans and approval requirements. The present
documentation migration does not authorize those runs.

## Relationship to the earlier project

This repository is [p-diff-general](https://github.com/CoolestToDoIt/p-diff-general).
The earlier [p-diff-nn](https://github.com/CoolestToDoIt/p-diff-nn) checkout is now
named `p-diff-mnist`; reference commit `2fe703d` identifies the prior work.
That single-task MNIST prototype ran end to end, but diffusion lost to simpler
baselines. Its decoder favored the mean template and suppressed source variation.
Those negative results and already-seen MNIST test outcomes must be preserved.

This experiment is independent: no old source files, datasets, or models were
copied. The earlier results motivate the simpler initial architecture and explicit
baseline gates; they are not benchmarks for this project's transfer claims.
See [provenance](../docs/provenance.md) for the project relationship and
[foundation scope](01_Foundation.md) for the implemented phase.

## Build update: October 27, 2026

The user's subsequent instruction to keep building authorized the bounded source
pilot and conditional baseline work. Both are complete: 256/64 source targets and
a trained deterministic context-to-weight hypernetwork now exist. The latter uses
the previously untrained context encoder plus a linear parameter head; the table
above describes the foundation snapshot. Diffusion and compression remain planned.
The clean local install, test suite, and smoke check now pass; dependency versions
are recorded in `requirements-lock.txt`. See the
[conditional baseline report](03_Conditional_Baseline_Results.md) for current
results and the failed accuracy gate. No further tuning or final-test access followed.

## Pre-diffusion update

The user's next instruction authorized the bounded pre-diffusion diagnosis and
two behavior-loss runs. A direct generator reached 86.51% validation accuracy;
an analytic-anchored residual generator reached 90.28%, clearing the provisional
accuracy and conditioning gates while remaining below analytic inference at
91.31%. Gaussian controls, task-level paired intervals, an audited training-only
bundle and a checksum/schedule preflight are implemented.
[Current results](04_Pre_Diffusion_Results.md) supersede the earlier next-step
notes. [Diffusion settings and implementation checks](05_Diffusion_Proposal.md)
are proposed; no diffusion implementation or training has occurred.
