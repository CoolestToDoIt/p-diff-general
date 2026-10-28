# p-diff-general

Generate classifier parameters conditioned on labeled examples of a new task.
The research question is whether one trained generator can construct useful
classifiers for **tasks excluded from generator training**, without per-task
classifier gradient updates during generation.

Status: foundation, full source collection, deterministic weight/behavior baselines,
Gaussian controls, and audited pre-diffusion inputs implemented and evaluated.
Diffusion is not yet implemented or trained.
The initial smoke run uses an analytic context-only baseline. Initial
tasks are synthetic binary problems; success here is not evidence of general-purpose
problem solving or transfer across unrelated datasets.

## Design

```text
labeled task context -> set encoder -> conditional diffusion -> parameter code
                                                               |
                                    fixed decoder/transform <--+
                                              |
                                    classifier weights -> predictions
```

The first target is a fixed linear 8-input, 2-output classifier (18 parameters).
Direct parameter generation avoids unnecessary compression and hidden-neuron
permutation problems at this stage. The later MLP phase will test compact residual
codes plus a separate learned decoder, retaining that two-model architecture only
if reconstruction and baseline comparisons justify it.

## Run locally

Python 3.11+; CPU is sufficient for the foundation. No external dataset is needed.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m p_diff_general.smoke --output artifacts/foundation-smoke
```

Use a fresh output directory. The smoke command evaluates eight training tasks and
eight validation tasks, exports standalone classifiers and writes `tasks.csv` plus
`summary.json`. It performs zero training updates and never opens test tasks.
`ContextEncoder` is an implemented but untrained future conditioning component.

Load an exported classifier without task or source data:

```python
from p_diff_general.parameters import load_classifier
model = load_classifier("artifacts/foundation-smoke/models/validation-000.pt")
# model(x): logits for float32 input x with shape [batch, 8]
```

The benchmark design reserves 256 training, 64 validation and 64 test tasks. Each
has 64 context examples, 256 source-optimization examples and 256 query examples,
using independent random streams. Task splits hold out entire classification
problems, not merely examples from the same problem. This first family tests
in-distribution generalization to unseen tasks; later families test broader transfer.

## Plans and evidence

- [Broader implementation plan](plans/Project_Plan.md)
- [Architecture and current implementation](plans/Architecture.md)
- [Foundation scope and result](plans/01_Foundation.md)
- [Source and baseline pilot results](plans/02_Source_And_Baseline_Results.md)
- [Initial learned conditional baseline results](plans/03_Conditional_Baseline_Results.md)
- [Current pre-diffusion results and readiness](plans/04_Pre_Diffusion_Results.md)
- [Diffusion proposal — not executed](plans/05_Diffusion_Proposal.md)
- [Relationship to the earlier project](docs/provenance.md)

Each substantial training phase needs its own bounded plan and approval. Generation
must not read query labels or source targets. Future candidate selection needs a
separate selection set; final query labels cannot select the best generated model.
A successful result must distinguish useful task conditioning from unconditional
sampling, task-ID memorization, and simply training a classifier on context examples.

## Related project

[p-diff-nn](https://github.com/CoolestToDoIt/p-diff-nn) is the earlier single-task
MNIST prototype (local folder now named `p-diff-mnist`). Its locked negative result
and reconstruction diagnosis motivate this independent experiment; its models and
MNIST test outcomes are not benchmarks for this new project.

## Source collection and learned baseline

The fixed 256/64 source collection and 1,000-update deterministic hypernetwork
run are complete. Validation accuracy was 85.73%, versus 89.82% for context-only
optimization and 91.31% for the analytic baseline. Shuffled contexts scored 51.12%.
The learned baseline misses the provisional accuracy gate; no automatic tuning
or diffusion training followed. Final test tasks remain sealed.

Reproduce the bounded commands in fresh output directories:

```sh
python -m p_diff_general.pilot --output artifacts/source-pilot-repeat
python -m p_diff_general.pilot --full-collection --output artifacts/source-collection-repeat
python -m p_diff_general.hypernetwork --collection artifacts/source-collection-repeat --output artifacts/hypernetwork-repeat
```

Load the learned model for a labeled `Context`:

```python
from p_diff_general.hypernetwork import load_hypernetwork
from p_diff_general.parameters import restore
generator = load_hypernetwork("artifacts/conditional-baseline/hypernetwork.pt")
# classifier = restore(generator(context))
```

A clean editable installation, tests, and smoke run have been verified in `.venv`.
For the recorded macOS ARM64 / Python 3.12 dependency versions, install
`requirements-lock.txt` before installing the editable package. The snapshot is
not a wheel-hashed cross-platform lock.

## Pre-diffusion handoff

The fixed follow-up runs are complete. Validation accuracy is 90.28% for an
analytic-plus-learned correction, 86.51% for direct behavior-loss generation, and
91.31% for analytic inference. The residual model clears the provisional
conditioning/accuracy gates but does not improve on the analytic method.
Gaussian controls and all 20-sample outcomes are preserved. Final test tasks
remain sealed. See the [report](plans/04_Pre_Diffusion_Results.md).

The training-only input bundle, checksums and bounded diffusion proposal are ready.
Check them without fitting a model:

```sh
python -m p_diff_general.pre_diffusion --output artifacts/pre-diffusion --preflight configs/diffusion_proposal.json
```

Load the residual generator without source/query data:

```python
from p_diff_general.pre_diffusion import load_behavior
generator = load_behavior("artifacts/pre-diffusion/behavior_residual.pt")
# classifier = restore(generator(context))
```

Diffusion model/sampler implementation and their mathematical tests are the next
step. No diffusion training has started.
