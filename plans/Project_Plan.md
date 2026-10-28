# Task-conditioned neural parameter generation

Created October 27, 2026. Foundation authorized by the request to make the broader
implementation plan and start the project. Substantial later training phases need
separate plans and approval. This is an independent project linked to
[p-diff-nn](https://github.com/CoolestToDoIt/p-diff-nn).

## Objective and honest scope

Train one conditional diffusion model that takes labeled examples describing a
problem and generates a classifier's weights. On a new task, the generated classifier
predicts immediately; no classifier gradient updates or hidden access to query
labels occurs after generation. Generator/source training and conditioning costs
still count toward total cost. There is no claim of supporting arbitrary tasks,
arbitrary architectures, natural-language specifications or unseen modalities.

The first success is transfer to unseen tasks from a declared synthetic task family.
A later image milestone and held-out task families are required before broader
claims. A task is a distribution/labeling problem, not a checkpoint, minibatch or
random seed of training the same classifier on the same labels.

## System

Training: task specification -> independent context/source/query samples -> source
classifier targets -> canonical parameter vectors -> conditional generator training.
Labeled context -> permutation-invariant, label-sensitive task embedding. Source
weights are training targets; neither source examples nor query labels are inputs
to inference-time generation.

Generation: context examples -> task encoder -> conditional reverse diffusion ->
parameter vector or compact code -> optional frozen residual decoder -> classifier.
The task ID, procedural teacher parameters and source optimizer state are forbidden
conditioning inputs. Generated classifiers load without generator, context or data.

Initial target: a fixed 8-to-2 linear classifier, 18 float32 parameters. Canonicalize
the shared-logit ambiguity by subtracting classwise mean weights/biases. This removes
one source of non-identifiability without requiring nonlinear neuron alignment.
No autoencoder is needed for 18 parameters. A later fixed 8-to-32-to-2 MLP has 354
parameters; test alignment and reconstruction there before a learned decoder.

The intended compressed architecture has two learned parts: a conditional diffusion
code generator and a deterministic parameter decoder. The task encoder is another
learned conditioning component. A stochastic expansion model is a later experiment,
not a prerequisite. Mean anchoring, linear PCA and direct parameter generation
remain baselines; additional models must earn their complexity.

## Task benchmark and leakage controls

Version 1: binary classification in eight dimensions. Each task draws its own
orientation, class separation, shift, diagonal covariance and label orientation.
Both classes are balanced; contexts preserve task-local class identities. Tasks
share a linear architecture and input/output meaning. Contexts contain examples,
not latent teacher descriptions. The family is easy by design: diagnose whether
conditioning works before testing complex architectures.

Reserve 256 training, 64 validation and 64 test task identities, seed 42. Entire
tasks belong to one split. Per task, use independently seeded streams for 64 context,
256 source-optimization and 256 query examples. Store IDs, hashes and generation
version. Shared family parameters across splits mean this is within-family task
transfer, not out-of-family evidence. Test generation is explicitly blocked during
development; tests verify the guard without generating test data.

Train task encoders, diffusion, normalization and compression on training tasks
only. Validation tasks choose declared hyperparameters/checkpoints. Validation
query labels may score those choices; final test query labels may only report locked
performance. Never choose the best generated candidate using final query labels.
If candidate selection is tested, allocate a separate selection stream and compare
equal candidate budgets. Save generated vectors and selections before test scoring.

Source fitting must use the declared source stream, never query labels. Because
source targets use more task examples than the 64-example context, disclose this
training supervision advantage and compare against a context-only classifier. A
source classifier is a reference, not an inference-time option available for free.

## Phased implementation

| Phase | Work | Completion / stop condition |
|---|---|---|
| 1: foundation | Task IDs/streams, classifier codec, context encoder interface, analytic baseline, smoke/test checks | Repeatable exports and no query-to-conditioning path; no learned model claims |
| 2: bounded source pilot | Train compatible linear source models on 32 train + 8 validation tasks; context-only optimization and analytic baselines | Source fitting works, target canonicalization checked, accuracy/cost/variance evidence saved |
| 3: conditional baselines | Prepare full 256/64 training/validation task collection; deterministic context-to-weight hypernetwork and pooled/unconditional controls | Confirm unseen-task conditioning works and quantify simple baseline strength |
| 4: conditional diffusion | Epsilon-prediction residual MLP, task embedding, direct canonical 18D weights; fixed training budget/seed and schedule | Equal-budget comparison with deterministic, analytic, Gaussian/empirical and unconditioned controls |
| 5: locked task evaluation | Lock artifacts/code/config/seeds; generate classifiers for 64 held-out tasks; report every sample and task | No test-driven selection; preserve negative outcomes; no automatic retries |
| 6: fixed MLP and compression | Test 354D targets, shared initialization/alignment, mean-anchored residual decoder vs PCA and direct weights | Reconstruction preserves behavior and variation before another generator is trained |
| 7: image tasks and broader transfer | Specify labeled image tasks and semantic group-level holdouts; fixed image classifier or feature interface | Entire problem definitions held out; reuse of images/classes/feature training disclosed |
| 8: release | Fresh-environment reproduction, standalone generator/classifiers, notebook, final report and artifact checksums | Another user can reproduce a declared result and understand its limits |

Phases 1-3 are implemented through the first fixed deterministic baseline run;
see [measured results](03_Conditional_Baseline_Results.md). The user authorized
continued building on October 27, 2026. Phases 4-8 remain a roadmap; the failed
conditional baseline gate led to the separately authorized
[bounded pre-diffusion follow-up](04_Pre_Diffusion.md), now complete.
See [current results](04_Pre_Diffusion_Results.md): the residual model clears
the provisional gate, and the analytic baseline remains strongest. The
[diffusion proposal](05_Diffusion_Proposal.md) is ready; no diffusion run occurred. One CPU pilot at a time, explicit cap and partial-output preservation.
The first source pilot has a separate concrete plan. Do not continue tuning after a
failed gate without a revised plan.

## Baselines and evaluation

Compare context-only diagonal-centroid/LDA-style weights, ordinary context-only
classifier optimization, a deterministic hypernetwork, task-agnostic pooled weights,
unconditional diffusion, simple conditional parameter-distribution sampling, and
conditional diffusion. Include source-fitted references and chance-level constants.
A training-target bootstrap must not use a held-out task's source weights. Retrieval
may use context embeddings but must retrieve only training tasks' targets.

For each unseen task, report accuracy distribution across a fixed 20 generation seeds,
mean/median/worst, task-level failure rate, and prediction/weight diversity. Avoid
best-of-20 query accuracy. Task is the primary aggregation unit; samples within tasks
are correlated. Bootstrap intervals over tasks, not individual queries or checkpoints.
Report context sizes and independent generator training seeds when approved.

Provisional first-family target, frozen before the learned pilot starts: mean
per-task accuracy within two percentage points of the context-only optimized
classifier, with at least 80% of tasks within five points, and accuracy above a
condition-shuffled/unconditioned control. Whether diffusion beats the deterministic
or analytic baseline is a separate measured claim; passing alone is insufficient.
Report mean/worst and all paired task differences rather than relying on one target.
These are intended pilot targets, not measured results or promises.

Measure source collection, generator training, context encoding, sampling/export,
candidate selection and evaluation separately. Compare ordinary training and analytic
inference at equal context size, plus total amortized cost. Report memory and bundle
size. No speed advantage is claimed from excluding the upstream training expense.

## Risks and decision points

- Task-ID memorization: task IDs never enter conditioning; hold out entire tasks.
- Context insufficiency: compare to context-only fitting, not only richer source targets.
- Parameter symmetry: use canonical linear logits first; align MLP units if later needed.
- Compression collapse/template bias: direct-weight and mean/PCA baselines; dual gates
  on reconstructed accuracy and centered variation before training a latent generator.
- Apparent diversity from bad samples: pair accuracy and diversity; report all failures.
- Weak evidence of generality: within-family and cross-family results labeled separately;
  image tasks need semantic holdout design before downloads/training.
- Diffusion adds no value: preserve the negative result and prefer simpler models.

## Deliverables and references

Python/PyTorch package, versioned JSON benchmark config, task/target manifests,
source and generation commands, leakage/mathematical tests, standalone checkpoint
bundles, full task-level CSVs, plots, notebook and final report. Dependencies and
hardware/runtime versions recorded with experiments; artifacts remain ignored until
a separately reviewed release location is chosen.

References: [HyperNetworks](https://arxiv.org/abs/1609.09106) motivates a direct
weight-generation baseline; [Conditional Neural Processes](https://arxiv.org/abs/1807.01613)
motivates conditioning on observation sets; [DDPM](https://arxiv.org/abs/2006.11239)
is the planned diffusion baseline. These inform the design; no paper code is copied.
