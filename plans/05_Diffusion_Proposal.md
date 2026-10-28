# First diffusion experiment proposal — not executed

This is the handoff boundary requested by the user: complete the pre-diffusion work,
then stop before diffusion implementation or training. This proposal does not open
final test tasks or authorize a run. The exact proposed settings live in
[diffusion_proposal.json](../configs/diffusion_proposal.json).

## Question and target

Does conditional diffusion produce useful classifiers for unseen tasks, compared
with the measured deterministic, analytic, Gaussian, and unconditional controls?
The experiment uses direct, canonical 18D linear weights; no parameter decoder,
autoencoder, MLP classifier, image dataset, or compression is involved.

Read the pre-diffusion results before choosing to proceed. A useful conditioning
result does not imply diffusion beats analytic inference. Earlier failed baselines
remain part of the report; validation has already informed development and is not
a final unbiased test.

## Inputs and isolation

The prepared `artifacts/pre-diffusion/diffusion_training_bundle.pt` has only the
256 training tasks' context tensors and source targets, training mean/scale,
classifier manifest, and source manifest checksum. Verify its checksum against
`readiness.json`, then load it through `load_training_bundle`. Validate the exact
training IDs, canonical targets, context hashes, shapes, and statistics before
fitting. Task IDs identify records but must not enter the model.

The generator's public interface accepts only labeled context and a local noise
seed. Keep query data and validation targets out of the optimizer's dataset and
API. Validation scoring happens after generated parameters are frozen/exported.
No test access flag or test-scoring command belongs in the training runner.

## Fixed model and budget

Train one conditional epsilon-prediction model and one unconditional control, each
with seed 42, Adam 0.001, batch 64, exactly 10,000 updates, one CPU thread, and a
10-minute wall-time cap. Source data collection costs are reported separately.
Use the existing set encoder (width 32, embedding 64), a 32D sinusoidal timestep
embedding, and a denoiser with width 128 and three residual SiLU MLP blocks. The
unconditional run uses the same denoiser with a zero context embedding; report
the extra encoder parameter/training cost of the conditional run explicitly.

Fit standardized training targets with std clamped to 0.05. Use 1,000 forward noise
steps with linearly spaced betas from 0.0001 to 0.02, uniform timestep sampling,
and plain epsilon MSE. This yields terminal cumulative alpha near 0.00004,
so the terminal distribution is close to the Gaussian sampling prior. Generation uses the complete DDPM reverse chain, without
guidance, clipping, candidate selection, or shortened-step sampling. Unstandardize
and canonicalize weights before standalone export. Keep random streams separate
for initialization, task batches, timesteps, forward noise, and generation.

Checkpoint the last update; preserve partial output on a cap and report capped
runs separately. Never select a checkpoint using query accuracy. No tuning,
extra training seeds, budget expansion, or retry follows an unsuccessful run.
Evaluation has a separate 20-minute total wall-time cap; preserve partial samples
and mark incomplete results rather than reducing the declared sample budget.

## Comparisons and scoring

Use all 64 validation tasks and fixed generation seeds 0–19. Evaluate correct
conditioning and rotated contexts using the same noise seeds. Evaluate the
unconditional model with the same candidate budget. Include both Gaussian controls,
the original weight-MSE hypernetwork, both behavior baselines, context-only Adam,
source references, and analytic inference. Treat sources' 256-label supervision
and context inference's 64 labels explicitly.

Save every generated vector before scoring. Report each sample and task, with
sample means, median, worst, prediction and weight diversity, sample failures,
task failures below 70%, and task-bootstrap paired intervals. Never report the
best-of-20 query result as usable performance. Account for encoding, sampling,
export, scoring, training, and source collection separately. Do not claim speed
from omitting upstream training cost.

The provisional gate requires mean task accuracy within two percentage points
of context-only optimization, at least 80% of tasks no more than five points below
it, and a positive paired improvement over shuffled conditioning and unconditional
generation. Report bootstrap uncertainty; a strict conditioning claim requires
the paired interval's lower bound above zero. Compare against the strongest measured
baseline independently: merely passing this gate is not evidence of added value
for diffusion. Preserve any negative result.

## Implementation checks required before the first run

- Exact forward-noise and reverse-posterior algebra, including the noiseless final
  reverse step and timestep indexing.
- Finite epsilon loss and gradients; correct shapes and training-only normalization.
- Local seed reproducibility without changing caller RNG state.
- Conditioning changes generated weights, while context order does not.
- Standalone checkpoint and classifier roundtrips; manifest/checksum rejection.
- No query/validation labels in fitting, no test-task generation, no output overwrite,
  and partial-checkpoint preservation on the wall-time cap.

After implementation and these checks, run only the fixed experiment. Final test
access remains a later, separately locked evaluation decision.
