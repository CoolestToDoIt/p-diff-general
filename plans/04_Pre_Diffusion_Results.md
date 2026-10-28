# Pre-diffusion results and handoff

Completed October 27, 2026 under the [bounded plan](04_Pre_Diffusion.md). Both
follow-ups completed exactly 1,000 updates, with no checkpoint selection, tuning,
extra training seed, or retry. Direct behavior fitting took 6.57 seconds; residual
behavior fitting took 8.66 seconds. All work ran in the project-local environment.

## Measured validation results

All values are means over the same 64 validation tasks. Gaussian methods average
all 20 fixed samples within each task, then average tasks; no best sample is chosen.

| Method | Mean accuracy | Worst task accuracy | Tasks within 5 points of context fitting |
|---|---:|---:|---:|
| Analytic context baseline | 91.31% | 75.39% | 100% |
| Analytic + learned behavior correction | 90.28% | 69.14% | 95.31% |
| Context-only Adam | 89.82% | 72.27% | 100% |
| Direct behavior-loss hypernetwork | 86.51% | 64.84% | 70.31% |
| Conditional residual Gaussian | 85.49% | 60.53% | 62.50% |
| Original weight-MSE hypernetwork | 85.73% | 56.64% | 70.31% |
| Unconditional Gaussian | 50.53% | 42.40% | 0% |

Rotating task contexts reduced the residual model to 50.60% and the direct
behavior model to 51.14%. The residual model's paired gain over its shuffled
control was 39.68 percentage points, with a task-bootstrap 95% interval of
[34.86, 44.31]. This supports useful context dependence on this task family.

The residual model exceeds context-only fitting by 0.46 points, but its paired
95% interval spans [-0.35, 1.08]; superiority to context fitting is not established.
It trails analytic inference by 1.03 points, interval [-1.69, -0.54]. The analytic
method remains the strongest measured inference baseline.

The residual model clears the provisional pre-diffusion baseline gates: mean
accuracy within two points of context fitting, at least 80% of tasks within five
points, and positive paired conditioning benefit. This is not a claim that a
learned correction improves the analytic method. The direct model still fails
the first two gates. One seed and reused development tasks limit the conclusions.

## Diagnosis

On validation tasks, normalized weight error correlated with the original model's
accuracy gap to analytic inference at -0.46; direction cosine correlated at +0.68.
These are descriptive task-level correlations. They suggest that weight MSE alone
is an imperfect behavioral proxy; they do not identify a causal failure mechanism.

Changing the objective to source-example cross-entropy improved the direct model
from 85.73% to 86.51%, but did not close the gap. Anchoring predictions to analytic
weights reached 90.28%. This supports retaining the analytic anchor as a strong
control. The two follow-ups also differ in initialization/base representation,
so the improvement cannot be attributed to the objective alone.

Source labels were optimization supervision only on training tasks, using 256
source examples per task. Inference uses 64 context examples and no gradients.
Validation labels scored fixed outputs and never chose a training checkpoint.
All classifiers/samples were saved before their evaluation query was sampled;
the prior integrity audit independently regenerated query streams to check hashes,
without providing those labels to the fitting API.

## Data and controls ready

The audit checked exact task IDs/specifications, 960 distinct role-stream hashes,
finite canonical source vectors, target metadata, training-only mean/std, and
SHA-256 checksums of every source classifier. The training bundle contains only
256 training contexts and source parameter targets; it has no validation targets,
query examples, source examples, task teacher parameters, or test data.

The bundle loader verifies exact training IDs, context hashes, canonical targets,
statistics and an optional externally supplied checksum. The preflight always
supplies the checksum recorded in readiness metadata. Gaussian distributions were
fit solely to training targets (or source-minus-analytic residuals) and exported
with mean/covariance factors. Each method sampled seeds 0–19 independently through
local RNGs. All 12,800 sample outcomes are retained; samples within tasks are not
independent evaluation units. Task summaries include weight variance, prediction
disagreement over all 190 sample pairs, worst sample, and sample failure rate.

Durable evidence:

- [Task metrics](results/04_pre_diffusion_tasks.csv),
  [original-model diagnostics](results/04_pre_diffusion_diagnostics.csv), and
  [all stochastic samples](results/04_pre_diffusion_samples.csv).
- [Summary and paired intervals](results/04_pre_diffusion_summary.json),
  [readiness checks](results/04_pre_diffusion_readiness.json), and
  [proposal preflight](results/04_pre_diffusion_preflight.json).

Full checkpoints, Gaussian distributions, generated classifiers, sampled vectors,
audit checksums, and the training bundle remain in ignored
`artifacts/pre-diffusion/`. Bootstrap intervals use 2,000 task resamples with a
fixed local seed. Runtime comparisons are not claimed: fitting times were measured,
but this run did not separately time every export/scoring/sampling stage.

## Boundary: ready to implement diffusion, not yet trained

[The proposed experiment](05_Diffusion_Proposal.md) and its
[machine-readable settings](../configs/diffusion_proposal.json) are ready. The
preflight verifies the bundle checksum, baseline gates, fixed seed/sample contract,
sealed test tasks, and a noise schedule whose terminal distribution matches a
Gaussian prior closely enough (terminal cumulative alpha about 0.0000404).
A 100-step linear schedule ending at beta 0.02 would leave too much signal; the
proposal therefore specifies 1,000 steps. A test rejects that prior mismatch.

Diffusion is neither implemented nor trained. The next action is to implement the
proposed model, noise process and reverse sampler, pass their mathematical and
leakage checks, and only then run the fixed bounded experiment. Final test tasks
remain sealed. No compression, decoder, image tasks, or dataset download occurred.

Re-run the existing-input preflight without any training:

```sh
python -m p_diff_general.pre_diffusion --output artifacts/pre-diffusion --preflight configs/diffusion_proposal.json
```

Reproduce the two fixed baseline experiments and diagnostics into a fresh directory:

```sh
python -m p_diff_general.pre_diffusion --collection artifacts/full-source-collection --previous artifacts/conditional-baseline --output artifacts/pre-diffusion-repeat
```
