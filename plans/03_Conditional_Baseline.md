# Conditional baseline build

October 27, 2026. Proceeding under the user's instruction to keep building, after
all 40 source-pilot tasks cleared the fixed integrity and 70% source-accuracy gates.

Collect targets for the declared 256 training and 64 validation tasks using the
same zero-init Adam 0.03 / 200-update source and context fitting protocol. Reuse
identical task definitions, never test tasks. Preserve every result; any source
below 70% is flagged for review. Collection has the same 30-minute cap.

Train one deterministic hypernetwork: existing class-pooling context encoder
(width 32, embedding 64), followed by a linear 18-output head. Predict normalized
canonical source weights using training-only mean/std, with std clamped to 0.05.
Use seed 42, Adam 0.001, 1,000 updates, minibatches of 32 tasks sampled from a
separate deterministic generator, and weight MSE. No query labels enter fitting.
Freeze the final update; no validation checkpoint selection or hyperparameter
search. CPU fitting cap: five minutes. Preserve the final available model if capped.

Evaluate all training/validation tasks against analytic context generation,
context/source optimization, and a pooled training-target mean. Rotate contexts
by one task within each split as a conditioning-shuffle control. Save generated
classifiers before scoring. Report every task and all controls, including failure
or underperformance. This checks deterministic conditioning, not diffusion.
Do not open final test tasks or automatically tune after this run. Diffusion needs
its own concrete implementation plan informed by this result.
