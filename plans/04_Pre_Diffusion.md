# Pre-diffusion diagnosis and bounded baselines

Authorized October 27, 2026 by the request to continue until the pre-diffusion work
is ready. No diffusion training or final-test access is included.

1. Audit all 256/64 task identities, stream hashes, finite canonical targets and
   training-only statistics. Diagnose the existing hypernetwork using per-task
   normalized weight error, direction agreement, query loss and paired accuracy
   differences. Summarize correlations descriptively, without causal claims.
2. Run exactly two fixed follow-ups, not a search: a direct context-to-weight
   hypernetwork trained on source-example cross-entropy, and the same model
   predicting a residual correction to the analytic context baseline. This tests
   behavior loss and the analytic inductive bias separately. Each uses seed 42,
   Adam 0.001, 1,000 updates, batch 32 tasks, existing encoder width 32 / embedding
   64, and a five-minute CPU wall-time cap. Source labels are training supervision;
   inference still uses context64 only. Residual head starts at zero and scales
   corrections by training-only source std clamped to 0.05. No early stopping,
   validation checkpoint choice, extra seeds or automatic retries.
3. Export fixed classifiers before query scoring. Score all tasks, rotated-context
   controls, and paired differences to context fitting and analytic inference.
   Report mean, worst, and fraction within five points of context-only fitting.
   Bootstrap paired mean intervals over validation tasks, never individual rows.
4. Implement training-only conditional residual Gaussian and unconditional Gaussian
   controls with a fixed 20-sample budget. Report all samples, task-level means,
   prediction/weight diversity and failure rates; no best-of-query selection.
5. Prepare a concrete diffusion proposal and a machine-readable, integrity-checked
   training bundle. Keep validation targets out of training bundles. Add tests of
   loss gradients, train/evaluation boundaries, checkpoint reload, sampling seeds,
   and readiness rejection for corrupt data. Stop before diffusion implementation
   or training; disclose any failed baseline gate in the proposal.

All outcomes survive, including negative ones. Validation is development data
already examined in earlier runs. This work cannot establish final-test success.
