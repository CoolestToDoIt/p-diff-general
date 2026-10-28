# Phase 2 proposal: source targets and baseline fitting

Authorization: approved by the October 27, 2026 request to keep building.
Completed once; see [results](02_Source_And_Baseline_Results.md).

Question: can we prepare consistent, useful weight targets for multiple tasks, and
how well do simple context-only classifiers already perform?

Scope upon approval:
1. Use the first 32 training and 8 validation task IDs from benchmark version 1;
   root seed42, context64/source256/query256. No test tasks, external data or task-family
   changes. Save the exact spec, role hashes and classifier manifest before fitting.
2. Fit one linear source classifier per task using only source examples: zero init,
   full-batch cross-entropy, Adam0.03, exactly 200 updates, no search/retries. Canonicalize
   logits after fitting; verify identical predictions and finite target vectors.
3. Fit one context-only classifier per task with the same optimizer, initialization
   and update count on context examples. Also evaluate the analytic diagonal-centroid
   context baseline and constant-class reference. Query examples only score results;
   no query-driven training or candidate/checkpoint selection.
4. Record all 40 tasks' source/context/baseline accuracies, training losses/times,
   weight variation and source-to-context prediction disagreement. Save loadable
   source targets and training-only parameter mean/std; keep validation targets
   separate and forbidden as generator training rows.
5. Integrity gates: no overlapping task IDs or role streams; no query labels in
   conditioning or optimization; canonicalization preserves logits up to an equal
   shared shift and preserves predictions. Report conditioning/task difficulty.
   If a source classifier is below 70% query accuracy, retain it and flag the pilot
   for review rather than filtering it out or changing task generation.

Budget: at most 30 CPU minutes total, one fixed source/context run per task, no new
packages, paid compute, learned hypernetwork, diffusion training or decoder work.
Preserve partial models and metrics if capped. Implementation and synthetic tests,
all-task CSVs, timing/runtime provenance and report belong in the deliverable.

This pilot validates the task/target pipeline; it does not show generator transfer.
A separate plan must authorize the full task collection and learned conditional
baseline. Official test tasks remain unopened until a later locked evaluation.
