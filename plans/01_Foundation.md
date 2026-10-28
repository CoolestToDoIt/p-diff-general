# Phase 1: task-conditioned generation foundation

Authorization: approved October 27, 2026 by the request to make the broader plan and
start the project. Scope is code, synthetic smoke checks and plans; no generator
training, downloads, paid compute or final test access.

Implemented: reproducible 8D binary task family; whole-task split identities;
independent context/source/query streams; explicit test access guard; strict
18-parameter classifier manifest and standalone export/reload; example-set context
encoder; analytic context-only diagonal-centroid parameter generator; CPU smoke CLI.
The encoder is untrained. The analytic generator does not establish diffusion success.

Tests cover deterministic streams without global RNG consumption, task/role
separation, blocked test access, malformed contexts, exact codec/logit roundtrips,
manifest rejection, no output overwrite, context-order invariance, label sensitivity,
and an offline no-training/no-test smoke export.

Run `python -m pytest -q` and `python -m p_diff_general.smoke` after installation.
Use a fresh output directory. Smoke scope is at most eight tasks per training and
validation split with context64/query256. Binaries stay under ignored `artifacts/`;
commit task-level evidence and a concise result report under `plans/`.

The foundation was verified using the existing local PyTorch environment from the
renamed earlier project to avoid an unnecessary dependency download. The new source
imports only its own package and PyTorch; the old project is not a runtime dependency.
A fresh-environment installation check remains a release requirement.

The subsequent [bounded source/baseline pilot](02_Source_And_Baseline_Pilot.md)
was authorized by the request to continue building and is now complete; see its
[results](02_Source_And_Baseline_Results.md).
