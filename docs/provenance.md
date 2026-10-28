# Relationship and provenance

Related repository: https://github.com/CoolestToDoIt/p-diff-nn
New repository: https://github.com/CoolestToDoIt/p-diff-general

The earlier repository is currently checked out locally as `p-diff-mnist`.
Reference commit: `2fe703df658f3e1a3b9f9ad1298cf4cfea6941dc`.
Its single-task MNIST experiment generated usable weights but did not demonstrate
an advantage over simpler methods. Follow-up diagnostics found large mean-template
reconstruction bias and compressed source variation. Keep those results intact.

This repository's foundation was written independently on October 27, 2026. No
source files, trained weights, datasets or artifact binaries were copied from the
old repository. Conceptual lessons carried over: strict parameter manifests,
training-only fitted statistics, dataset-free classifier loading, fixed evaluation
rules, full distributions, and explicit negative results. The initial 18-parameter
linear target intentionally postpones compression and hidden-neuron alignment.

No license file was found in the earlier repository at the reference commit. If
code is copied later, record its original path/commit and establish the applicable
license before redistribution. No third-party paper implementation is included.

Research references (conceptual, not implementation dependencies):
- Ha, Dai and Le, *HyperNetworks*: https://arxiv.org/abs/1609.09106 — networks producing
  another network's parameters; motivates the deterministic conditioning baseline.
- Garnelo et al., *Conditional Neural Processes*: https://arxiv.org/abs/1807.01613 —
  prediction conditioned on sets of observations; informs the context-set interface.
- Ho et al., *Denoising Diffusion Probabilistic Models*:
  https://arxiv.org/abs/2006.11239 — planned epsilon-prediction diffusion baseline.
