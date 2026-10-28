"""Audit sources, compare bounded behavior baselines, and prepare diffusion inputs.

This module does not implement or train diffusion and never opens test tasks.
"""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import centroid_parameters
from .hypernetwork import Hypernetwork
from .parameters import manifest, restore, save_classifier
from .pilot import canonicalize
from .tasks import Context, data_hash, sample_task, task_specs


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class TrainingTask:
    task_id: str
    context: Context
    source: Context
    target: torch.Tensor


def audit_collection(collection):
    """Return training data and evaluation identities; no validation labels returned."""
    collection = Path(collection)
    config = json.loads((collection / 'manifest.json').read_text())
    summary = json.loads((collection / 'summary.json').read_text())
    specs = task_specs('train', 256) + task_specs('validation', 64)
    if (config['counts'] != {'train': 256, 'validation': 64}
            or config['manifest'] != manifest() or config['root_seed'] != 42
            or config['examples'] != {'context': 64, 'source': 256, 'query': 256}
            or summary['status'] != 'complete' or summary['completed_tasks'] != 320
            or summary['source_below_70_percent'] or summary['test_tasks_accessed']):
        raise ValueError('Incompatible or incomplete collection')
    if [r['task_id'] for r in config['tasks']] != [s.task_id for s in specs]:
        raise ValueError('Unexpected task identities')
    training, hashes, files = [], [], {}
    for spec, record in zip(specs, config['tasks']):
        if record['spec'] != {'split': spec.split, 'index': spec.index,
                              'root_seed': spec.root_seed, 'family': spec.family}:
            raise ValueError('Spec identity mismatch')
        streams = {}
        for role, count in config['examples'].items():
            streams[role] = sample_task(spec, role, count)
            digest = data_hash(streams[role])
            if digest != record['hashes'][role]:
                raise ValueError('Stream hash mismatch')
            hashes.append(digest)
        file = collection / 'source' / spec.split / f'{spec.index:03d}.pt'
        payload = torch.load(file, map_location='cpu', weights_only=True)
        target = payload['vector']
        if payload['manifest'] != manifest() or payload['metadata']['task_id'] != spec.task_id:
            raise ValueError('Target identity mismatch')
        restore(target)
        torch.testing.assert_close(canonicalize(target), target, atol=1e-6, rtol=1e-5)
        files[str(file.relative_to(collection))] = checksum(file)
        if spec.split == 'train':
            training.append(TrainingTask(spec.task_id, streams['context'], streams['source'], target))
    if len(set(hashes)) != len(hashes):
        raise ValueError('Repeated data streams')
    values = torch.stack([t.target for t in training])
    stats = torch.load(collection / 'train_statistics.pt', map_location='cpu', weights_only=True)
    if stats['count'] != 256 or stats['split'] != 'train' or stats['manifest'] != manifest():
        raise ValueError('Invalid training statistics identity')
    torch.testing.assert_close(stats['mean'], values.mean(0))
    torch.testing.assert_close(stats['std'], values.std(0, unbiased=False))
    audit = {'tasks': 320, 'training_tasks': 256, 'validation_tasks': 64,
             'unique_streams': len(hashes), 'source_manifest_sha256': checksum(collection / 'manifest.json'),
             'source_file_sha256': files, 'test_tasks_accessed': False}
    return training, specs, audit


class BehaviorGenerator(Hypernetwork):
    def __init__(self, mean, std, residual=False):
        super().__init__(mean, std)
        self.residual = residual
        if residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def vectors(self, contexts):
        offsets = self.normalized(contexts) * self.scale
        base = (torch.stack([centroid_parameters(c) for c in contexts])
                if self.residual else self.mean.expand_as(offsets))
        values = offsets + base
        weights, bias = values[:, :16].reshape(-1, 2, 8), values[:, 16:]
        return torch.cat([(weights - weights.mean(1, keepdim=True)).flatten(1),
                          bias - bias.mean(1, keepdim=True)], 1)

    def forward(self, context):
        return self.vectors([context])[0]


def behavior_loss(vectors, x, y):
    """Differentiable classifier execution; restore() deliberately freezes weights."""
    logits = torch.einsum('bni,bci->bnc', x, vectors[:, :16].reshape(-1, 2, 8))
    logits = logits + vectors[:, None, 16:]
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


def load_behavior(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['manifest'] != manifest() or payload['residual'] not in (True, False):
        raise ValueError('Invalid behavior checkpoint')
    state = payload['state_dict']
    with torch.random.fork_rng(devices=[]):
        model = BehaviorGenerator(state['mean'], state['scale'], payload['residual'])
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def train_behavior(training, residual, file):
    if [t.task_id for t in training] != [s.task_id for s in task_specs('train', 256)]:
        raise ValueError('Fitting accepts only the complete training split')
    values = torch.stack([t.target for t in training])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = BehaviorGenerator(values.mean(0), values.std(0, unbiased=False), residual)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(42)
    started, updates, last_loss = time.perf_counter(), 0, None
    for step in range(1000):
        if time.perf_counter() - started >= 300:
            break
        indices = torch.randint(len(training), (32,), generator=rng).tolist()
        batch = [training[i] for i in indices]
        vectors = model.vectors([t.context for t in batch])
        loss = behavior_loss(vectors, torch.stack([t.source.x for t in batch]),
                             torch.stack([t.source.y for t in batch]))
        if not torch.isfinite(loss):
            raise ValueError('Non-finite behavior loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        updates, last_loss = step + 1, loss.item()
    info = {'updates': updates, 'status': 'complete' if updates == 1000 else 'capped',
            'seconds': time.perf_counter() - started, 'last_source_loss': last_loss,
            'seed': 42, 'residual': residual, 'learning_rate': .001, 'batch_size': 32}
    torch.save({'state_dict': model.state_dict(), 'manifest': manifest(), **info}, file)
    return load_behavior(file), info


def correlation(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    denominator = a.norm() * b.norm()
    return (a.dot(b) / denominator).item() if denominator > 0 else None


def paired_summary(differences):
    values = torch.tensor(differences, dtype=torch.float64)
    rng = torch.Generator().manual_seed(42)
    indices = torch.randint(len(values), (2000, len(values)), generator=rng)
    boot = values[indices].mean(1)
    return {'mean_difference': values.mean().item(),
            'task_bootstrap_95_percent': torch.quantile(boot, torch.tensor([.025, .975], dtype=torch.float64)).tolist(),
            'fraction_within_five_points': (values >= -.05).double().mean().item()}


def baseline_gates(rows, summary, methods):
    validation = summary['splits']['validation']['methods']
    gates = {}
    for method in methods:
        relative = validation[method]['versus_context_fit']
        paired_shuffled = paired_summary([float(r[method + '_accuracy']) - float(r[method + '_shuffled_accuracy'])
                                         for r in rows if r['split'] == 'validation'])
        gates[method] = {
            'within_two_points_of_context_fit': relative['mean_difference'] >= -.02,
            'at_least_80_percent_tasks_within_five_points': relative['fraction_within_five_points'] >= .8,
            'conditioning_improvement_interval_above_zero': paired_shuffled['task_bootstrap_95_percent'][0] > 0,
            'paired_versus_shuffled': paired_shuffled,
            'beats_analytic_mean': validation[method]['versus_analytic']['mean_difference'] > 0}
    return gates


class GaussianControl:
    """Training-only full covariance Gaussian; optional analytic context anchor."""
    def __init__(self, training, conditional):
        if not training or any('/train/' not in t.task_id for t in training):
            raise ValueError('Gaussian fitting accepts training tasks only')
        values = torch.stack([t.target - centroid_parameters(t.context) if conditional else t.target
                              for t in training])
        self.mean = values.mean(0)
        centered = values - self.mean
        covariance = centered.T @ centered / len(values)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        self.factor = eigenvectors * eigenvalues.clamp_min(0).sqrt()[None, :]
        self.conditional = conditional

    def sample(self, context, seed):
        rng = torch.Generator().manual_seed(seed)
        value = self.mean + self.factor @ torch.randn(18, generator=rng)
        if self.conditional:
            value = value + centroid_parameters(context)
        return canonicalize(value)


def prepare_bundle(training, output, audit):
    if [t.task_id for t in training] != [s.task_id for s in task_specs('train', 256)]:
        raise ValueError('Bundle requires exact training identities')
    values = torch.stack([t.target for t in training])
    bundle = {'manifest': manifest(), 'split': 'train', 'task_ids': [t.task_id for t in training],
              'context_x': torch.stack([t.context.x for t in training]),
              'context_y': torch.stack([t.context.y for t in training]), 'targets': values,
              'mean': values.mean(0), 'scale': values.std(0, unbiased=False).clamp_min(.05),
              'source_manifest_sha256': audit['source_manifest_sha256']}
    torch.save(bundle, output / 'diffusion_training_bundle.pt')
    return bundle


def load_training_bundle(path, *, expected_sha256=None):
    if expected_sha256 is not None and checksum(path) != expected_sha256:
        raise ValueError('Training bundle checksum mismatch')
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if (bundle['manifest'] != manifest() or bundle['split'] != 'train'
            or bundle['task_ids'] != [s.task_id for s in task_specs('train', 256)]
            or bundle['targets'].shape != (256, 18)
            or bundle['context_x'].shape != (256, 64, 8)
            or bundle['context_y'].shape != (256, 64)):
        raise ValueError('Invalid training bundle identities or shapes')
    for spec, x, y, target in zip(task_specs('train', 256), bundle['context_x'], bundle['context_y'], bundle['targets']):
        context = Context(x, y)
        if data_hash(context) != data_hash(sample_task(spec, 'context', 64)):
            raise ValueError('Training context hash mismatch')
        restore(target)
        torch.testing.assert_close(canonicalize(target), target, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(bundle['mean'], bundle['targets'].mean(0))
    torch.testing.assert_close(bundle['scale'], bundle['targets'].std(0, unbiased=False).clamp_min(.05))
    return bundle


def preflight(output, proposal):
    """Check data/proposal readiness without constructing a diffusion model."""
    output = Path(output)
    readiness = json.loads((output / 'readiness.json').read_text())
    settings = json.loads(Path(proposal).read_text())
    load_training_bundle(output / 'diffusion_training_bundle.pt',
                         expected_sha256=readiness['training_bundle_sha256'])
    if (settings['status'] != 'proposal_not_executed' or not settings['test_tasks_sealed']
            or settings['test_evaluation_authorized'] or settings['automatic_retries'] != 0
            or settings['train_tasks'] != 256 or settings['validation_tasks'] != 64
            or settings['evaluation']['generation_seeds'] != list(range(20))
            or settings['evaluation']['query_selection']
            or settings['evaluation']['candidate_selection'] != 'none'):
        raise ValueError('Proposal violates the bounded experiment contract')
    schedule = settings['diffusion']
    beta = torch.linspace(schedule['beta_start'], schedule['beta_end'], schedule['steps'], dtype=torch.float64)
    if len(beta) < 2 or not ((beta > 0) & (beta < 1)).all():
        raise ValueError('Invalid diffusion schedule')
    terminal_alpha = (1 - beta).prod().item()
    if terminal_alpha / (1 - terminal_alpha) >= .001:
        raise ValueError('Terminal distribution insufficiently noised for Gaussian prior')
    gates = readiness['baseline_gates']['behavior_residual']
    if not readiness['baseline_runs_complete'] or not all(gates[key] for key in (
            'within_two_points_of_context_fit', 'at_least_80_percent_tasks_within_five_points',
            'conditioning_improvement_interval_above_zero')):
        raise ValueError('Conditional baseline prerequisite not met')
    return {'data_and_baselines_ready': True, 'terminal_alpha_bar': terminal_alpha,
            'proposal_sha256': checksum(proposal), 'diffusion_implemented': False,
            'diffusion_trained': False, 'test_tasks_accessed': False,
            'remaining': 'Implement and test the proposed diffusion model before its first training run.'}


def run(collection, previous, output):
    collection, previous, output = Path(collection), Path(previous), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    training, specs, audit = audit_collection(collection)
    write_json(output / 'audit.json', audit)
    prepare_bundle(training, output, audit)
    load_training_bundle(output / 'diffusion_training_bundle.pt')
    models, fitting = {}, {}
    for method, residual in [('behavior_direct', False), ('behavior_residual', True)]:
        models[method], fitting[method] = train_behavior(training, residual, output / f'{method}.pt')
    controls = {name: GaussianControl(training, conditional) for name, conditional in
                [('gaussian_unconditional', False), ('gaussian_conditional', True)]}
    for name, control in controls.items():
        torch.save({'mean': control.mean, 'factor': control.factor, 'conditional': control.conditional,
                    'manifest': manifest(), 'split': 'train', 'count': len(training)}, output / f'{name}.pt')
    values = torch.stack([t.target for t in training])
    scale = values.std(0, unbiased=False).clamp_min(.05)
    rows, diagnostics, samples = [], [], []
    for split in ('train', 'validation'):
        split_specs = [s for s in specs if s.split == split]
        contexts = [sample_task(s, 'context', 64) for s in split_specs]
        for position, (spec, context) in enumerate(zip(split_specs, contexts)):
            vectors = {'analytic': centroid_parameters(context)}
            with torch.no_grad():
                for method, model in models.items():
                    vectors[method] = model(context)
                    vectors[method + '_shuffled'] = model(contexts[(position + 1) % len(contexts)])
            for method, vector in vectors.items():
                save_classifier(output / 'classifiers' / method / split / f'{spec.index:03d}.pt', vector,
                                {'task_id': spec.task_id, 'method': method})
            draws = {method: [control.sample(context, seed) for seed in range(20)]
                     for method, control in controls.items()}
            # Freeze all stochastic samples before opening this task's query data.
            for method, vectors_drawn in draws.items():
                torch.save(torch.stack(vectors_drawn), output / f'{method}-{split}-{spec.index:03d}.pt')
            query = sample_task(spec, 'query', 256)
            row = {'task_id': spec.task_id, 'split': split}
            with torch.no_grad():
                for method, vector in vectors.items():
                    row[method + '_accuracy'] = (restore(vector)(query.x).argmax(1) == query.y).float().mean().item()
                for method, vectors_drawn in draws.items():
                    predictions = torch.stack([restore(v)(query.x).argmax(1) for v in vectors_drawn])
                    accuracy = (predictions == query.y).float().mean(1)
                    row[method + '_accuracy'] = accuracy.mean().item()
                    row[method + '_worst_sample'] = accuracy.min().item()
                    row[method + '_sample_failure_rate'] = (accuracy < .7).float().mean().item()
                    # Pairwise disagreement over all 190 distinct sample pairs.
                    disagreement = [(predictions[a] != predictions[b]).float().mean().item()
                                    for a in range(20) for b in range(a)]
                    row[method + '_prediction_disagreement'] = sum(disagreement) / len(disagreement)
                    row[method + '_weight_variance'] = torch.stack(vectors_drawn).var(0, unbiased=False).mean().item()
                    for seed, score in enumerate(accuracy.tolist()):
                        samples.append({'task_id': spec.task_id, 'split': split, 'method': method,
                                        'seed': seed, 'accuracy': score})
                target = torch.load(collection / 'source' / split / f'{spec.index:03d}.pt', weights_only=True)['vector']
                old = torch.load(previous / 'hypernetwork' / split / f'{spec.index:03d}.pt', weights_only=True)['vector']
                context_fit = torch.load(collection / 'context' / split / f'{spec.index:03d}.pt', weights_only=True)['vector']
                row['context_fit_accuracy'] = (restore(context_fit)(query.x).argmax(1) == query.y).float().mean().item()
                row['previous_accuracy'] = (restore(old)(query.x).argmax(1) == query.y).float().mean().item()
                old_logits, target_logits = restore(old)(query.x), restore(target)(query.x)
                old_direction = old[:8] - old[8:16]
                target_direction = target[:8] - target[8:16]
                diagnostics.append({'task_id': spec.task_id, 'split': split,
                    'normalized_weight_mse': ((old - target) / scale).square().mean().item(),
                    'direction_cosine': F.cosine_similarity(old_direction, target_direction, dim=0).item(),
                    'query_loss': F.cross_entropy(old_logits, query.y).item(),
                    'source_prediction_disagreement': (old_logits.argmax(1) != target_logits.argmax(1)).float().mean().item(),
                    'accuracy_gap_to_analytic': row['previous_accuracy'] - row['analytic_accuracy']})
            rows.append(row)
    write_csv(output / 'tasks.csv', rows)
    write_csv(output / 'diagnostics.csv', diagnostics)
    write_csv(output / 'samples.csv', samples)
    summary = {'fitting': fitting, 'test_tasks_accessed': False, 'diffusion_trained': False,
               'generation_seeds': list(range(20)), 'splits': {}}
    methods = list(vectors) + list(controls) + ['context_fit', 'previous']
    for split in ('train', 'validation'):
        subset = [r for r in rows if r['split'] == split]
        diags = [r for r in diagnostics if r['split'] == split]
        summary['splits'][split] = {
            'methods': {method: {'mean_accuracy': sum(r[method + '_accuracy'] for r in subset) / len(subset),
                                'worst_task_accuracy': min(r[method + '_accuracy'] for r in subset),
                                'task_failure_rate_below_70_percent': sum(r[method + '_accuracy'] < .7 for r in subset) / len(subset),
                                'versus_context_fit': paired_summary([r[method + '_accuracy'] - r['context_fit_accuracy'] for r in subset]),
                                'versus_analytic': paired_summary([r[method + '_accuracy'] - r['analytic_accuracy'] for r in subset])}
                        for method in methods},
            'weight_error_accuracy_gap_correlation': correlation([r['normalized_weight_mse'] for r in diags], [r['accuracy_gap_to_analytic'] for r in diags]),
            'direction_cosine_accuracy_gap_correlation': correlation([r['direction_cosine'] for r in diags], [r['accuracy_gap_to_analytic'] for r in diags])}
    write_json(output / 'summary.json', summary)
    gates = baseline_gates(rows, summary, models)
    write_json(output / 'readiness.json', {
        'training_bundle_sha256': checksum(output / 'diffusion_training_bundle.pt'),
        'training_tasks': 256, 'validation_tasks': 64, 'test_tasks_accessed': False,
        'diffusion_implemented': False, 'diffusion_trained': False,
        'baseline_runs_complete': all(info['status'] == 'complete' for info in fitting.values()),
        'baseline_gates': gates,
        'next_step': 'Review measured baseline gates and the bounded diffusion proposal before training.'})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection')
    parser.add_argument('--previous')
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight', metavar='PROPOSAL', help='Check existing inputs only; never train')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.output, args.preflight), indent=2))
    elif args.collection and args.previous:
        print(json.dumps(run(args.collection, args.previous, args.output), indent=2))
    else:
        parser.error('Provide --collection and --previous, or --preflight')


if __name__ == '__main__':
    main()
