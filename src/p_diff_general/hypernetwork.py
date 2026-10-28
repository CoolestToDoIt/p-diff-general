"""Fixed-budget deterministic context-to-weight baseline, with sealed test tasks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import ContextEncoder, centroid_parameters
from .parameters import load_classifier, manifest, restore, save_classifier
from .pilot import canonicalize, data_hash
from .tasks import TaskSpec, sample_task, task_specs


class Hypernetwork(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.encoder = ContextEncoder()
        self.head = nn.Linear(64, 18)
        self.register_buffer('mean', mean.clone())
        self.register_buffer('scale', std.clamp_min(.05).clone())

    def normalized(self, contexts):
        return self.head(torch.stack([self.encoder(context) for context in contexts]))

    def forward(self, context):
        vector = self.normalized([context])[0] * self.scale + self.mean
        return canonicalize(vector)


def load_hypernetwork(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['manifest'] != manifest():
        raise ValueError('Classifier manifest mismatch')
    state = payload['state_dict']
    with torch.random.fork_rng(devices=[]):
        model = Hypernetwork(state['mean'], state['scale'])
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def run(collection, output):
    collection, output = Path(collection), Path(output)
    config = json.loads((collection / 'manifest.json').read_text())
    source_summary = json.loads((collection / 'summary.json').read_text())
    if (config['counts'] != {'train': 256, 'validation': 64}
            or config['manifest'] != manifest() or source_summary['status'] != 'complete'
            or source_summary['completed_tasks'] != 320):
        raise ValueError('Need a complete, compatible 256/64 source collection')
    if source_summary['source_below_70_percent']:
        raise ValueError('Source collection failed the accuracy gate; review before fitting')
    expected = task_specs('train', 256) + task_specs('validation', 64)
    if [record['task_id'] for record in config['tasks']] != [spec.task_id for spec in expected]:
        raise ValueError('Unexpected task identities or ordering')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    specs, contexts, targets = [], [], []
    for record in config['tasks']:
        spec = TaskSpec(**record['spec'])
        if spec.task_id != record['task_id']:
            raise ValueError('Task identity mismatch')
        if spec.split not in {'train', 'validation'}:
            raise ValueError('Test tasks forbidden')
        context = sample_task(spec, 'context', 64)
        if data_hash(context) != record['hashes']['context']:
            raise ValueError('Context hash mismatch')
        file = collection / 'source' / spec.split / f'{spec.index:03d}.pt'
        payload = torch.load(file, map_location='cpu', weights_only=True)
        if payload['manifest'] != manifest() or payload['metadata']['task_id'] != spec.task_id:
            raise ValueError('Source target identity or manifest mismatch')
        restore(payload['vector'])
        specs.append(spec)
        contexts.append(context)
        targets.append(payload['vector'])
    train_indices = [i for i, spec in enumerate(specs) if spec.split == 'train']
    values = torch.stack([targets[i] for i in train_indices])
    # Recompute training statistics, rather than trusting a mixed-split cache.
    mean, std = values.mean(0), values.std(0, unbiased=False)
    rng = torch.Generator().manual_seed(42)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = Hypernetwork(mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    normalized_targets = (values - mean) / model.scale
    started = time.perf_counter()
    updates, last_loss = 0, None
    for step in range(1000):
        if time.perf_counter() - started >= 300:
            break
        indices = torch.randint(len(train_indices), (32,), generator=rng)
        batch = [contexts[train_indices[i]] for i in indices.tolist()]
        prediction = model.normalized(batch)
        loss = F.mse_loss(prediction, normalized_targets[indices])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        updates, last_loss = step + 1, loss.item()
    training_seconds = time.perf_counter() - started
    model.eval().requires_grad_(False)
    checkpoint = {'state_dict': model.state_dict(), 'manifest': manifest(), 'seed': 42,
                  'updates': updates, 'learning_rate': .001, 'batch_size': 32,
                  'source_manifest_sha256': hashlib.sha256((collection / 'manifest.json').read_bytes()).hexdigest(),
                  'torch': str(torch.__version__)}
    torch.save(checkpoint, output / 'hypernetwork.pt')
    loaded = load_hypernetwork(output / 'hypernetwork.pt')
    rows = []
    for split in ('train', 'validation'):
        indices = [i for i, spec in enumerate(specs) if spec.split == split]
        for position, i in enumerate(indices):
            spec, context = specs[i], contexts[i]
            with torch.no_grad():
                vector = loaded(context)
                torch.testing.assert_close(vector, model(context))
                shuffled = loaded(contexts[indices[(position + 1) % len(indices)]])
            vectors = {'hypernetwork': vector, 'shuffled': shuffled,
                       'pooled': canonicalize(mean), 'analytic': centroid_parameters(context)}
            # Freeze every generated classifier before opening the scoring stream.
            for method, v in vectors.items():
                save_classifier(output / method / split / f'{spec.index:03d}.pt', v,
                                {'task_id': spec.task_id, 'method': method})
            query = sample_task(spec, 'query', 256)
            row = {'task_id': spec.task_id, 'split': split}
            with torch.no_grad():
                for method in vectors:
                    classifier = load_classifier(output / method / split / f'{spec.index:03d}.pt')
                    row[method + '_accuracy'] = (classifier(query.x).argmax(1) == query.y).float().mean().item()
            rows.append(row)
    with (output / 'tasks.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {'phase': 'deterministic_hypernetwork', 'updates': updates,
               'status': 'complete' if updates == 1000 else 'capped',
               'training_seconds': training_seconds, 'last_minibatch_weight_mse': last_loss,
               'test_tasks_accessed': False, 'diffusion_trained': False, 'splits': {}}
    for split in ('train', 'validation'):
        subset = [r for r in rows if r['split'] == split]
        summary['splits'][split] = {method: sum(r[method + '_accuracy'] for r in subset) / len(subset)
                                    for method in vectors}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.collection, args.output), indent=2))


if __name__ == '__main__':
    main()
