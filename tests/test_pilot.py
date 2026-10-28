import pytest
import torch
from p_diff_general.parameters import restore
from p_diff_general.pilot import canonicalize, fit
from p_diff_general.tasks import TaskSpec, sample_task
import time


def test_canonicalization_preserves_probabilities_and_removes_shared_shift():
    vector = torch.linspace(-2, 3, 18)
    canonical = canonicalize(vector)
    data = sample_task(TaskSpec('train', 2), 'source', 256)
    torch.testing.assert_close(restore(vector)(data.x).softmax(1),
                               restore(canonical)(data.x).softmax(1))
    torch.testing.assert_close(canonical[:16].reshape(2, 8).sum(0), torch.zeros(8), atol=3e-7, rtol=0)
    torch.testing.assert_close(canonicalize(canonical), canonical)


def test_fit_uses_fixed_stream_is_repeatable_and_obeys_deadline():
    torch.set_num_threads(1)
    data = sample_task(TaskSpec('train', 0), 'source', 256)
    state = torch.random.get_rng_state()
    first, loss, _ = fit(data, time.perf_counter() + 60)
    second, _, _ = fit(data, time.perf_counter() + 60)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(first, second)
    assert loss < .6932
    with pytest.raises(TimeoutError):
        fit(data, time.perf_counter() - 1)


def test_pilot_exports_all_tasks_and_training_only_statistics(tmp_path, monkeypatch):
    from p_diff_general import pilot
    from p_diff_general.parameters import load_classifier
    calls = []
    original = pilot.sample_task

    def guarded_sample(spec, role, count):
        assert spec.split in {'train', 'validation'}
        calls.append((spec.task_id, role))
        return original(spec, role, count)

    monkeypatch.setattr(pilot, 'sample_task', guarded_sample)
    output = tmp_path / 'pilot'
    summary = pilot.run(output)
    assert summary['completed_tasks'] == 40 and summary['status'] == 'complete'
    assert not summary['test_tasks_accessed'] and not summary['generator_trained']
    assert len(list((output / 'source').rglob('*.pt'))) == 40
    assert len(list((output / 'context').rglob('*.pt'))) == 40
    vectors = torch.stack([torch.load(p, weights_only=True)['vector']
                           for p in sorted((output / 'source/train').glob('*.pt'))])
    stats = torch.load(output / 'train_statistics.pt', weights_only=True)
    assert stats['count'] == 32 and stats['split'] == 'train'
    torch.testing.assert_close(stats['mean'], vectors.mean(0))
    torch.testing.assert_close(stats['std'], vectors.std(0, unbiased=False))
    data = original(TaskSpec('validation', 0), 'query', 256)
    model = load_classifier(output / 'source/validation/000.pt')
    assert model(data.x).shape == (256, 2)
    with pytest.raises(FileExistsError):
        pilot.run(output)


def test_pilot_cap_preserves_already_exported_source(tmp_path, monkeypatch):
    import json
    from p_diff_general import pilot
    calls = 0

    def capped_fit(data, deadline):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError('cap')
        return torch.zeros(18), .69, .01

    monkeypatch.setattr(pilot, 'fit', capped_fit)
    output = tmp_path / 'capped'
    summary = pilot.run(output)
    assert summary['status'] == 'capped' and summary['completed_tasks'] == 0
    assert (output / 'source/train/000.pt').exists()
    assert (output / 'manifest.json').exists()
    assert json.loads((output / 'summary.json').read_text())['status'] == 'capped'
    assert not (output / 'train_statistics.pt').exists()
