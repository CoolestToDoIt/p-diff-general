from dataclasses import replace
import pytest
import torch
from p_diff_general.conditioning import centroid_parameters
from p_diff_general.parameters import restore, manifest
from p_diff_general.pre_diffusion import (TrainingTask, BehaviorGenerator, behavior_loss,
    load_behavior, GaussianControl, prepare_bundle, load_training_bundle, checksum,
    train_behavior, paired_summary)
from p_diff_general.tasks import sample_task, task_specs


def training_tasks(count=4):
    return [TrainingTask(s.task_id, sample_task(s, 'context', 64), sample_task(s, 'source', 256),
                         centroid_parameters(sample_task(s, 'source', 256)))
            for s in task_specs('train', count)]


def test_behavior_loss_matches_classifier_and_backpropagates_to_head():
    tasks = training_tasks()
    targets = torch.stack([t.target for t in tasks])
    model = BehaviorGenerator(targets.mean(0), targets.std(0, unbiased=False), residual=True)
    vectors = model.vectors([t.context for t in tasks])
    for task, vector in zip(tasks, vectors):
        torch.testing.assert_close(vector, centroid_parameters(task.context))
    x, y = torch.stack([t.source.x for t in tasks]), torch.stack([t.source.y for t in tasks])
    loss = behavior_loss(vectors, x, y)
    expected = torch.stack([torch.nn.functional.cross_entropy(restore(v)(t.source.x), t.source.y)
                            for t, v in zip(tasks, vectors)]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.head.weight.grad.norm() > 0
    assert model.head.bias.grad.norm() > 0


def test_behavior_checkpoint_standalone_and_validation_training_rejected(tmp_path):
    tasks = training_tasks()
    model = BehaviorGenerator(torch.zeros(18), torch.ones(18), residual=True)
    file = tmp_path / 'behavior.pt'
    torch.save({'manifest': manifest(), 'residual': True, 'state_dict': model.state_dict()}, file)
    state = torch.random.get_rng_state()
    loaded = load_behavior(file)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(loaded(tasks[0].context), model(tasks[0].context))
    bad = replace(tasks[0], task_id=tasks[0].task_id.replace('/train/', '/validation/'))
    with pytest.raises(ValueError, match='training'):
        train_behavior([bad] * 256, False, file)
    with pytest.raises(ValueError, match='training'):
        GaussianControl([bad], True)


def test_gaussian_controls_use_local_noise_and_context_anchor():
    tasks = training_tasks()
    control = GaussianControl(tasks, True)
    context = tasks[0].context
    state = torch.random.get_rng_state()
    first = control.sample(context, 7)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(first, control.sample(context, 7))
    assert not torch.equal(first, control.sample(context, 8))
    difference = control.sample(tasks[1].context, 7) - first
    torch.testing.assert_close(difference, centroid_parameters(tasks[1].context) - centroid_parameters(context), atol=1e-6, rtol=1e-5)
    unconditional = GaussianControl(tasks, False)
    assert torch.equal(unconditional.sample(context, 7), unconditional.sample(tasks[1].context, 7))


def test_bundle_rejects_validation_context_corruption_and_changed_checksum(tmp_path):
    tasks = training_tasks(256)
    prepare_bundle(tasks, tmp_path, {'source_manifest_sha256': 'fixture'})
    file = tmp_path / 'diffusion_training_bundle.pt'
    digest = checksum(file)
    bundle = load_training_bundle(file, expected_sha256=digest)
    assert bundle['targets'].shape == (256, 18)
    bundle['context_x'][0, 0, 0] += 1
    torch.save(bundle, file)
    with pytest.raises(ValueError, match='checksum'):
        load_training_bundle(file, expected_sha256=digest)
    with pytest.raises(ValueError, match='hash'):
        load_training_bundle(file)
    bundle['split'] = 'validation'
    torch.save(bundle, file)
    with pytest.raises(ValueError, match='identities'):
        load_training_bundle(file)


def test_paired_interval_aggregates_over_tasks():
    summary = paired_summary([.1] * 64)
    assert summary['mean_difference'] == pytest.approx(.1)
    assert summary['task_bootstrap_95_percent'] == pytest.approx([.1, .1])
    assert summary['fraction_within_five_points'] == 1


def test_behavior_cap_saves_a_loadable_partial_checkpoint(tmp_path, monkeypatch):
    from p_diff_general import pre_diffusion
    tasks = training_tasks(256)
    times = iter([0., 301., 302.])
    monkeypatch.setattr(pre_diffusion.time, 'perf_counter', lambda: next(times))
    model, info = train_behavior(tasks, True, tmp_path / 'capped.pt')
    assert info['status'] == 'capped' and info['updates'] == 0
    assert info['last_source_loss'] is None
    torch.testing.assert_close(model(tasks[0].context), centroid_parameters(tasks[0].context))


def test_preflight_rejects_prior_mismatch_and_test_access(tmp_path):
    import json
    from pathlib import Path
    from p_diff_general.pre_diffusion import preflight, write_json
    tasks = training_tasks(256)
    prepare_bundle(tasks, tmp_path, {'source_manifest_sha256': 'fixture'})
    write_json(tmp_path / 'readiness.json', {
        'training_bundle_sha256': checksum(tmp_path / 'diffusion_training_bundle.pt'),
        'baseline_runs_complete': True,
        'baseline_gates': {'behavior_residual': {
            'within_two_points_of_context_fit': True,
            'at_least_80_percent_tasks_within_five_points': True,
            'conditioning_improvement_interval_above_zero': True}}})
    settings = json.loads((Path(__file__).parents[1] / 'configs/diffusion_proposal.json').read_text())
    proposal = tmp_path / 'proposal.json'
    write_json(proposal, settings)
    result = preflight(tmp_path, proposal)
    assert result['data_and_baselines_ready'] and not result['diffusion_trained']
    settings['diffusion']['steps'] = 100
    write_json(proposal, settings)
    with pytest.raises(ValueError, match='prior'):
        preflight(tmp_path, proposal)
    settings['test_evaluation_authorized'] = True
    write_json(proposal, settings)
    with pytest.raises(ValueError, match='contract'):
        preflight(tmp_path, proposal)
