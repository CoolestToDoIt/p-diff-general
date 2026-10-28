import json
import pytest
import torch
from p_diff_general.tasks import Context, TaskSpec, sample_task, task_specs
from p_diff_general.conditioning import ContextEncoder, centroid_parameters
from p_diff_general.parameters import Classifier, flatten, restore, save_classifier, load_classifier
from p_diff_general.smoke import run


def test_tasks_repeat_without_consuming_global_rng_and_have_disjoint_roles():
    spec = TaskSpec("train", 0)
    state = torch.random.get_rng_state()
    first = sample_task(spec, "context", 64)
    repeat = sample_task(spec, "context", 64)
    assert torch.equal(first.x, repeat.x) and torch.equal(first.y, repeat.y)
    assert torch.equal(state, torch.random.get_rng_state())
    source = sample_task(spec, "source", 64)
    query = sample_task(spec, "query", 64)
    assert not torch.equal(first.x, source.x) and not torch.equal(first.x, query.x)
    assert {tuple(row) for row in first.x.tolist()}.isdisjoint(tuple(row) for row in query.x.tolist())
    other = sample_task(TaskSpec("validation", 0), "context", 64)
    assert not torch.equal(first.x, other.x)
    ids = [s.task_id for split in ("train", "validation", "test") for s in task_specs(split, 8)]
    assert len(set(ids)) == len(ids)


def test_test_access_is_blocked_and_bad_context_rejected():
    with pytest.raises(ValueError, match="sealed"):
        sample_task(TaskSpec("test", 0), "query", 256)
    with pytest.raises(ValueError):
        sample_task(TaskSpec("train", 0), "unknown", 64)
    with pytest.raises(ValueError):
        Context(torch.zeros(4, 8), torch.zeros(4, dtype=torch.long))


def test_codec_preserves_logits_rejects_mismatch_and_is_standalone(tmp_path):
    model = Classifier().eval()
    x = torch.randn(9, 8)
    vector = flatten(model)
    state = torch.random.get_rng_state()
    fresh = restore(vector)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(model(x), fresh(x))
    file = tmp_path/"classifier.pt"
    save_classifier(file, vector, {"method": "test"})
    assert torch.equal(model(x), load_classifier(file)(x))
    with pytest.raises(FileExistsError):
        save_classifier(file, vector, {})
    with pytest.raises(ValueError):
        restore(vector.double())
    with pytest.raises(ValueError):
        restore(torch.full((18,), float("nan")))
    payload = torch.load(file, weights_only=True)
    payload["manifest"]["architecture"] = "other"
    torch.save(payload, tmp_path/"bad.pt")
    with pytest.raises(ValueError, match="manifest"):
        load_classifier(tmp_path/"bad.pt")


def test_conditioning_is_set_invariant_label_sensitive_and_context_only():
    torch.set_num_threads(1)
    context = sample_task(TaskSpec("train", 1), "context", 64)
    shuffled = Context(context.x.flip(0), context.y.flip(0))
    swapped = Context(context.x, 1-context.y)
    encoder = ContextEncoder().eval()
    torch.testing.assert_close(encoder(context), encoder(shuffled), rtol=1e-5, atol=1e-6)
    assert encoder(context).shape == (64,)
    assert not torch.allclose(encoder(context), encoder(swapped))
    torch.testing.assert_close(centroid_parameters(context), centroid_parameters(shuffled))
    vector = centroid_parameters(context)
    swapped_vector = centroid_parameters(swapped)
    torch.testing.assert_close(vector[:16].reshape(2, 8).flip(0), swapped_vector[:16].reshape(2, 8))
    # Query labels cannot enter either conditioning API; a caller has no query argument.
    before = centroid_parameters(context)
    query = sample_task(TaskSpec("train", 1), "query", 64)
    query.y.fill_(0)
    assert torch.equal(before, centroid_parameters(context))


def test_smoke_exports_models_without_test_access_or_training(tmp_path):
    summary = run(tmp_path/"smoke", tasks_per_split=2)
    assert not summary["test_tasks_accessed"] and summary["training_updates"] == 0
    assert len(list((tmp_path/"smoke/models").glob("*.pt"))) == 4
    assert summary["splits"]["validation"]["count"] == 2
    assert json.loads((tmp_path/"smoke/summary.json").read_text())["phase"] == "foundation_smoke"
    with pytest.raises(FileExistsError):
        run(tmp_path/"smoke", tasks_per_split=2)
