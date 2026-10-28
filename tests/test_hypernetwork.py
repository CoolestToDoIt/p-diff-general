import torch
import pytest
from p_diff_general.hypernetwork import Hypernetwork, load_hypernetwork
from p_diff_general.parameters import manifest
from p_diff_general.tasks import TaskSpec, Context, sample_task


def test_hypernetwork_checkpoint_roundtrip_and_context_order(tmp_path):
    context = sample_task(TaskSpec('validation', 0), 'context', 64)
    model = Hypernetwork(torch.zeros(18), torch.zeros(18)).eval()
    assert torch.isfinite(model(context)).all()
    assert model(context).shape == (18,)
    shuffled = Context(context.x.flip(0), context.y.flip(0))
    torch.testing.assert_close(model(context), model(shuffled), atol=1e-6, rtol=1e-5)
    file = tmp_path / 'hypernetwork.pt'
    payload = {'manifest': manifest(), 'state_dict': model.state_dict()}
    torch.save(payload, file)
    state = torch.random.get_rng_state()
    loaded = load_hypernetwork(file)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(loaded(context), model(context))
    assert not any(p.requires_grad for p in loaded.parameters())
    payload['manifest']['architecture'] = 'wrong'
    torch.save(payload, file)
    with pytest.raises(ValueError, match='manifest'):
        load_hypernetwork(file)
