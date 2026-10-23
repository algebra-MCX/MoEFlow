import pytest
import torch

from moeflow.fixed import FixedEPLayer


@pytest.fixture
def layer(small_cfg):
    return FixedEPLayer(small_cfg, 0, 2, None, None, capacity=1, device=torch.device("cpu"))


@pytest.mark.parametrize("direct", [False, True])
def test_external_experts_copy_by_default_and_alias_only_when_requested(layer, direct):
    output = torch.arange(2 * layer.H, dtype=torch.bfloat16).reshape(2, layer.H)
    expected = output.clone()
    partials = layer.stage_experts(lambda rows: output, lambda rows: (rows,), direct_output=direct)
    if direct:
        assert partials.shape == layer.ret_send.shape and partials.data_ptr() == output.data_ptr()
        assert torch.count_nonzero(layer.ret_send) == 0
    else:
        assert partials is None and torch.equal(layer.ret_send.view_as(output), expected)
        output.zero_()
        assert torch.equal(layer.ret_send.view_as(output), expected)


@pytest.mark.parametrize("problem", ["shape", "dtype", "stride", "device"])
def test_direct_output_rejects_incompatible_buffers_before_communication(layer, problem):
    output = torch.zeros(2, layer.H, dtype=torch.bfloat16)
    if problem == "shape":
        output = output[:, :1]
    elif problem == "dtype":
        output = output.float()
    elif problem == "stride":
        output = torch.zeros(layer.H, 2, dtype=torch.bfloat16).t()
    else:
        output = torch.empty_like(output, device="meta")
    with pytest.raises(ValueError, match="Direct expert output must"):
        layer.stage_experts(lambda rows: output, lambda rows: (rows,), direct_output=True)


def test_direct_output_requires_an_explicit_preparation_contract(layer):
    with pytest.raises(ValueError, match="custom input preparation"):
        layer.stage_experts(lambda *args: None, direct_output=True)
