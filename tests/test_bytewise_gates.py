import pytest
import torch
from bench.validation import check_backend_pair, require_matching_variants

from moeflow.integrations.vllm_diagnostics import compare_outputs
from moeflow.tolerances import bitwise_equal


def test_layer_byte_comparison_distinguishes_signed_zero():
    positive, negative = torch.tensor([[0.0]]), torch.tensor([[-0.0]])
    result = compare_outputs(positive, negative)
    assert result["finite"] and result["max_abs_diff"] == 0
    assert not result["bitwise_equal"]
    assert not check_backend_pair(positive, negative, positive)["bitwise_equal"]
    with pytest.raises(RuntimeError, match="outputs differ"):
        require_matching_variants(positive, negative)


def test_layer_byte_comparison_requires_matching_dtype():
    assert not compare_outputs(torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bfloat16))["bitwise_equal"]


def test_identical_nan_bits_never_pass_the_performance_gate():
    value = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="outputs differ"):
        require_matching_variants(value, value.clone())


@pytest.mark.parametrize("value", [torch.tensor(1.0), torch.empty(0, 2), torch.arange(6).reshape(2, 3).t()])
def test_byte_comparison_handles_scalars_empty_and_strided_tensors(value):
    assert bitwise_equal(value, value.contiguous().clone())
