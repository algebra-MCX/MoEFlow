import pytest
import torch

from moeflow.config import MoEConfig

# Narrow but structurally faithful config for fast CPU tests: same routing semantics,
# expert count divisible by ep in {1, 2, 4, 8}, clamp limit low enough to be exercised.
SMALL = MoEConfig(
    hidden_size=64, moe_intermediate_size=32, n_routed_experts=16, top_k=4, routed_scaling_factor=2.5, swiglu_limit=1.0
)


@pytest.fixture
def small_cfg():
    return SMALL


def pytest_collection_modifyitems(config, items):
    has_gpu = torch.cuda.is_available()
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
