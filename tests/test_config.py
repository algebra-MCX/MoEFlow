import json
import os

import pytest

from moeflow.config import GLM53_FLASH, MoEConfig

HF_CONFIG = os.path.join(os.environ.get("MOEFLOW_MODEL_PATH", "checkpoints/GLM-5.3-Flash"), "config.json")


def test_from_hf_config_reads_text_config(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 8,
                    "moe_intermediate_size": 4,
                    "n_routed_experts": 6,
                    "n_shared_experts": 1,
                    "num_experts_per_tok": 2,
                    "n_group": 1,
                    "topk_group": 1,
                    "norm_topk_prob": False,
                    "routed_scaling_factor": 1.5,
                    "scoring_func": "softmax",
                    "topk_method": "greedy",
                }
            }
        )
    )
    cfg = MoEConfig.from_hf_config(p)
    assert (cfg.hidden_size, cfg.n_routed_experts, cfg.top_k, cfg.swiglu_limit) == (8, 6, 2, None)


def test_real_config_matches_defaults():
    try:
        cfg = MoEConfig.from_hf_config(HF_CONFIG)
    except FileNotFoundError:
        pytest.skip("GLM-5.3-Flash snapshot not present")
    assert cfg == GLM53_FLASH


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
def test_experts_divisible(ep):
    assert GLM53_FLASH.experts_per_rank(ep) * ep == 288


def test_experts_not_divisible_raises():
    with pytest.raises(ValueError):
        GLM53_FLASH.experts_per_rank(5)


def test_calc_diff_matches_deepep_definition():
    import torch

    from moeflow.tolerances import calc_diff

    x = torch.tensor([1.0, 2.0, 3.0])
    assert calc_diff(x, x) == 0.0
    y = x + 0.5
    xd, yd = x.double() + 1, y.double() + 1
    assert calc_diff(x, y) == float(1 - 2 * (xd * yd).sum() / (xd * xd + yd * yd).sum())
