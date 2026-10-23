import json

import pytest
import torch
from safetensors.torch import save_file

from moeflow.weights import DEFAULT_SNAPSHOT, CheckpointReader, dequant_block_fp8


def test_dequant_block_fp8_matches_elementwise_definition():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(200, 130, generator=g) * 4).to(torch.float8_e4m3fn)
    s = torch.rand(2, 2, generator=g) + 0.5
    out = dequant_block_fp8(w, s)
    for i, j in [(0, 0), (127, 127), (128, 0), (0, 128), (199, 129)]:
        assert out[i, j].item() == (w[i, j].float() * s[i // 128, j // 128]).item()  # fp32 product


def test_reader_layouts_on_synthetic_checkpoint(tmp_path):
    H, I = 256, 128
    p = "model.language_model.layers.3.mlp"
    t = {}
    for e in range(2):
        for proj, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            t[f"{p}.experts.{e}.{proj}.weight"] = torch.full(shape, float(e + 1)).to(torch.float8_e4m3fn)
            t[f"{p}.experts.{e}.{proj}.weight_scale_inv"] = torch.full((shape[0] // 128, shape[1] // 128), 0.5)
    t[f"{p}.gate.weight"] = torch.ones(2, H, dtype=torch.bfloat16)
    t[f"{p}.gate.e_score_correction_bias"] = torch.zeros(2)
    save_file(t, tmp_path / "m.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "m.safetensors" for k in t}}))
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"weight_block_size": [128, 128]}}))
    r = CheckpointReader(tmp_path)
    w13, w2 = r.expert_weights(3, range(1, 2))
    assert w13.shape == (1, 2 * I, H) and w2.shape == (1, H, I) and w13.dtype == torch.bfloat16
    assert torch.all(w13 == 1.0) and torch.all(w2 == 1.0)  # 2 * 0.5
    gw, b = r.router(3)
    assert gw.dtype == torch.float32 and gw.shape == (2, H) and b.shape == (2,)


@pytest.mark.slow
def test_real_checkpoint_layer3_expert0():
    if not DEFAULT_SNAPSHOT.exists():
        pytest.skip("checkpoint not present")
    r = CheckpointReader()
    w13, w2 = r.expert_weights(3, range(0, 1))
    assert w13.shape == (1, 4096, 4096) and w2.shape == (1, 4096, 2048)
    assert torch.isfinite(w13.float()).all() and 0 < w13.float().abs().mean() < 1
    gw, bias = r.router(3)
    assert gw.shape == (288, 4096) and bias.shape == (288,)
