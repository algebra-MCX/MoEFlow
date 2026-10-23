import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def test_trim_keeps_first_layers_and_trims_config(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    names = ["model.language_model.embed_tokens.weight", "lm_head.weight", "model.visual.patch_embed.proj.weight"]
    names += [f"model.language_model.layers.{i}.mlp.gate.weight" for i in range(6)]
    save_file({n: torch.full((2,), float(i)) for i, n in enumerate(names)}, src / "model-00001-of-00001.safetensors")
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "model-00001-of-00001.safetensors" for n in names}})
    )
    (src / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 5,
                    "num_nextn_predict_layers": 1,
                    "layer_types": list("abcde"),
                    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 2,
                    "indexer_types": ["full"] * 5,
                    "linear_attn_config": {"kda_layers": [0, 1, 2, 4], "full_attn_layers": [3]},
                }
            }
        )
    )
    (src / "tokenizer.json").write_text("{}")
    r = subprocess.run(
        [
            sys.executable,
            "scripts/make_trimmed_checkpoint.py",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--num-layers",
            "4",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    t = json.loads((dst / "config.json").read_text())["text_config"]
    assert t["num_hidden_layers"] == 4 and t["num_nextn_predict_layers"] == 0
    assert t["layer_types"] == list("abcd") and t["mlp_layer_types"] == ["dense"] * 3 + ["sparse"]
    assert t["linear_attn_config"] == {"kda_layers": [0, 1, 2], "full_attn_layers": [3]}
    kept = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert sorted(k for k in kept if ".layers." in k) == [
        f"model.language_model.layers.{i}.mlp.gate.weight" for i in range(4)
    ]
    assert "lm_head.weight" in kept and "model.visual.patch_embed.proj.weight" in kept
    with safe_open(dst / kept["lm_head.weight"], framework="pt") as h:
        assert torch.equal(h.get_tensor("lm_head.weight"), torch.full((2,), 1.0))
    assert (dst / "tokenizer.json").is_symlink()
