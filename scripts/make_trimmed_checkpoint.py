"""Build a layer-trimmed GLM-5.3-Flash checkpoint (width unchanged) from the real weights.

Keeps decoder layers [0, num_layers), embeddings, final norm, lm_head and the vision tower;
drops the remaining decoder layers and the MTP layer. Tokenizer/processor files are symlinked.

    python scripts/make_trimmed_checkpoint.py --src <snapshot> --dst checkpoints/GLM-5.3-Flash-4L --num-layers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
PER_LAYER_LISTS = ("layer_types", "indexer_types", "mlp_layer_types")


def keep(name: str, num_layers: int) -> bool:
    m = LAYER_RE.match(name)
    return m is None or int(m.group(1)) < num_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--num-layers", type=int, default=4)
    args = ap.parse_args()
    src, dst, n = Path(args.src), Path(args.dst), args.num_layers
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    t = cfg["text_config"]
    assert n <= t["num_hidden_layers"]
    t["num_hidden_layers"] = n
    t["num_nextn_predict_layers"] = 0
    for k in PER_LAYER_LISTS:
        if isinstance(t.get(k), list):
            t[k] = t[k][:n]
    la = t.get("linear_attn_config") or {}
    for k in ("kda_layers", "full_attn_layers"):
        if k in la:
            la[k] = [i for i in la[k] if i < n]
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    by_file = defaultdict(list)
    for name, f in index.items():
        if keep(name, n):
            by_file[f].append(name)
    new_map, out_id = {}, 0
    for f in sorted(by_file):
        tensors = {}
        with safe_open(src / f, framework="pt") as h:
            for name in by_file[f]:
                tensors[name] = h.get_tensor(name)
        out_id += 1
        out_name = f"model-{out_id:05d}.safetensors"
        save_file(tensors, dst / out_name, metadata={"format": "pt"})
        new_map.update({k: out_name for k in tensors})
        print(f"{f} -> {out_name}: {len(tensors)} tensors", flush=True)
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": new_map}, indent=1))

    for p in src.iterdir():
        if p.name.endswith(".safetensors") or p.name in ("config.json", "model.safetensors.index.json"):
            continue
        link = dst / p.name
        if not link.exists():
            os.symlink(p.resolve(), link)
    (dst / "TRIMMED_FROM.json").write_text(json.dumps({"src": str(src), "num_layers": n}, indent=1))


if __name__ == "__main__":
    main()
