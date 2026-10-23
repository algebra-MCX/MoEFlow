"""Real expert weights from the GLM-5.3-Flash FP8 checkpoint, dequantized to BF16.

The checkpoint stores gate/up/down projections per expert as float8_e4m3fn with
`weight_scale_inv` block scales over 128x128 tiles (quantization_config.weight_block_size):
    w_bf16[i, j] = w_fp8[i, j] * scale_inv[i // 128, j // 128]
Layouts returned match the rest of the project: w13 [L, 2I, H] = [gate; up], w2 [L, H, I].
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_SNAPSHOT = Path(os.environ.get("MOEFLOW_MODEL_PATH", "checkpoints/GLM-5.3-Flash"))
PREFIX = "model.language_model.layers.{layer}.mlp"


def dequant_block_fp8(w: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int] = (128, 128)) -> torch.Tensor:
    """Block-scaled FP8 -> fp32. Partial edge blocks use the scale of the block they fall in."""
    rows, cols = w.shape
    s = scale_inv.float().repeat_interleave(block[0], 0)[:rows].repeat_interleave(block[1], 1)[:, :cols]
    return w.float() * s


class CheckpointReader:
    def __init__(self, snapshot: str | Path = DEFAULT_SNAPSHOT):
        self.root = Path(snapshot)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        cfg = json.loads((self.root / "config.json").read_text())
        q = cfg.get("quantization_config") or {}
        self.block = tuple(q.get("weight_block_size", (128, 128)))

    def get(self, names: list[str]) -> dict[str, torch.Tensor]:
        names_by_file = defaultdict(list)
        for name in names:
            names_by_file[self.index[name]].append(name)
        tensors = {}
        for filename, tensor_names in names_by_file.items():
            with safe_open(self.root / filename, framework="pt") as checkpoint:
                for name in tensor_names:
                    tensors[name] = checkpoint.get_tensor(name)
        return tensors

    def _proj(self, base: str, raw: dict) -> torch.Tensor:
        weight = raw[f"{base}.weight"]
        if weight.dtype == torch.float8_e4m3fn:
            return dequant_block_fp8(weight, raw[f"{base}.weight_scale_inv"], self.block)
        return weight.float()

    def expert_weights(self, layer: int, experts: range, dtype=torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
        p = PREFIX.format(layer=layer)
        names = []
        for e in experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"{p}.experts.{e}.{proj}"
                names.append(f"{base}.weight")
                if f"{base}.weight_scale_inv" in self.index:
                    names.append(f"{base}.weight_scale_inv")
        raw = self.get(names)
        w13, w2 = [], []
        for e in experts:
            base = f"{p}.experts.{e}"
            gate, up = self._proj(f"{base}.gate_proj", raw), self._proj(f"{base}.up_proj", raw)
            w13.append(torch.cat([gate, up], 0).to(dtype))
            w2.append(self._proj(f"{base}.down_proj", raw).to(dtype))
        return torch.stack(w13), torch.stack(w2)

    def router(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate weight [E, H] (fp32) and e_score_correction_bias [E] (fp32)."""
        p = PREFIX.format(layer=layer)
        raw = self.get([f"{p}.gate.weight", f"{p}.gate.e_score_correction_bias"])
        return raw[f"{p}.gate.weight"].float(), raw[f"{p}.gate.e_score_correction_bias"].float()
