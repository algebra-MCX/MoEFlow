"""Minimal Triton kernel smoke test on the local GPU (vector add, checks sm_100 codegen)."""

import json
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


def main():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(x)
    k = _add[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
    torch.cuda.synchronize()
    ok = torch.equal(o, x + y)
    info = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "triton": triton.__version__,
        "torch": torch.__version__,
        "bitwise_equal_to_torch": ok,
        "compiled_target": str(getattr(k, "metadata", {}).target) if hasattr(k, "metadata") else None,
    }
    print(json.dumps(info))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
