"""vLLM plugin unit tests (no model load)."""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
pytestmark = pytest.mark.vllm


def test_register_is_noop_without_env(monkeypatch):
    monkeypatch.delenv("MOEFLOW_RUNNER", raising=False)
    monkeypatch.delenv("MOEFLOW_ALL2ALL", raising=False)
    layer = importlib.import_module("vllm.model_executor.layers.fused_moe.layer")
    before = layer.FusedMoEFactory
    from moeflow.integrations import vllm_plugin

    vllm_plugin.register()
    assert layer.FusedMoEFactory is before


def test_fixed_buffer_pool_separates_configuration_and_dispatch_mode(monkeypatch):
    from dataclasses import replace

    import moeflow.fixed
    from moeflow.config import MoEConfig
    from moeflow.integrations import vllm_runner

    monkeypatch.setattr(vllm_runner, "_BUFFER_POOL", {})
    monkeypatch.setattr(moeflow.fixed, "FixedEPLayer", lambda *args, **kw: SimpleNamespace(**kw))
    cfg, group = MoEConfig(), object()
    get = vllm_runner._shared_fixed_layer
    plain = get(cfg, 0, 4, 8, "cuda:0", group)
    assert get(cfg, 0, 4, 8, "cuda:0", group) is plain
    fused = get(cfg, 0, 4, 8, "cuda:0", group, fused_dispatch=True)
    assert fused is not plain and fused.fused_dispatch
    assert get(replace(cfg, top_k=4), 0, 4, 8, "cuda:0", group) is not plain


def test_factory_wrapper_injects_runner_cls_only_when_unset():
    from moeflow.integrations.vllm_plugin import _wrap_factory
    from moeflow.integrations.vllm_runner import MoEFlowRunner

    seen = []
    wrapped = _wrap_factory(lambda *a, **kw: seen.append(kw.get("runner_cls")))
    wrapped(num_experts=4)
    sentinel = object()
    wrapped(num_experts=4, runner_cls=sentinel)
    assert seen == [MoEFlowRunner, sentinel]


def test_register_runner_patches_all_bindings(monkeypatch):
    from moeflow.integrations import vllm_plugin

    layer = importlib.import_module("vllm.model_executor.layers.fused_moe.layer")
    pkg = importlib.import_module("vllm.model_executor.layers.fused_moe")
    monkeypatch.setattr(layer, "FusedMoEFactory", layer.FusedMoEFactory)
    monkeypatch.setattr(pkg, "FusedMoEFactory", pkg.FusedMoEFactory)
    vllm_plugin.register_runner()
    assert getattr(layer.FusedMoEFactory, "_moeflow_wrapped", False)
    assert pkg.FusedMoEFactory is layer.FusedMoEFactory


def test_compare_routing_is_slot_order_invariant():
    from moeflow.integrations.vllm_runner import _compare_routing

    ids = torch.tensor([[3, 1], [0, 2]])
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    v_ids = torch.tensor([[1, 3], [0, 5]])
    v_w = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    r = _compare_routing(ids, w, v_ids, v_w)
    assert r["token_expert_set_match"] == 0.5 and r["max_abs_weight_diff_on_match"] == 0.0


@pytest.mark.parametrize(
    "local_tokens,global_counts,expected",
    [(1, [1, 64], 64), (1, [1, 512], None), (0, [0, 0], 1), (5, None, 8)],
)
def test_fixed_capacity_uses_global_decode_count(monkeypatch, local_tokens, global_counts, expected):
    import vllm.forward_context

    from moeflow.integrations.vllm_runner import MoEFlowRunner

    metadata = None if global_counts is None else SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(global_counts))
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: SimpleNamespace(dp_metadata=metadata))
    runner = SimpleNamespace(moeflow_max_tokens=128)
    assert MoEFlowRunner._fixed_capacity(runner, local_tokens) == expected


@pytest.mark.parametrize("monolithic", [False, True])
def test_fixed_prefill_delegates_to_native_runner(monkeypatch, monolithic):
    from moeflow.integrations import vllm_runner

    runner = object.__new__(vllm_runner.MoEFlowRunner)
    runner._moeflow_calls = 0
    runner.moeflow_mode = "fixed"
    runner.routed_experts = SimpleNamespace(quant_method=SimpleNamespace(is_monolithic=monolithic))
    runner._fixed_capacity = lambda _: None
    records, calls = [], []
    runner._record = records.append
    expected = object()

    def native(self, *args):
        assert self is runner
        calls.append(args)
        return expected

    monkeypatch.setattr(vllm_runner.MoERunner, "_apply_quant_method", native)
    hidden, logits, shared, input_ids = torch.zeros(512, 2), object(), object(), object()
    result = runner._apply_quant_method(hidden, logits, shared, input_ids, True)
    assert result is expected and calls == [(hidden, logits, shared, input_ids, True)]
    assert runner._moeflow_calls == 1 and not records[0]["intercepted"]


def test_fixed_logging_is_not_exhausted_by_prefill_calls(tmp_path):
    from moeflow.integrations.vllm_runner import MoEFlowRunner

    path = tmp_path / "runner.jsonl"
    runner = SimpleNamespace(_moeflow_calls=100, _moeflow_fixed_calls=1, _moeflow_log=path, layer_name="moe")
    MoEFlowRunner._record(runner, {"mode": "fixed", "intercepted": True})
    assert json.loads(path.read_text())["mode"] == "fixed"
    runner._moeflow_fixed_calls = 9
    MoEFlowRunner._record(runner, {"mode": "fixed", "intercepted": True})
    assert len(path.read_text().splitlines()) == 1


def test_routing_trace_excludes_warmup_and_bounds_request_samples(tmp_path, monkeypatch):
    import vllm.distributed
    import vllm.forward_context

    from moeflow.integrations.vllm_runner import MoEFlowRunner

    monkeypatch.setattr(vllm.distributed, "get_ep_group", lambda: SimpleNamespace(rank_in_group=1, world_size=4))
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: SimpleNamespace(is_padding=None))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    runner = SimpleNamespace(
        _routing_trace_dir=tmp_path, _routing_trace_calls=0, _moeflow_calls=100, moeflow_mode="fixed", layer_name="moe"
    )
    ids, weights = torch.zeros(200, 8, dtype=torch.int32), torch.full((200, 8), 0.3125)
    MoEFlowRunner._record_routing_trace(runner, ids, weights)
    assert not list(tmp_path.glob("*.jsonl"))
    (tmp_path / "enabled").touch()
    MoEFlowRunner._record_routing_trace(runner, ids, weights)
    path = next(tmp_path.glob("*.jsonl"))
    row = json.loads(path.read_text())
    assert row["tokens"] == 200 and row["tokens_recorded"] == len(row["expert_ids"]) == 128
    assert row["ep_rank"] == 1 and row["weights"][0] == [0.3125] * 8
    runner._routing_trace_calls = 32
    MoEFlowRunner._record_routing_trace(runner, ids, weights)
    assert len(path.read_text().splitlines()) == 1


def test_routing_trace_filters_idle_padding(tmp_path, monkeypatch):
    import vllm.distributed
    import vllm.forward_context

    from moeflow.integrations.vllm_runner import MoEFlowRunner

    monkeypatch.setattr(vllm.distributed, "get_ep_group", lambda: SimpleNamespace(rank_in_group=1, world_size=4))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        vllm.forward_context, "get_forward_context", lambda: SimpleNamespace(is_padding=torch.tensor([True, False]))
    )
    runner = SimpleNamespace(
        _routing_trace_dir=tmp_path, _routing_trace_calls=0, _moeflow_calls=100, moeflow_mode="fixed", layer_name="moe"
    )
    (tmp_path / "enabled").touch()
    MoEFlowRunner._record_routing_trace(runner, torch.tensor([[7], [9]]), torch.ones(2, 1))
    row = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert row["padding_known"] and row["execution_rows"] == 2 and row["tokens"] == 1
    assert row["expert_ids"] == [[9]]


@pytest.fixture
def nccl_world1(tmp_path):
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path}/store", rank=0, world_size=1, device_id=torch.device("cuda", 0)
    )
    yield dist.group.WORLD
    dist.destroy_process_group()


@pytest.mark.gpu
def test_prepare_finalize_roundtrip_world1(nccl_world1):
    """prepare -> scale experts -> finalize reproduces sum_k w_k (e_k + 1) x on one rank."""
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceDelegate

    from moeflow.integrations.vllm_pf import MoEFlowPrepareAndFinalize

    pf = MoEFlowPrepareAndFinalize(nccl_world1, num_local_experts=4)
    x = torch.tensor([[1.0, 2.0], [3.0, -1.0]], device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 3], [2, 1]], device="cuda", dtype=torch.int32)
    w = torch.tensor([[0.5, 1.0], [2.0, 0.25]], device="cuda")
    qc = FusedMoEQuantConfig.make(None)
    a, scale, meta, g_ids, g_w = pf.prepare(x, w, ids, 4, None, False, qc, False)
    assert scale is None and torch.equal(a, x) and torch.equal(g_ids, ids) and torch.equal(g_w, w)
    assert meta.expert_num_tokens.tolist() == [1, 1, 1, 1]
    fused = torch.stack([a * (g_ids[:, k : k + 1] + 1).to(a.dtype) for k in range(2)], dim=1)  # [M, K, H]
    out = torch.empty_like(x)
    pf.finalize(out, fused, g_w, g_ids, False, TopKWeightAndReduceDelegate())
    expect = torch.stack([(0.5 * 1 + 1.0 * 4) * x[0].float(), (2.0 * 3 + 0.25 * 2) * x[1].float()]).to(x.dtype)
    assert torch.equal(out, expect)
