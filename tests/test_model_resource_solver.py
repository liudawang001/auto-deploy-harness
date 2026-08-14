"""Phase A6 tests: deterministic model resource solver."""
import pytest

from auto_harness.config import HarnessConfig
from auto_harness.model_runtime.resource_solver import ModelResourceSolver
from auto_harness.model_runtime.schemas import ModelFilePlan, ResolvedModelSpec

GB = 1024 ** 3
SHA = "a" * 40


def _spec(**overrides):
    defaults = dict(
        status="resolved",
        source="huggingface",
        repo_id="org/model",
        resolved_revision=SHA,
        model_identity="huggingface:org/model@%s" % SHA,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        dtype="float16",
        parameter_count=7000000000,
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=32768,
    )
    defaults.update(overrides)
    return ResolvedModelSpec(**defaults)


def _plan(weight_bytes):
    plan = ModelFilePlan(
        model_identity="huggingface:org/model@%s" % SHA,
        files=[
            {"path": "model.safetensors", "role": "weight_shard", "size_bytes": weight_bytes, "sha256": "x", "required": True},
            {"path": "config.json", "role": "config", "size_bytes": 100, "sha256": "y", "required": True},
        ],
        total_size_bytes=weight_bytes + 100,
    )
    plan.plan_hash = plan.compute_plan_hash()
    return plan


def _config(**overrides):
    kwargs = {"model_inference_enabled": True}
    kwargs.update(overrides)
    return HarnessConfig(**kwargs)


def _host(**overrides):
    defaults = dict(
        gpu_indexes=[0],
        gpu_memory_total_bytes=24 * GB,
        gpu_memory_free_bytes=23 * GB,
        ram_total_bytes=128 * GB,
        ram_available_bytes=96 * GB,
        disk_total_bytes=512 * GB,
        disk_free_bytes=200 * GB,
    )
    defaults.update(overrides)
    return defaults


class TestResourceSolver:
    def test_7b_fp16_idle_24gb_allowed(self):
        decision = ModelResourceSolver().solve(
            _spec(), _plan(14 * GB), _config(), _host()
        )
        assert decision.status == "allowed"
        assert decision.selected_dtype == "float16"
        assert decision.weight_bytes == 14 * GB

    def test_14b_fp16_24gb_blocked(self):
        decision = ModelResourceSolver().solve(
            _spec(parameter_count=14000000000, hidden_size=5120, num_hidden_layers=40,
                  num_attention_heads=40, num_key_value_heads=8),
            _plan(28 * GB),
            _config(),
            _host(),
        )
        assert decision.status == "insufficient_gpu_memory"

    def test_context_length_increases_kv_cache(self):
        solver = ModelResourceSolver()
        short = solver.solve(_spec(), _plan(14 * GB), _config(model_runtime_max_model_len=2048), _host())
        long = solver.solve(_spec(), _plan(14 * GB), _config(model_runtime_max_model_len=4096), _host())
        assert long.kv_cache_bytes > short.kv_cache_bytes

    def test_gqa_reduces_kv_cache(self):
        solver = ModelResourceSolver()
        gqa = solver.solve(
            _spec(num_key_value_heads=4), _plan(14 * GB), _config(), _host()
        )
        mha = solver.solve(
            _spec(num_key_value_heads=28), _plan(14 * GB), _config(), _host()
        )
        assert gqa.kv_cache_bytes < mha.kv_cache_bytes

    def test_missing_fields_uncertain(self):
        decision = ModelResourceSolver().solve(
            _spec(hidden_size=None, num_hidden_layers=None, num_attention_heads=None),
            _plan(14 * GB),
            _config(),
            _host(),
        )
        assert decision.status == "uncertain"
        assert any("KV cache" in w for w in decision.warnings)

    def test_gpu_busy_vs_too_big(self):
        # model too big: 28GB > 24GB total -> insufficient_gpu_memory
        too_big = ModelResourceSolver().solve(
            _spec(parameter_count=14000000000, hidden_size=5120, num_hidden_layers=40,
                  num_attention_heads=40, num_key_value_heads=8),
            _plan(28 * GB), _config(), _host(),
        )
        assert too_big.status == "insufficient_gpu_memory"
        # model fits total but free VRAM too low -> gpu_busy
        busy = ModelResourceSolver().solve(
            _spec(), _plan(14 * GB), _config(),
            _host(gpu_memory_total_bytes=24 * GB, gpu_memory_free_bytes=8 * GB),
        )
        assert busy.status == "gpu_busy"

    def test_insufficient_ram(self):
        decision = ModelResourceSolver().solve(
            _spec(), _plan(14 * GB), _config(),
            _host(ram_available_bytes=4 * GB),
        )
        assert decision.status == "insufficient_system_memory"

    def test_insufficient_disk(self):
        decision = ModelResourceSolver().solve(
            _spec(), _plan(14 * GB), _config(),
            _host(disk_free_bytes=1 * GB),
        )
        assert decision.status == "insufficient_disk"

    def test_negative_weight_uncertain(self):
        decision = ModelResourceSolver().solve(
            _spec(), _plan(-1), _config(), _host()
        )
        assert decision.status == "uncertain"

    def test_huge_weight_does_not_overflow(self):
        decision = ModelResourceSolver().solve(
            _spec(), _plan(2 ** 70), _config(), _host()
        )
        assert decision.status == "insufficient_gpu_memory"

    def test_decision_hash_recomputes(self):
        solver = ModelResourceSolver()
        decision = solver.solve(_spec(), _plan(14 * GB), _config(), _host())
        assert decision.compute_decision_hash() == decision.decision_hash

    def test_not_resolved_unsupported_model(self):
        decision = ModelResourceSolver().solve(
            _spec(status="remote_code_required", requires_remote_code=True),
            _plan(14 * GB), _config(), _host(),
        )
        assert decision.status == "unsupported_model"
