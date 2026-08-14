"""Phase A0 tests: guarded model inference configuration and CLI wiring.

Verifies:
1. Default configuration is disabled and carries the documented defaults.
2. configs/default.json and resources/default.json stay in sync.
3. Boundary validation for model_runtime, tensor_parallel_size,
   gpu_memory_utilization, positive integers, and safety ratios.
4. CLI rejects illegal model-inference arguments with exit code 2 before
   any network/download/Docker action.
5. CLI applies valid model-inference overrides to the config.
"""
import json
import pytest
from pathlib import Path

from auto_harness.config import HarnessConfig
from auto_harness.cli import main


def _load_default_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestModelInferenceDefaults:
    def test_disabled_by_default(self):
        config = HarnessConfig()
        assert config.model_inference_enabled is False
        assert config.model_runtime == "vllm"
        assert config.model_runtime_mode == "managed_vllm"
        assert config.model_runtime_image == ""
        assert config.model_runtime_require_image_digest is True
        assert config.model_runtime_port == 8000
        assert config.model_runtime_dtype == "auto"
        assert config.model_runtime_max_model_len == 4096
        assert config.model_runtime_max_num_seqs == 1
        assert config.model_runtime_gpu_memory_utilization == 0.9
        assert config.model_runtime_tensor_parallel_size == 1
        assert config.model_runtime_startup_timeout_seconds == 900
        assert config.model_runtime_request_timeout_seconds == 120
        assert config.model_runtime_shm_size == "8g"
        assert config.model_runtime_allow_remote_code is False
        assert config.model_runtime_allow_quantized is False
        assert config.model_runtime_require_immutable_revision is True
        assert config.model_runtime_require_strong_weight_integrity is True
        assert config.model_runtime_disk_safety_ratio == 1.2
        assert config.model_runtime_ram_safety_ratio == 1.2
        assert config.model_id_override == ""
        assert config.model_revision_override == ""

    def test_default_json_files_in_sync(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_default = _load_default_json(repo_root / "configs" / "default.json")
        resources_default = _load_default_json(
            repo_root / "src" / "auto_harness" / "resources" / "default.json"
        )
        model_keys = {
            "model_inference_enabled",
            "model_runtime",
            "model_runtime_mode",
            "model_runtime_image",
            "model_runtime_require_image_digest",
            "model_runtime_port",
            "model_runtime_dtype",
            "model_runtime_max_model_len",
            "model_runtime_max_num_seqs",
            "model_runtime_gpu_memory_utilization",
            "model_runtime_tensor_parallel_size",
            "model_runtime_startup_timeout_seconds",
            "model_runtime_request_timeout_seconds",
            "model_runtime_shm_size",
            "model_runtime_allow_remote_code",
            "model_runtime_allow_quantized",
            "model_runtime_require_immutable_revision",
            "model_runtime_require_strong_weight_integrity",
            "model_runtime_disk_safety_ratio",
            "model_runtime_ram_safety_ratio",
        }
        for key in model_keys:
            assert key in config_default, "configs/default.json missing %s" % key
            assert key in resources_default, "resources/default.json missing %s" % key
            assert config_default[key] == resources_default[key], (
                "config drift for %s: %r != %r"
                % (key, config_default[key], resources_default[key])
            )
        assert config_default["model_inference_enabled"] is False
        assert resources_default["model_inference_enabled"] is False


class TestModelInferenceValidation:
    def test_invalid_model_runtime_rejected(self):
        with pytest.raises(ValueError, match="model_runtime"):
            HarnessConfig(model_runtime="tgi")

    def test_invalid_runtime_mode_rejected(self):
        with pytest.raises(ValueError, match="model_runtime_mode"):
            HarnessConfig(model_runtime_mode="local")

    def test_tensor_parallel_size_must_be_one(self):
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            HarnessConfig(model_runtime_tensor_parallel_size=2)

    def test_gpu_memory_utilization_below_range(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            HarnessConfig(model_runtime_gpu_memory_utilization=0.4)

    def test_gpu_memory_utilization_above_range(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            HarnessConfig(model_runtime_gpu_memory_utilization=0.96)

    def test_gpu_memory_utilization_bool_rejected(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            HarnessConfig(model_runtime_gpu_memory_utilization=True)

    @pytest.mark.parametrize(
        "field",
        [
            "model_runtime_port",
            "model_runtime_max_model_len",
            "model_runtime_max_num_seqs",
            "model_runtime_startup_timeout_seconds",
            "model_runtime_request_timeout_seconds",
        ],
    )
    def test_positive_integer_fields(self, field):
        with pytest.raises(ValueError, match=field):
            HarnessConfig(**{field: 0})
        with pytest.raises(ValueError, match=field):
            HarnessConfig(**{field: -5})
        with pytest.raises(ValueError, match=field):
            HarnessConfig(**{field: True})

    @pytest.mark.parametrize(
        "field",
        ["model_runtime_disk_safety_ratio", "model_runtime_ram_safety_ratio"],
    )
    def test_positive_ratio_fields(self, field):
        with pytest.raises(ValueError, match=field):
            HarnessConfig(**{field: 0})
        with pytest.raises(ValueError, match=field):
            HarnessConfig(**{field: -1.0})
        with pytest.raises(ValueError, match=field):
            HarnessConfig(**{field: True})

    def test_enabled_with_valid_values(self):
        config = HarnessConfig(model_inference_enabled=True)
        assert config.model_inference_enabled is True


class TestModelInferenceCLI:
    def test_illegal_gpu_utilization_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "deploy",
                    "--repo", "https://example.com/repo",
                    "--model-inference",
                    "--model-gpu-memory-utilization", "0.4",
                ]
            )
        assert exc.value.code == 2

    def test_illegal_model_runtime_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "deploy",
                    "--repo", "https://example.com/repo",
                    "--model-inference",
                    "--model-runtime", "tgi",
                ]
            )
        assert exc.value.code == 2

    def test_illegal_max_model_len_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "deploy",
                    "--repo", "https://example.com/repo",
                    "--model-inference",
                    "--model-max-model-len", "0",
                ]
            )
        assert exc.value.code == 2

    def test_valid_overrides_applied(self, tmp_path):
        from unittest.mock import MagicMock, patch
        config = HarnessConfig(
            runs_dir=str(tmp_path / "runs"),
            default_controller="langgraph",
            agent_provider="mock",
            agent_plan_first_provider="mock",
        )
        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.deploy.return_value = "task_123"
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                exit_code = main(
                    [
                        "deploy",
                        "--repo", "https://example.com/repo",
                        "--model-inference",
                        "--model-runtime", "vllm",
                        "--model-max-model-len", "2048",
                        "--model-gpu-memory-utilization", "0.8",
                        "--model-id-override", "huggingface:org/model",
                    ]
                )
            assert exit_code == 0
            assert config.model_inference_enabled is True
            assert config.model_runtime == "vllm"
            assert config.model_runtime_max_model_len == 2048
            assert config.model_runtime_gpu_memory_utilization == 0.8
            assert config.model_id_override == "huggingface:org/model"

    def test_default_off_does_not_enable(self, tmp_path):
        from unittest.mock import MagicMock, patch
        config = HarnessConfig(
            runs_dir=str(tmp_path / "runs"),
            default_controller="langgraph",
            agent_provider="mock",
            agent_plan_first_provider="mock",
        )
        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.deploy.return_value = "task_123"
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                exit_code = main(
                    ["deploy", "--repo", "https://example.com/repo"]
                )
            assert exit_code == 0
            assert config.model_inference_enabled is False
