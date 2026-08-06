import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from auto_harness.config import HarnessConfig
from auto_harness.env.ownership import EnvironmentOwnership
from auto_harness.env.postcheck import EnvironmentPostchecker
from auto_harness.modules.env_deploy import EnvDeployModule
from auto_harness.modules.env_solve import EnvSolveModule
from auto_harness.preflight.compatibility import EnvironmentCompatibilityResolver
from auto_harness.preflight.gpu import NvidiaGpuProbe
from auto_harness.preflight.policy import EnvironmentPreflightPolicy
from auto_harness.preflight.service import HostPreflightService
from auto_harness.recovery.graph_adapter import GraphRecoveryAdapter


def completed(code=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


def test_gpu_probe_distinguishes_not_found_timeout_and_detected():
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    detected = lambda *args, **kwargs: completed(
        stdout="0, GPU-1, A100, 535.54, 81920, 70000\n"
    )
    assert NvidiaGpuProbe(command_runner=missing).probe()["status"] == "not_found"
    assert NvidiaGpuProbe(command_runner=timeout).probe()["status"] == "timeout"
    result = NvidiaGpuProbe(command_runner=detected).probe()
    assert result["status"] == "detected"
    assert result["devices"][0]["uuid"] == "GPU-1"
    assert result["devices"][0]["memory_free_mb"] == 70000


def test_compatibility_selects_owned_matching_conda_environment(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env_root = repo / ".conda" / "envs"
    prefix = env_root / "demo"
    config = HarnessConfig(
        env_backend="conda",
        conda_envs_dir=str(env_root),
    )
    conda_file = {
        "found": True,
        "name": "demo",
        "python": "3.10",
        "channels": ["conda-forge"],
        "conda_dependencies": ["python=3.10", "numpy=1.26"],
        "pip_dependencies": [],
    }
    resolver = EnvironmentCompatibilityResolver()
    first = resolver.resolve(
        repo,
        {},
        {"gpu_required": False},
        conda_file,
        {
            "gpu": {"status": "not_found", "devices": []},
            "environment_runtimes": {
                "conda": {"available": True, "path": "/opt/conda/bin/conda"},
            },
        },
        {"environments": []},
        config,
    )
    prefix.mkdir(parents=True)
    EnvironmentOwnership().write(
        prefix,
        first["project_id"],
        first["repo_fingerprint"],
        "operation-1",
        first["spec_hash"],
        "3.10",
    )
    inventory = {
        "environments": [{
            "prefix": str(prefix.resolve()),
            "owned_by_harness": True,
            "owner_project_id": first["project_id"],
            "spec_hash": first["spec_hash"],
        }],
    }
    second = resolver.resolve(
        repo,
        {},
        {"gpu_required": False},
        conda_file,
        {
            "gpu": {"status": "not_found", "devices": []},
            "environment_runtimes": {
                "conda": {"available": True, "path": "/opt/conda/bin/conda"},
            },
        },
        inventory,
        config,
    )
    assert second["status"] == "allowed"
    assert second["action"] == "reuse"
    assert second["target_prefix"] == str(prefix.resolve())


def test_compatibility_blocks_missing_conda_and_required_gpu(tmp_path):
    config = HarnessConfig(env_backend="conda", conda_allow_venv_fallback=False)
    decision = EnvironmentCompatibilityResolver().resolve(
        tmp_path,
        {},
        {"gpu_required": True},
        {"found": True, "name": "demo", "python": "3.10"},
        {
            "gpu": {"status": "not_found", "devices": []},
            "environment_runtimes": {},
        },
        {"environments": []},
        config,
    )
    assert decision["status"] == "blocked"
    assert decision["action"] == "block"


def test_compatibility_preserves_probe_uncertainty_and_rejects_channels(tmp_path):
    config = HarnessConfig(env_backend="venv", preflight_require_gpu=True)
    resolver = EnvironmentCompatibilityResolver()
    uncertain = resolver.resolve(
        tmp_path,
        {},
        {"gpu_required": True},
        {"found": False},
        {
            "gpu": {"status": "timeout", "devices": [], "errors": ["timeout"]},
            "environment_runtimes": {},
        },
        {"environments": []},
        config,
    )
    rejected = resolver.resolve(
        tmp_path,
        {},
        {"gpu_required": False},
        {
            "found": True,
            "python": "3.10",
            "rejected_channels": ["https://untrusted.invalid/channel"],
        },
        {
            "gpu": {"status": "not_found", "devices": []},
            "environment_runtimes": {},
        },
        {"environments": []},
        config,
    )
    assert uncertain["status"] == "uncertain"
    assert uncertain["action"] == "request_approval"
    assert rejected["status"] == "blocked"


def test_environment_policy_enforces_tool_prefix_channel_and_package(tmp_path):
    prefix = (tmp_path / ".conda" / "envs" / "demo").resolve()
    config = HarnessConfig(conda_envs_dir=str(tmp_path / ".conda" / "envs"))
    decision = {
        "tool": "/opt/conda/bin/conda",
        "target_prefix": str(prefix),
    }
    policy = EnvironmentPreflightPolicy()
    allowed = policy.validate_mutation_command(
        ["/opt/conda/bin/conda", "create", "-y", "-p", str(prefix), "-c", "conda-forge", "python=3.10"],
        decision,
        tmp_path,
        config,
    )
    unsafe = policy.validate_mutation_command(
        ["/opt/conda/bin/conda", "install", "-y", "-p", str(prefix), "git+https://example.invalid/pkg"],
        decision,
        tmp_path,
        config,
    )
    outside = policy.validate_mutation_command(
        ["/opt/conda/bin/conda", "create", "-y", "-p", "/tmp/base", "python=3.10"],
        decision,
        tmp_path,
        config,
    )
    pip_install = policy.validate_mutation_command(
        ["/opt/conda/bin/conda", "run", "-p", str(prefix), "python", "-m", "pip", "install", "numpy>=1.26"],
        decision,
        tmp_path,
        config,
    )
    arbitrary_run = policy.validate_mutation_command(
        ["/opt/conda/bin/conda", "run", "-p", str(prefix), "python", "-c", "print('unsafe')"],
        decision,
        tmp_path,
        config,
    )
    assert allowed["allowed"] is True
    assert unsafe["allowed"] is False
    assert outside["allowed"] is False
    assert pip_install["allowed"] is True
    assert arbitrary_run["allowed"] is False


def test_environment_policy_rejects_conda_source_bypasses_and_allows_safe_requirement_file(tmp_path):
    prefix = (tmp_path / ".conda" / "envs" / "demo").resolve()
    config = HarnessConfig(conda_envs_dir=str(prefix.parent))
    decision = {
        "tool": "/opt/conda/bin/conda",
        "target_prefix": str(prefix),
    }
    (tmp_path / "requirements.txt").write_text("numpy>=1.26\n", encoding="utf-8")
    policy = EnvironmentPreflightPolicy()
    channel_bypass = policy.validate_mutation_command(
        [decision["tool"], "install", "-y", "-p", str(prefix), "evil-channel::pkg"],
        decision, tmp_path, config,
    )
    file_bypass = policy.validate_mutation_command(
        [decision["tool"], "install", "-y", "-p", str(prefix), "--file", "/tmp/spec.txt"],
        decision, tmp_path, config,
    )
    requirement_file = policy.validate_mutation_command(
        [
            decision["tool"], "run", "-p", str(prefix),
            "python", "-m", "pip", "install", "-r", "requirements.txt",
        ],
        decision, tmp_path, config,
    )
    assert channel_bypass["allowed"] is False
    assert file_bypass["allowed"] is False
    assert requirement_file["allowed"] is True


def test_postcheck_requires_prefix_python_packages_and_gpu(tmp_path):
    prefix = (tmp_path / "env").resolve()
    prefix.mkdir()
    payload = {
        "executable": str(prefix / "bin" / "python"),
        "version": "3.10.14",
        "packages": {"numpy": "1.26.4"},
        "gpu_runtime": {"available": True, "device_count": 1},
    }
    runner = lambda *args, **kwargs: completed(stdout=json.dumps(payload))
    evidence = EnvironmentPostchecker(command_runner=runner).check(
        "/opt/conda/bin/conda",
        prefix,
        "3.10",
        ["numpy>=1.26"],
        True,
        "sha256:test",
    )
    assert evidence["status"] == "passed"
    payload["gpu_runtime"]["available"] = False
    failed = EnvironmentPostchecker(command_runner=runner).check(
        "/opt/conda/bin/conda",
        prefix,
        "3.10",
        ["numpy>=1.26"],
        True,
        "sha256:test",
    )
    assert failed["status"] == "failed"


def test_postcheck_fails_closed_for_invalid_constraints_and_supports_python_ranges():
    checker = EnvironmentPostchecker()
    assert checker._package_satisfied(
        "numpy>=99;python_version>='3.0'",
        {"numpy": "1.0"},
    ) is False
    assert checker._package_satisfied(
        "numpy>=99;python_version<'3.0'",
        {"numpy": "1.0"},
    ) is True
    assert checker._python_satisfied("3.11.9", ">=3.10,<3.12") is True


def test_env_solve_normalizes_gpu_preflight_and_honors_cpu_fallback(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("torch\n", encoding="utf-8")
    capabilities = {
        "host": {"platform": "linux", "machine": "x86_64"},
        "gpu": {
            "status": "detected",
            "driver_cuda_version": "12.4",
            "devices": [{"index": 0, "memory_free_mb": 24000}],
        },
        "environment_runtimes": {},
    }
    decision = {
        "status": "allowed",
        "backend": "venv",
        "action": "create",
        "python": "3.10",
        "torch_variant": "cu121",
        "selected_gpu_index": 0,
    }
    result = EnvSolveModule().solve(
        repo,
        {
            "frameworks": ["torch"],
            "install_plan": [["python3", "-m", "pip", "install", "-r", "requirements.txt"]],
        },
        {"gpu_required": True, "torch_variant": "cuda_or_cpu"},
        preflight={
            "capabilities": capabilities,
            "compatibility_decision": decision,
            "policy": {"allowed": True},
            "conda_file": {"found": False},
        },
    )
    solution = result.data["analysis"]["env_solution"]
    assert solution["torch_solution"]["selected"]["variant"] == "cu121"
    assert solution["local_environment"]["platform"] == "linux"
    assert solution["local_environment"]["cuda"]["version"] == "12.4"

    fallback_decision = {**decision, "torch_variant": "cpu", "fallback": "cpu", "selected_gpu_index": -1}
    fallback = EnvSolveModule().solve(
        repo,
        {
            "frameworks": ["torch"],
            "install_plan": [["python3", "-m", "pip", "install", "-r", "requirements.txt"]],
        },
        {"gpu_required": True, "torch_variant": "cuda_or_cpu"},
        preflight={
            "capabilities": {
                **capabilities,
                "gpu": {"status": "not_found", "devices": []},
            },
            "compatibility_decision": fallback_decision,
            "policy": {"allowed": True},
            "conda_file": {"found": False},
        },
    )
    fallback_solution = fallback.data["analysis"]["env_solution"]
    assert fallback_solution["gpu_requested"] is True
    assert fallback_solution["gpu_required"] is False
    assert fallback_solution["torch_solution"]["selected"]["variant"] == "cpu"


def test_env_solve_consumes_preflight_decision(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    decision = {
        "status": "allowed",
        "backend": "conda",
        "tool": "/opt/conda/bin/conda",
        "action": "create",
        "target_prefix": str((repo / ".conda" / "envs" / "demo").resolve()),
        "python": "3.10",
        "spec_hash": "sha256:test",
        "project_id": "project",
        "repo_fingerprint": "sha256:repo",
    }
    result = EnvSolveModule(
        local_environment={"cuda": {"available": False}},
        env_backend="auto",
    ).solve(
        repo,
        {"install_plan": [["python3", "-m", "pip", "install", "numpy"]]},
        {"gpu_required": False},
        preflight={
            "compatibility_decision": decision,
            "policy": {"allowed": True},
            "conda_file": {"found": False},
        },
    )
    conda = result.data["conda"]
    assert result.status == "passed"
    assert conda["tool"] == "/opt/conda/bin/conda"
    assert conda["environment_prefix"] == decision["target_prefix"]
    assert conda["commands"][0][0] == "/opt/conda/bin/conda"


def test_env_deploy_dry_run_has_no_ownership_side_effect(tmp_path):
    prefix = (tmp_path / ".conda" / "envs" / "demo").resolve()
    decision = {
        "status": "allowed",
        "backend": "conda",
        "tool": "/opt/conda/bin/conda",
        "action": "create",
        "target_prefix": str(prefix),
        "python": "3.10",
        "spec_hash": "sha256:test",
        "project_id": "project",
        "repo_fingerprint": "sha256:repo",
    }
    analysis = {
        "env_solution": {
            "backend": "conda",
            "gpu_required": False,
            "compatibility_decision": decision,
            "conda": {
                "action": "create",
                "tool": decision["tool"],
                "environment_prefix": str(prefix),
                "commands": [[decision["tool"], "create", "-y", "-p", str(prefix), "python=3.10"]],
                "spec": {"python": "3.10", "conda_dependencies": [], "pip_dependencies": []},
            },
        },
    }
    result = EnvDeployModule().deploy(
        tmp_path,
        analysis,
        execute=False,
        config=HarnessConfig(conda_envs_dir=str(tmp_path / ".conda" / "envs")),
    )
    assert result.status == "passed"
    assert result.data["executed"] is False
    assert not prefix.exists()


def test_env_deploy_execute_requires_preflight_mutation_authorization(tmp_path):
    prefix = (tmp_path / ".conda" / "envs" / "demo").resolve()
    tool = "/opt/conda/bin/conda"
    analysis = {
        "env_solution": {
            "backend": "conda",
            "preflight_policy": {"allowed": True, "mutation_authorized": False},
            "compatibility_decision": {
                "status": "allowed",
                "backend": "conda",
                "tool": tool,
                "action": "create",
                "target_prefix": str(prefix),
                "spec_hash": "sha256:test",
            },
            "conda": {
                "action": "create",
                "commands": [[tool, "create", "-y", "-p", str(prefix), "python=3.10"]],
            },
        },
    }
    result = EnvDeployModule().deploy(
        tmp_path,
        analysis,
        execute=True,
        config=HarnessConfig(conda_envs_dir=str(prefix.parent)),
    )
    assert result.status == "failed"
    assert result.error == "preflight mutation authorization missing"
    assert not prefix.exists()


def test_env_deploy_commits_ownership_only_after_postcheck(tmp_path):
    prefix = (tmp_path / ".conda" / "envs" / "demo").resolve()
    tool = "/opt/conda/bin/conda"
    decision = {
        "status": "allowed",
        "backend": "conda",
        "tool": tool,
        "action": "create",
        "target_prefix": str(prefix),
        "python": "3.10",
        "spec_hash": "sha256:test",
        "project_id": "project",
        "repo_fingerprint": "sha256:repo",
    }
    analysis = {
        "env_solution": {
            "backend": "conda",
            "gpu_required": False,
            "preflight_policy": {"allowed": True, "mutation_authorized": True},
            "compatibility_decision": decision,
            "conda": {
                "action": "create",
                "tool": tool,
                "environment_prefix": str(prefix),
                "commands": [[tool, "create", "-y", "-p", str(prefix), "python=3.10"]],
                "spec": {"python": "3.10", "conda_dependencies": [], "pip_dependencies": []},
            },
        },
    }

    def command_runner(cmd, cwd, timeout_seconds):
        return SimpleNamespace(
            cmd=cmd,
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
        )

    class PassingPostcheck:
        def check(self, *args, **kwargs):
            return {"status": "passed", "spec_hash": "sha256:test"}

    result = EnvDeployModule(
        command_runner=command_runner,
        postchecker=PassingPostcheck(),
    ).deploy(
        tmp_path,
        analysis,
        execute=True,
        config=HarnessConfig(conda_envs_dir=str(prefix.parent)),
        run_dir=tmp_path / "run",
        task_id="task",
    )
    marker = EnvironmentOwnership().read(prefix)
    assert result.status == "passed"
    assert marker["project_id"] == "project"
    assert marker["spec_hash"] == "sha256:test"
    assert marker["operation_id"] == result.data["operation_id"]
    operation = json.loads(
        (tmp_path / "run" / "operations" / (result.data["operation_id"] + ".json")).read_text(
            encoding="utf-8"
        )
    )
    assert operation["status"] == "committed"
    assert (tmp_path / "run" / "environment" / "environment_postcheck.json").exists()


def test_environment_operation_identity_matches_langgraph_recovery(tmp_path):
    repo = tmp_path / "repo"
    run_dir = tmp_path / "run"
    prefix = repo / ".conda" / "envs" / "demo"
    decision = {
        "target_prefix": str(prefix),
        "tool": "/opt/conda/bin/conda",
        "python": "3.10",
        "project_id": "project",
        "repo_fingerprint": "sha256:repo",
        "spec_hash": "sha256:spec",
        "action": "create",
    }
    conda = {
        "action": "create",
        "tool": decision["tool"],
        "environment_prefix": str(prefix),
        "spec": {
            "python": "3.10",
            "spec_hash": decision["spec_hash"],
            "conda_dependencies": ["numpy=1.26"],
            "pip_dependencies": [],
        },
    }
    solution = {
        "backend": "conda",
        "python": "3.10",
        "gpu_required": False,
        "compatibility_decision": decision,
        "conda": conda,
    }
    state = {
        "task_id": "task",
        "run_dir": str(run_dir),
        "repo_dir": str(repo),
        "runtime_policy": {},
        "stage_results": {
            "env_solve": {"data": {"analysis": {"env_solution": solution}}},
        },
    }
    graph_operation = GraphRecoveryAdapter().build_operation(state, "env_deploy")
    module_operation = EnvDeployModule()._build_environment_operation(
        "task", run_dir, repo, solution, conda, decision,
    )
    assert module_operation["operation_id"] == graph_operation["operation_id"]


def test_preflight_service_writes_complete_evidence_bundle(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    class GpuProbe:
        def probe(self):
            return {"status": "not_found", "devices": [], "errors": []}

    class RuntimeProbe:
        def probe(self):
            return {
                "conda": {"available": False, "path": ""},
                "mamba": {"available": False, "path": ""},
                "micromamba": {"available": False, "path": ""},
            }

    class InventoryProbe:
        def probe(self, *args, **kwargs):
            raise AssertionError("venv preflight must not inventory Conda environments")

    result = HostPreflightService(
        gpu_probe=GpuProbe(),
        runtime_probe=RuntimeProbe(),
        inventory_probe=InventoryProbe(),
    ).run(
        repo,
        {},
        {"gpu_required": False},
        HarnessConfig(env_backend="venv"),
        run_dir=tmp_path / "run",
    )
    assert result["compatibility_decision"]["status"] == "allowed"
    assert set(result["evidence_paths"]) == {
        "host_capabilities",
        "gpu_probe",
        "conda_runtime_probe",
        "conda_environment_inventory",
        "compatibility_decision",
        "policy_decision",
    }
    assert all(Path(path).exists() for path in result["evidence_paths"].values())
