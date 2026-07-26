"""Tests for Docker minimal security hardening.

Validates:
- Docker command drops all capabilities
- Docker command sets no-new-privileges
- Docker command has memory, CPU, PID limits
- Docker command mounts /tmp as tmpfs
- Host network is rejected
- Repo mount mode is recorded
- Secret values not in command
"""
import pytest

from auto_harness.runtime.sandbox import (
    DockerSandboxBackend,
    DockerSandboxBackend_FORBIDDEN_SECRETS,
)
from pathlib import Path


class TestDockerSecurity:
    """Test Docker sandbox security hardening."""

    def test_docker_command_drops_all_capabilities(self):
        """Docker command must include --cap-drop ALL."""
        backend = DockerSandboxBackend()
        cmd = backend.wrap(Path("/repo"), ["python", "-c", "print('hello')"])
        assert "--cap-drop" in cmd.effective_cmd
        cap_idx = cmd.effective_cmd.index("--cap-drop")
        assert cmd.effective_cmd[cap_idx + 1] == "ALL"

    def test_docker_command_sets_no_new_privileges(self):
        """Docker command must include --security-opt no-new-privileges."""
        backend = DockerSandboxBackend()
        cmd = backend.wrap(Path("/repo"), ["python", "-c", "print('hello')"])
        assert "--security-opt" in cmd.effective_cmd
        opt_idx = cmd.effective_cmd.index("--security-opt")
        assert cmd.effective_cmd[opt_idx + 1] == "no-new-privileges"

    def test_docker_command_has_memory_cpu_pid_limits(self):
        """Docker command must include --memory, --cpus, --pids-limit."""
        backend = DockerSandboxBackend()
        cmd = backend.wrap(Path("/repo"), ["python", "-c", "print('hello')"])
        assert "--memory" in cmd.effective_cmd
        mem_idx = cmd.effective_cmd.index("--memory")
        assert cmd.effective_cmd[mem_idx + 1] == "8g"

        assert "--cpus" in cmd.effective_cmd
        cpus_idx = cmd.effective_cmd.index("--cpus")
        assert cmd.effective_cmd[cpus_idx + 1] == "4.0"

        assert "--pids-limit" in cmd.effective_cmd
        pids_idx = cmd.effective_cmd.index("--pids-limit")
        assert cmd.effective_cmd[pids_idx + 1] == "512"

    def test_docker_command_mounts_tmpfs(self):
        """Docker command must include --tmpfs for /tmp."""
        backend = DockerSandboxBackend()
        cmd = backend.wrap(Path("/repo"), ["python", "-c", "print('hello')"])
        assert "--tmpfs" in cmd.effective_cmd
        tmpfs_idx = cmd.effective_cmd.index("--tmpfs")
        tmpfs_val = cmd.effective_cmd[tmpfs_idx + 1]
        assert tmpfs_val.startswith("/tmp:")
        assert "noexec" in tmpfs_val
        assert "nosuid" in tmpfs_val

    def test_host_network_is_rejected(self):
        """Host network must be rejected."""
        with pytest.raises(ValueError, match="host network is not allowed"):
            DockerSandboxBackend(network="host")

    def test_repo_mount_mode_is_recorded(self):
        """Repo mount mode must be recorded in security_options."""
        backend_rw = DockerSandboxBackend(repo_mount_mode="rw")
        cmd_rw = backend_rw.wrap(Path("/repo"), ["python"])
        assert cmd_rw.security_options["repo_mount_mode"] == "rw"

        # Verify the volume mount includes :rw
        vol_idx = cmd_rw.effective_cmd.index("-v")
        assert ":rw" in cmd_rw.effective_cmd[vol_idx + 1]

        backend_ro = DockerSandboxBackend(repo_mount_mode="ro")
        cmd_ro = backend_ro.wrap(Path("/repo"), ["python"])
        assert cmd_ro.security_options["repo_mount_mode"] == "ro"
        vol_idx2 = cmd_ro.effective_cmd.index("-v")
        assert ":ro" in cmd_ro.effective_cmd[vol_idx2 + 1]

    def test_secret_values_not_in_command(self):
        """Forbidden secret env vars must not appear in the command."""
        backend = DockerSandboxBackend()
        cmd = backend.wrap(Path("/repo"), ["python", "-c", "print('hello')"])
        # The command should not contain any secret variable names
        cmd_str = " ".join(cmd.effective_cmd)
        for secret in DockerSandboxBackend_FORBIDDEN_SECRETS:
            assert secret not in cmd_str, "secret %s found in command" % secret

    def test_security_options_recorded(self):
        """All security options must be recorded in the artifact."""
        backend = DockerSandboxBackend()
        cmd = backend.wrap(Path("/repo"), ["python"])
        opts = cmd.security_options
        assert opts is not None
        assert opts["cap_drop_all"] is True
        assert opts["no_new_privileges"] is True
        assert opts["memory"] == "8g"
        assert opts["cpus"] == 4.0
        assert opts["pids_limit"] == 512
        assert opts["repo_mount_mode"] == "rw"
        assert opts["read_only_rootfs"] is False

    def test_custom_security_parameters(self):
        """Custom security parameters must be applied."""
        backend = DockerSandboxBackend(
            memory="4g",
            cpus=2.0,
            pids_limit=256,
            tmpfs_size="512m",
            repo_mount_mode="ro",
        )
        cmd = backend.wrap(Path("/repo"), ["python"])
        assert cmd.security_options["memory"] == "4g"
        assert cmd.security_options["cpus"] == 2.0
        assert cmd.security_options["pids_limit"] == 256

    def test_cap_drop_false_removes_flag(self):
        """Setting cap_drop_all=False should not add --cap-drop."""
        backend = DockerSandboxBackend(cap_drop_all=False)
        cmd = backend.wrap(Path("/repo"), ["python"])
        assert "--cap-drop" not in cmd.effective_cmd

    def test_no_new_privileges_false_removes_flag(self):
        """Setting no_new_privileges=False should not add --security-opt."""
        backend = DockerSandboxBackend(no_new_privileges=False)
        cmd = backend.wrap(Path("/repo"), ["python"])
        assert "no-new-privileges" not in cmd.effective_cmd

    def test_user_flag_when_set(self):
        """Setting user should add --user flag."""
        backend = DockerSandboxBackend(user="1000:1000")
        cmd = backend.wrap(Path("/repo"), ["python"])
        assert "--user" in cmd.effective_cmd
        user_idx = cmd.effective_cmd.index("--user")
        assert cmd.effective_cmd[user_idx + 1] == "1000:1000"

    def test_read_only_rootfs(self):
        """Setting read_only_rootfs=True should add --read-only."""
        backend = DockerSandboxBackend(read_only_rootfs=True)
        cmd = backend.wrap(Path("/repo"), ["python"])
        assert "--read-only" in cmd.effective_cmd
        assert cmd.security_options["read_only_rootfs"] is True


class TestDockerPhaseProfiles:
    def test_install_profile_allows_only_required_writes(self, tmp_path):
        backend = DockerSandboxBackend.for_phase(
            "install",
            model_cache_dir=tmp_path / "cache",
            cap_drop_all=False,
            no_new_privileges=False,
            repo_mount_mode="ro",
            read_only_rootfs=True,
        )
        command = backend.wrap(tmp_path / "repo", ["pip", "install", "-r", "requirements.txt"])

        assert command.security_options["phase"] == "install"
        assert command.security_options["repo_mount_mode"] == "rw"
        assert command.security_options["read_only_rootfs"] is False
        assert command.security_options["model_cache_mount_mode"] == "rw"
        assert command.security_options["cap_drop_all"] is True
        assert command.security_options["no_new_privileges"] is True

    @pytest.mark.parametrize("phase", ["runtime", "verify"])
    def test_runtime_profiles_force_non_root_and_read_only(self, tmp_path, phase):
        backend = DockerSandboxBackend.for_phase(
            phase,
            gpus="all",
            model_cache_dir=tmp_path / "cache",
            repo_mount_mode="rw",
            read_only_rootfs=False,
            user="",
        )
        command = backend.wrap(tmp_path / "repo", ["python", "app.py"])
        joined = " ".join(command.effective_cmd)

        assert command.security_options["phase"] == phase
        assert command.security_options["repo_mount_mode"] == "ro"
        assert command.security_options["read_only_rootfs"] is True
        assert command.security_options["user"] == "65532:65532"
        assert command.security_options["model_cache_mount_mode"] == "ro"
        assert "--read-only" in command.effective_cmd
        assert "HOME=/tmp" in command.effective_cmd
        assert "PYTHONDONTWRITEBYTECODE=1" in command.effective_cmd
        assert ":/workspace/model_cache:ro" in joined
        assert command.gpus == ("none" if phase == "verify" else "all")

    def test_env_deploy_and_runner_use_distinct_profiles(self, tmp_path):
        from auto_harness.modules.env_deploy import EnvDeployModule
        from auto_harness.modules.runner import RunnerModule

        deploy = EnvDeployModule().deploy(
            tmp_path,
            {"install_plan": [["pip", "install", "flask"]]},
            execution_backend="docker",
        )
        runner = RunnerModule().run(
            tmp_path,
            {"run_candidates": [{"cmd": ["python", "app.py"], "expected_port": 7860}]},
            execution_backend="docker",
        )

        install_security = deploy.data["sandbox"]["commands"][0]["security_options"]
        runtime_security = runner.data["sandbox"]["security_options"]
        assert install_security["phase"] == "install"
        assert install_security["repo_mount_mode"] == "rw"
        assert runtime_security["phase"] == "runtime"
        assert runtime_security["repo_mount_mode"] == "ro"
