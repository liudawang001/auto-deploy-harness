"""Project snapshot builder for LLM plan-first deployment.

Collects project file tree, selected files, detected signals, memory hits,
and performs secret redaction. The snapshot is the input to LLMDeploymentPlanner.
"""
import hashlib
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.agent.safety import AgentInputSanitizer
from auto_harness.agent_runtime.core_evidence import CoreEvidenceSelector
from auto_harness.agent_runtime.repository_inventory import RepositoryInventoryBuilder
from auto_harness.capabilities import CapabilityDetector
from auto_harness.command_auth.discovery import CommandDiscoveryService
from auto_harness.deployment_contract import (
    DeploymentContractCompiler,
    DeploymentContractParser,
)
from auto_harness.tools.repository_policy import RepositoryReadPolicy


# Priority files to read first (in order of importance)
PRIORITY_FILES = (
    "README.md",
    "readme.md",
    "requirements.txt",
    "auto-deploy.yaml",
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    "deploy/entrypoint.sh",
    "scripts/README.md",
    "setup.py",
    "environment.yml",
    "environment.yaml",
    "Dockerfile",
    "app.py",
    "main.py",
    "server.py",
    "webui.py",
    "demo.py",
    "gradio_app.py",
    "api.py",
)

# Skip these directories entirely
SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache", ".venv", "venv"})

# Skip files with these extensions (binaries, weights, caches)
SKIP_EXTENSIONS = frozenset({
    ".bin", ".pth", ".pt", ".onnx", ".safetensors", ".gguf", ".ckpt",
    ".so", ".dylib", ".dll", ".exe", ".o", ".a",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".pyc", ".pyo", ".egg", ".whl",
})


class ProjectSnapshotBuilder:
    """Builds a redacted project snapshot for LLM consumption."""

    def __init__(
        self,
        max_files: int = 80,
        max_file_chars: int = 6000,
        max_tree_entries: int = 20000,
        context_mode: str = "eager_compat",
        core_budget_tokens: int = 12000,
    ) -> None:
        self.max_files = max_files
        self.max_file_chars = max_file_chars
        self.max_tree_entries = max_tree_entries
        self.context_mode = context_mode
        self.core_budget_tokens = core_budget_tokens
        self._last_total_file_count = 0
        self._last_excluded = {
            "sensitive_files": 0,
            "binary_files": 0,
            "oversized_files": 0,
        }

    def build(
        self,
        repo_dir: Path,
        task_id: str = "",
        memory_hits: Optional[List[Dict]] = None,
        selected_skills: Optional[List[Dict]] = None,
        skill_context: Optional[Dict] = None,
    ) -> Dict:
        """Build a project snapshot dict.

        Args:
            repo_dir: Repository directory.
            task_id: Task identifier.
            memory_hits: Optional memory hits for context.
            selected_skills: Optional list of selected skill dicts.
            skill_context: Optional skill context from SkillContextBuilder.
        """
        repo_dir = Path(repo_dir)
        memory_hits = memory_hits or []
        selected_skills = selected_skills or []
        skill_context = skill_context or {}

        # 1. Collect file tree
        file_tree = self._collect_file_tree(repo_dir)

        # 2. Select and read files. Layered mode keeps only compact core
        # evidence; eager_compat preserves the historical selection behavior.
        if self.context_mode == "layered":
            selected_files = CoreEvidenceSelector(
                budget_tokens=self.core_budget_tokens,
                max_file_chars=self.max_file_chars,
            ).select(repo_dir, file_tree, self._read_file)
        else:
            selected_files = self._select_files(repo_dir, file_tree)

        # 3. Detect signals
        detected_signals = self._detect_signals(repo_dir, file_tree, selected_files)
        capabilities, dependency_manifests = CapabilityDetector().detect(
            repo_dir, file_tree,
        )

        # 4. Redact secrets
        sanitizer = AgentInputSanitizer()
        sanitized_files = sanitizer.sanitize_selected_files(selected_files)

        # 5. Compute sha256 for each file
        file_digests = {}
        for name, content in selected_files.items():
            try:
                file_digests[name] = hashlib.sha256(
                    (repo_dir / name).read_bytes()
                ).hexdigest()
            except OSError:
                file_digests[name] = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()

        # Build selected_files with metadata
        selected_with_meta = {}
        for name in sanitized_files:
            content = sanitized_files[name]
            digest = file_digests.get(name, "")
            observed_content = content.rsplit("\n\n[truncated]", 1)[0]
            selected_with_meta[name] = {
                "path": name,
                "content": content,
                "sha256": digest,
                "observation_id": "core_%s" % digest[:12],
                "line_start": 1,
                "line_end": max(1, len(observed_content.splitlines())),
                "truncated": content.endswith("\n\n[truncated]"),
                "trust_level": "untrusted_repository",
            }

        inventory = RepositoryInventoryBuilder().build(
            repo_dir,
            file_tree,
            detected_signals=detected_signals,
            total_file_count=self._last_total_file_count,
            tree_truncated=self._last_total_file_count > len(file_tree),
            excluded=self._last_excluded,
        )
        command_registry = CommandDiscoveryService().discover(
            repo_dir,
            file_tree,
            inventory["repository_fingerprint"],
        )
        contract_result = DeploymentContractParser().parse_repo(repo_dir)
        deployment_candidates = []
        if contract_result.get("valid"):
            command_registry, deployment_candidate = (
                DeploymentContractCompiler().compile_registry(
                    repo_dir,
                    contract_result["contract"],
                    command_registry,
                )
            )
            deployment_candidates.append(deployment_candidate.to_dict())
        else:
            # Phase B2: expose adapter-composed deployment candidates so the
            # planner can select existing candidates by id even without an
            # explicit contract. Composition grants no execution authority.
            from auto_harness.deployment_adapters import (
                CandidateComposer,
                DeploymentAdapterRegistry,
                DetectionContext,
            )
            from auto_harness.modules.analyzer import ProjectAnalyzer

            legacy_frameworks = ProjectAnalyzer()._detect_frameworks(
                repo_dir, file_tree,
            )
            adapter_context = DetectionContext(
                repo_dir=Path(repo_dir),
                files=tuple(file_tree),
                capabilities=capabilities,
                legacy_frameworks=tuple(legacy_frameworks),
            )
            proposals = DeploymentAdapterRegistry.builtins().proposals(adapter_context)
            deployment_candidates = [
                item.to_dict()
                for item in CandidateComposer().compose(
                    proposals["run"],
                    proposals["environment"],
                    proposals["verify"],
                )
            ]

        return {
            "schema_version": 2,
            "context_mode": self.context_mode,
            "task_id": task_id,
            # repo_dir is needed by local runtime artifacts but planners must
            # not treat it as an authority or echo it into plans.
            "repo_dir": str(repo_dir),
            "repository_fingerprint": inventory["repository_fingerprint"],
            "repository_inventory": inventory,
            "command_registry": command_registry.to_dict(),
            "deployment_contract": self._contract_snapshot(contract_result),
            "deployment_candidates": deployment_candidates,
            "file_tree": file_tree,
            "file_tree_summary": {
                "total_file_count": self._last_total_file_count,
                "omitted_file_count": max(
                    0, self._last_total_file_count - len(file_tree)
                ),
                "truncated": self._last_total_file_count > len(file_tree),
            },
            "selected_files": selected_with_meta,
            "detected_signals": detected_signals,
            "capabilities": capabilities.to_dict(),
            "capability_evidence": [
                item.to_dict() for item in capabilities.evidence
            ],
            "dependency_manifests": [
                item.to_dict() for item in dependency_manifests
            ],
            "memory_hits": memory_hits,
            "selected_skills": selected_skills,
            "skill_context": skill_context,
            "redactions": sanitizer.redactions,
            "untrusted_content_risks": sanitizer.risks,
        }

    @staticmethod
    def _contract_snapshot(result: Dict) -> Dict:
        snapshot = {
            "found": bool(result.get("found")),
            "valid": bool(result.get("valid")),
            "path": str(result.get("path") or "auto-deploy.yaml"),
        }
        if result.get("reason_code"):
            snapshot["reason_code"] = str(result["reason_code"])
        if result.get("disabled"):
            snapshot["disabled"] = True
        contract = result.get("contract")
        if result.get("valid") and contract is not None:
            snapshot.update(contract.to_dict())
        return snapshot

    def _collect_file_tree(self, repo_dir: Path) -> List[str]:
        """Collect the full file tree, skipping .git and binary dirs."""
        result: List[str] = []
        seen = set()
        seen_files = set()
        self._last_total_file_count = 0
        self._last_excluded = {
            "sensitive_files": 0,
            "binary_files": 0,
            "oversized_files": 0,
        }
        for name in PRIORITY_FILES:
            path = repo_dir / name
            if not RepositoryReadPolicy.path_allowed(name):
                self._last_excluded["sensitive_files"] += 1
                continue
            if path.is_file() and path.suffix.lower() not in SKIP_EXTENSIONS:
                try:
                    identity = (path.stat().st_dev, path.stat().st_ino)
                except OSError:
                    continue
                if identity in seen_files:
                    continue
                result.append(name)
                seen.add(name)
                seen_files.add(identity)
        root_parts = len(repo_dir.parts)
        for path in sorted(repo_dir.rglob("*")):
            if path.is_symlink() or path.is_dir():
                continue
            # Skip files inside skipped directories
            relative_parts = path.parts[root_parts:]
            if any(part in SKIP_DIRS for part in relative_parts):
                continue
            # Ignore Harness-owned runtime artifacts created beside a target
            # checkout.  A cloned repository can legitimately live under a
            # path that itself contains a directory named ``runs``; only the
            # target-relative first component is relevant here.
            if relative_parts and relative_parts[0] in {"runs", ".conda"}:
                continue
            # Skip binary/weight files by extension
            if path.suffix.lower() in SKIP_EXTENSIONS:
                self._last_excluded["binary_files"] += 1
                continue
            try:
                rel = str(path.relative_to(repo_dir))
            except ValueError:
                continue
            if not RepositoryReadPolicy.path_allowed(rel):
                self._last_excluded["sensitive_files"] += 1
                continue
            try:
                identity = (path.stat().st_dev, path.stat().st_ino)
            except OSError:
                continue
            if identity in seen_files:
                continue
            if rel in seen:
                continue
            self._last_total_file_count += 1
            if len(result) < self.max_tree_entries:
                result.append(rel)
                seen.add(rel)
                seen_files.add(identity)
        self._last_total_file_count += len(
            [name for name in result if name in PRIORITY_FILES]
        )
        return result

    def collect_file_tree(self, repo_dir: Path) -> List[str]:
        """Public metadata-only tree collection used for freshness checks."""
        return self._collect_file_tree(Path(repo_dir))

    def _select_files(self, repo_dir: Path, file_tree: List[str]) -> Dict[str, str]:
        """Select priority files and read their content."""
        selected: Dict[str, str] = {}
        file_set = set(file_tree)

        # Read priority files first
        for name in PRIORITY_FILES:
            if (name in file_set or (repo_dir / name).is_file()) and len(selected) < self.max_files:
                path = repo_dir / name
                content = self._read_file(path)
                if content is not None:
                    selected[name] = content

        # Then read remaining .py, .yml, .yaml, .toml, .txt, .md files
        for rel in file_tree:
            if len(selected) >= self.max_files:
                break
            if rel in selected:
                continue
            if Path(rel).suffix.lower() not in (
                ".py", ".yml", ".yaml", ".toml", ".txt", ".md", ".cfg", ".ini", ".json",
            ):
                continue
            # Skip files in subdirectories if we already have enough
            if "/" in rel and len(selected) >= self.max_files // 2:
                continue
            path = repo_dir / rel
            content = self._read_file(path)
            if content is not None:
                selected[rel] = content

        return selected

    def _read_file(self, path: Path) -> Optional[str]:
        """Read a single file, truncated to max_file_chars."""
        try:
            if not path.is_file():
                return None
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        if len(text) > self.max_file_chars:
            if path.name.lower().startswith("readme"):
                text = self._readme_deployment_excerpt(text)
            else:
                text = text[: self.max_file_chars] + "\n\n[truncated]"
        return text

    def _readme_deployment_excerpt(self, text: str) -> str:
        """Keep bounded launch evidence from a long public README.

        Repositories often place Quick Start and source-install instructions
        after a long feature overview.  Prefix truncation therefore hides the
        most authoritative deployment evidence.  Preserve the document head
        plus bounded windows around deployment commands and localhost URLs.
        """
        lines = str(text).splitlines()
        selected = set(range(min(len(lines), 12)))
        deployment = re.compile(
            r"(?:pip\s+install|npm\s+(?:ci|run\s+build)|python\s+-m\s+venv|"
            r"\b(?:init|run|serve|server|start|app)\b|"
            r"https?://(?:127\.0\.0\.1|localhost):\d+)",
            re.IGNORECASE,
        )
        heading = re.compile(
            r"^#{1,4}\s+.*(?:quick\s*start|install|deploy|run|source)",
            re.IGNORECASE,
        )
        for index, line in enumerate(lines):
            if not (deployment.search(line) or heading.search(line.strip())):
                continue
            selected.update(range(max(0, index - 1), min(len(lines), index + 3)))

        output: List[str] = []
        previous = -2
        ordered = sorted(selected)
        # Newer/more specific deployment evidence is commonly near the end of
        # a README (for example "Install From Source").  Allocate half the
        # bounded output to the front and half to the tail so neither Quick
        # Start nor source-build instructions can crowd out the other.
        if len(ordered) > 80:
            ordered = ordered[:40] + ordered[-40:]
        for index in ordered:
            if index != previous + 1:
                output.append("[...]")
            output.append(lines[index])
            previous = index
            if len("\n".join(output)) >= self.max_file_chars - 64:
                break
        bounded = "\n".join(output)
        if len(bounded) > self.max_file_chars:
            bounded = bounded[: self.max_file_chars]
        return bounded + "\n\n[deployment-focused excerpt]\n[truncated]"

    def _detect_signals(
        self,
        repo_dir: Path,
        file_tree: List[str],
        selected_files: Dict[str, str],
    ) -> Dict:
        """Detect frameworks, entrypoint candidates, dependency files, model mentions, ports."""
        file_set = set(file_tree)
        # Compact README excerpts do not preserve original line numbers. Use
        # bounded raw authoritative files for deterministic command discovery
        # so an LLM can request the exact source lines returned in signals.
        # The snapshot sent to the provider remains compact and sanitized.
        signal_files = dict(selected_files)
        for rel in (
            "README.md", "readme.md", "pyproject.toml", "Makefile",
            "deploy/entrypoint.sh", "docker/entrypoint.sh",
        ):
            path = repo_dir / rel
            if rel not in file_set or not path.is_file():
                continue
            try:
                signal_files[rel] = path.read_text(
                    encoding="utf-8", errors="ignore",
                )[:1_000_000]
            except OSError:
                continue
        all_text = "\n".join(signal_files.values()).lower()

        # Framework detection (reuse analyzer patterns)
        frameworks: List[str] = []
        for key in ("gradio", "streamlit", "fastapi", "flask", "torch", "transformers", "vllm"):
            if key in all_text:
                frameworks.append(key)
        if "httpserver" in all_text or "basehttprequesthandler" in all_text or "from http.server" in all_text:
            frameworks.append("http.server")
        if "openai-compatible" in all_text or "openai compatible" in all_text:
            frameworks.append("openai_compatible")

        # Entrypoint candidates
        entrypoint_candidates = [
            name for name in ("app.py", "main.py", "server.py", "webui.py", "demo.py", "gradio_app.py", "api.py")
            if name in file_set
        ]

        # Dependency files
        dependency_files = [
            name for name in (
                "auto-deploy.yaml", "requirements.txt", "pyproject.toml", "uv.lock",
                "setup.py", "environment.yml", "environment.yaml",
            )
            if name in file_set
        ]

        # Model mentions (HuggingFace / ModelScope references)
        model_mentions: List[str] = []
        model_pattern = re.compile(
            r'["\']([A-Za-z0-9_\-/]+(?:/[A-Za-z0-9_\-]+)+)["\']',
        )
        for line in all_text.splitlines():
            if "from_pretrained" in line or "modelscope" in line:
                for match in model_pattern.finditer(line):
                    candidate = match.group(1)
                    if "/" in candidate and not candidate.startswith(".") and len(candidate) < 200:
                        model_mentions.append(candidate)

        # Port detection - look for common port patterns in source
        ports: List[int] = []
        # Match: HTTPServer(('host', PORT)), .run(port=PORT), port=PORT, :PORT
        port_patterns = [
            re.compile(r"HTTPServer\(\s*\(\s*['\"][^'\"]*['\"]\s*,\s*(\d{2,5})\s*\)", re.IGNORECASE),
            re.compile(r"(?:port\s*[=:]\s*)(\d{2,5})", re.IGNORECASE),
            re.compile(r"uvicorn\.run\([^)]*port\s*=\s*(\d{2,5})", re.IGNORECASE),
        ]
        for pattern in port_patterns:
            for match in pattern.finditer(all_text):
                try:
                    port = int(match.group(1))
                    if 1024 <= port <= 65535 and port not in ports:
                        ports.append(port)
                except (ValueError, TypeError):
                    continue

        console_scripts = self._console_scripts(signal_files.get("pyproject.toml", ""))
        documented_run_commands = self._documented_run_commands(
            signal_files,
            [item["name"] for item in console_scripts],
        )
        documented_ports = self._documented_local_ports(
            signal_files,
            [item["name"] for item in console_scripts],
        )
        if len(documented_ports) == 1:
            for item in documented_run_commands:
                if not int(item.get("expected_port") or 0):
                    item["expected_port"] = documented_ports[0]
        documented_setup_commands = self._documented_setup_commands(
            signal_files,
            [item["name"] for item in console_scripts],
        )
        python_requires = self._python_requires(signal_files.get("pyproject.toml", ""))
        source_build_commands = self._source_build_commands(file_set, signal_files)

        # Other signals
        has_dockerfile = "Dockerfile" in file_set
        has_environment_yml = "environment.yml" in file_set or "environment.yaml" in file_set

        return {
            "frameworks": sorted(set(frameworks)),
            "entrypoint_candidates": entrypoint_candidates,
            "dependency_files": dependency_files,
            "model_mentions": model_mentions[:10],
            "ports": ports[:5],
            "has_dockerfile": has_dockerfile,
            "has_environment_yml": has_environment_yml,
            "console_scripts": console_scripts,
            "documented_run_commands": documented_run_commands,
            "documented_setup_commands": documented_setup_commands,
            "python_requires": python_requires,
            "source_build_commands": source_build_commands,
        }

    @staticmethod
    def _source_build_commands(file_set: set, selected_files: Dict[str, str]) -> List[Dict]:
        """Detect narrowly-scoped, lockfile-backed frontend source builds.

        A Python source checkout may intentionally omit generated dashboard
        assets that are present in release wheels.  Only emit commands when
        repository-owned build instructions, an npm lockfile, and the missing
        output artifact all agree.  Commands remain argv arrays and never use
        a shell wrapper.
        """
        makefile = str(selected_files.get("Makefile", ""))
        readmes = "\n".join(
            content for path, content in selected_files.items()
            if Path(path).name.lower().startswith("readme")
        )
        has_dashboard_source = "dashboard/package.json" in file_set
        has_lock = "dashboard/package-lock.json" in file_set
        built_index_missing = "src/octop/dashboard/index.html" not in file_set
        makefile_declares_build = (
            "build-frontend:" in makefile
            and "cd $(DASHBOARD_DIR) && npm ci" in makefile
            and "npm run build" in makefile
        )
        if (
            has_dashboard_source
            and has_lock
            and built_index_missing
            and makefile_declares_build
        ):
            return [
                {
                    "cmd": ["npm", "--prefix", "dashboard", "ci"],
                    "source": "Makefile",
                    "reason": "lockfile-backed frontend dependencies required by build-frontend",
                },
                {
                    "cmd": ["npm", "--prefix", "dashboard", "run", "build"],
                    "source": "Makefile",
                    "reason": "build missing production dashboard artifact",
                },
            ]

        # Common source-release layout: a console/ SPA is built into
        # console/dist before an editable Python package is launched.  Require
        # both the lockfile and explicit public documentation so an arbitrary
        # nested package.json cannot introduce build commands.
        has_console_source = "console/package.json" in file_set
        has_console_lock = "console/package-lock.json" in file_set
        console_dist_missing = "console/dist/index.html" not in file_set
        documented_console_build = bool(re.search(
            r"cd\s+console\s*&&\s*npm\s+ci\s*&&\s*npm\s+run\s+build",
            readmes,
        ))
        if (
            has_console_source
            and has_console_lock
            and console_dist_missing
            and documented_console_build
        ):
            source = next(
                (
                    path for path, content in selected_files.items()
                    if Path(path).name.lower().startswith("readme")
                    and re.search(
                        r"cd\s+console\s*&&\s*npm\s+ci\s*&&\s*npm\s+run\s+build",
                        content,
                    )
                ),
                "README.md",
            )
            return [
                {
                    "cmd": ["npm", "--prefix", "console", "ci"],
                    "source": source,
                    "reason": "lockfile-backed console dependencies required by source install",
                },
                {
                    "cmd": ["npm", "--prefix", "console", "run", "build"],
                    "source": source,
                    "reason": "build missing console/dist production artifact",
                },
            ]

        # Common Python monorepo layout: a Vite application under
        # src/frontend is built before a Python CLI serves the resulting
        # static assets. The public source-run target is the trust anchor;
        # package.json alone is not enough to authorize a build.
        has_frontend_source = "src/frontend/package.json" in file_set
        has_frontend_lock = "src/frontend/package-lock.json" in file_set
        frontend_build_missing = "src/frontend/build/index.html" not in file_set
        source_target_builds_frontend = bool(re.search(
            r"(?m)^run_cli:\s*[^\n]*(?:install_frontend|build_frontend)[^\n]*$",
            makefile,
        ))
        if (
            has_frontend_source
            and has_frontend_lock
            and frontend_build_missing
            and source_target_builds_frontend
        ):
            return [
                {
                    "cmd": ["npm", "--prefix", "src/frontend", "ci"],
                    "source": "Makefile",
                    "reason": "lockfile-backed frontend dependencies required by source run target",
                },
                {
                    "cmd": ["npm", "--prefix", "src/frontend", "run", "build"],
                    "source": "Makefile",
                    "reason": "build missing source-checkout frontend assets",
                },
            ]
        return []

    @staticmethod
    def _console_scripts(pyproject: str) -> List[Dict[str, str]]:
        """Extract PEP 621 console scripts without executing repository code."""
        in_scripts = False
        scripts: List[Dict[str, str]] = []
        for raw in str(pyproject or "").splitlines():
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                in_scripts = line == "[project.scripts]"
                continue
            if not in_scripts or not line or line.startswith("#") or "=" not in line:
                continue
            name, target = (part.strip() for part in line.split("=", 1))
            name = name.strip('"\'')
            target = target.split("#", 1)[0].strip().strip('"\'')
            if (
                re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*", target)
            ):
                scripts.append({"name": name, "target": target, "source": "pyproject.toml"})
        return scripts[:20]

    @staticmethod
    def _documented_run_commands(
        selected_files: Dict[str, str],
        script_names: List[str],
    ) -> List[Dict]:
        """Extract safe service commands whose executable is a declared script."""
        names = set(script_names)
        if not names:
            return []
        preferred_verbs = {"run", "serve", "server", "start", "web", "api", "app"}
        commands: List[Dict] = []
        for path, content in selected_files.items():
            normalized = str(path).replace("\\", "/").lower()
            is_readme = Path(path).name.lower().startswith("readme")
            is_deploy_entrypoint = normalized in {
                "deploy/entrypoint.sh", "docker/entrypoint.sh",
            }
            if not (is_readme or is_deploy_entrypoint):
                continue
            lines = str(content).splitlines()
            for line_number, raw in enumerate(lines, start=1):
                raw_line = raw.strip()
                # Markdown examples commonly append an explanatory comment.
                # It is evidence for the port, but never part of the command.
                line = re.split(r"\s+#\s+", raw_line, maxsplit=1)[0].strip()
                if not line or any(token in line for token in (";", "&&", "||", "|", "`", "$(`")):
                    continue
                try:
                    cmd = shlex.split(line, posix=True)
                except ValueError:
                    continue
                if len(cmd) < 2 or cmd[0] not in names or cmd[1].lower() not in preferred_verbs:
                    continue
                if any(not isinstance(arg, str) or "\x00" in arg for arg in cmd):
                    continue
                commands.append({
                    "cmd": cmd,
                    "source": path,
                    "line": line_number,
                    "expected_port": ProjectSnapshotBuilder._documented_port(
                        "\n".join(lines[line_number - 1:line_number + 4]), cmd,
                    ),
                })
        # Prefer explicit ports, then explicit hosts, then shorter commands.
        commands.sort(key=lambda item: (
            "--port" not in item["cmd"],
            "--host" not in item["cmd"],
            len(item["cmd"]),
            item["source"],
            item["line"],
        ))
        deduped: List[Dict] = []
        seen = set()
        for item in commands:
            key = tuple(item["cmd"])
            if key not in seen:
                deduped.append(item)
                seen.add(key)
        return deduped[:10]

    @staticmethod
    def _documented_setup_commands(
        selected_files: Dict[str, str],
        script_names: List[str],
    ) -> List[Dict]:
        """Extract non-interactive initialization for a declared project CLI."""
        names = set(script_names)
        commands: List[Dict] = []
        for path, content in selected_files.items():
            normalized = str(path).replace("\\", "/").lower()
            is_readme = Path(path).name.lower().startswith("readme")
            is_deploy_entrypoint = normalized in {
                "deploy/entrypoint.sh", "docker/entrypoint.sh",
            }
            if not (is_readme or is_deploy_entrypoint):
                continue
            for line_number, raw in enumerate(str(content).splitlines(), start=1):
                line = re.split(r"\s+#\s+", raw.strip(), maxsplit=1)[0].strip()
                if not line or any(token in line for token in (";", "&&", "||", "|", "`", "$(")):
                    continue
                try:
                    cmd = shlex.split(line, posix=True)
                except ValueError:
                    continue
                if (
                    len(cmd) < 2
                    or cmd[0] not in names
                    or cmd[1] not in {"init", "setup"}
                    or any(
                        arg not in {
                            "--defaults", "--non-interactive", "--yes", "-y",
                            "--accept-security",
                        }
                        for arg in cmd[2:]
                    )
                ):
                    continue
                # Initialization must be explicitly non-interactive.
                if len(cmd) == 2:
                    continue
                commands.append({"cmd": cmd, "source": path, "line": line_number})
        # Prefer the fully unattended variant when both the README and the
        # project's own deployment entrypoint document the same initializer.
        commands.sort(
            key=lambda item: "--accept-security" in item["cmd"],
            reverse=True,
        )
        deduped: List[Dict] = []
        seen = set()
        for item in commands:
            key = tuple(item["cmd"])
            if key not in seen:
                deduped.append(item)
                seen.add(key)
        return deduped[:5]

    @staticmethod
    def _documented_port(raw_line: str, cmd: List[str]) -> int:
        for index, arg in enumerate(cmd[:-1]):
            if arg in ("--port", "-p"):
                try:
                    port = int(cmd[index + 1])
                except (TypeError, ValueError):
                    break
                if 0 < port <= 65535:
                    return port
        match = re.search(r"https?://[^\s/:]+:(\d{2,5})(?:\b|/)", raw_line)
        if match:
            port = int(match.group(1))
            if 0 < port <= 65535:
                return port
        # The documented URL may be separated from the command in the compact
        # README excerpt.  QwenPaw and similar CLIs still use their canonical
        # local console port when it appears anywhere in public instructions.
        match = re.search(r"(?:127\.0\.0\.1|localhost):(\d{2,5})", raw_line)
        if match:
            port = int(match.group(1))
            if 0 < port <= 65535:
                return port
        return 0

    @staticmethod
    def _documented_local_ports(
        selected_files: Dict[str, str],
        script_names: List[str],
    ) -> List[int]:
        ports: List[int] = []
        names = set(script_names)
        for path, content in selected_files.items():
            if not Path(path).name.lower().startswith("readme"):
                continue
            normalized_path = str(path).replace("\\", "/")
            if "/" in normalized_path and not normalized_path.lower().startswith("scripts/readme"):
                continue
            # A repository may contain unrelated plugin/service READMEs with
            # their own ports.  Only use documents that also mention the
            # declared project CLI.
            if names and not any(
                re.search(r"\b%s\b" % re.escape(name), str(content))
                for name in names
            ):
                continue
            for match in re.finditer(
                r"https?://(?:127\.0\.0\.1|localhost):(\d{2,5})(?:\b|/)",
                str(content),
                re.IGNORECASE,
            ):
                port = int(match.group(1))
                if 0 < port <= 65535 and port not in ports:
                    ports.append(port)
        return ports[:5]

    @staticmethod
    def _python_requires(pyproject: str) -> str:
        in_project = False
        for raw in str(pyproject or "").splitlines():
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                in_project = line == "[project]"
                continue
            if in_project and re.match(r"^requires-python\s*=", line):
                return line.split("=", 1)[1].split("#", 1)[0].strip().strip('"\'')
        return ""
