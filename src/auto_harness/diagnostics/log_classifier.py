import re
from typing import Dict, List


class LogClassifier:
    TOKEN_ENV_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]*(?:TOKEN|API_KEY|API_SECRET|ACCESS_KEY|SECRET_KEY)\b")
    BUILD_PACKAGE_PATTERNS = [
        re.compile(r"Failed building wheel for ([A-Za-z0-9_.-]+)", re.IGNORECASE),
        re.compile(r"Could not build wheels for ([A-Za-z0-9_.-]+)", re.IGNORECASE),
    ]

    RULES = [
        ("dependency_missing", re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"), "install missing Python package"),
        ("import_error", re.compile(r"ImportError: (.+)"), "inspect incompatible or missing dependency"),
        ("cuda_oom", re.compile(r"CUDA out of memory", re.IGNORECASE), "reduce model size, quantize, or use larger GPU"),
        ("torch_cuda_unavailable", re.compile(r"torch not compiled with CUDA enabled", re.IGNORECASE), "install CUDA-enabled torch wheel or use CPU mode"),
        ("disk_full", re.compile(r"No space left on device", re.IGNORECASE), "free disk space or change cache directory"),
        ("auth_required", re.compile(r"(401 Unauthorized|Repository Not Found|Invalid username or password)", re.IGNORECASE), "check required access token"),
        ("git_lfs_missing", re.compile(r"git-lfs: command not found|git lfs", re.IGNORECASE), "install git-lfs and retry model asset fetch"),
        ("wheel_build_failed", re.compile(r"Could not build wheels|subprocess-exited-with-error", re.IGNORECASE), "inspect build toolchain and package pins"),
        ("numpy_abi_conflict", re.compile(r"numpy\.dtype size changed|numpy ABI", re.IGNORECASE), "pin numpy<2 or rebuild dependent package"),
        ("pydantic_conflict", re.compile(r"pydantic.*(BaseModel|ValidationError|version)", re.IGNORECASE), "check pydantic v1/v2 compatibility"),
        ("protobuf_conflict", re.compile(r"Descriptors cannot be created directly|protobuf", re.IGNORECASE), "pin compatible protobuf version"),
        ("port_in_use", re.compile(r"Address already in use|port .* already in use", re.IGNORECASE), "choose another port or stop existing process"),
    ]

    def classify(self, text: str) -> Dict:
        matches: List[Dict] = []
        for category, pattern, suggestion in self.RULES:
            match = pattern.search(text or "")
            if not match:
                continue
            signal = match.group(0)
            if match.groups():
                signal = match.group(1)
            item = {
                "category": category,
                "signal": signal[-500:],
                "suggested_fix": suggestion,
                "confidence": 0.9,
            }
            if category == "auth_required":
                item["required_env_vars"] = self._required_env_vars(text)
                item["values_recorded"] = False
            item.update(self._structured_fields(category, signal, text))
            matches.append(item)
        if matches:
            top = matches[0]
            result = {
                "category": top["category"],
                "signal": top["signal"],
                "suggested_fix": top["suggested_fix"],
                "confidence": top["confidence"],
                "matches": matches,
            }
            for key in (
                "root_cause",
                "package",
                "package_constraints",
                "recommended_actions",
                "requires",
                "rerun_from",
            ):
                if key in top:
                    result[key] = top[key]
            if top["category"] == "auth_required":
                result["required_env_vars"] = top.get("required_env_vars") or self._required_env_vars(text)
                result["values_recorded"] = False
            return result
        return {
            "category": "unknown",
            "signal": (text or "")[-500:],
            "suggested_fix": "inspect stage evidence and logs",
            "confidence": 0.2,
            "matches": [],
        }

    def _required_env_vars(self, text: str) -> List[str]:
        names = set(self.TOKEN_ENV_PATTERN.findall(text or ""))
        lower = (text or "").lower()
        if "huggingface" in lower or "hugging face" in lower or "hf_" in lower or "repository not found" in lower:
            names.add("HF_TOKEN")
        if "modelscope" in lower:
            names.add("MODELSCOPE_TOKEN")
        return sorted(names)

    def _structured_fields(self, category: str, signal: str, text: str) -> Dict:
        if category == "dependency_missing":
            package = signal.strip()
            return {
                "root_cause": "Python dependency is missing: %s" % package,
                "package": package,
                "requires": {"dependency_install": True, "network": True, "source_edit": False},
                "rerun_from": "env_deploy",
                "recommended_actions": [
                    {
                        "type": "install_package",
                        "reason": "install missing Python package",
                        "requires": {"dependency_install": True, "network": True, "source_edit": False},
                        "payload": {"package": package},
                    }
                ],
            }
        if category == "numpy_abi_conflict":
            return self._pin_action("numpy<2", "numpy ABI mismatch commonly comes from numpy 2.x with legacy wheels")
        if category == "pydantic_conflict":
            return self._pin_action("pydantic<2", "legacy Gradio/FastAPI stacks may require pydantic v1")
        if category == "protobuf_conflict":
            return self._pin_action("protobuf<=3.20.3", "generated protobuf descriptors require an older protobuf runtime")
        if category == "wheel_build_failed":
            package = self._build_package(text)
            action = {
                "type": "install_package",
                "reason": "wheel build failed; retry with a prebuilt or pinned package when available",
                "requires": {"dependency_install": True, "network": True},
                "payload": {"package": package} if package else {},
            }
            if package in {"flash-attn", "flash_attn", "xformers", "bitsandbytes"}:
                action = {
                    "type": "skip_optional_extension",
                    "reason": "optional GPU extension failed to build; use a compatible wheel or disable optional extension",
                    "requires": {"dependency_install": True, "service_restart": True},
                    "payload": {"package": package},
                }
            return {
                "root_cause": "Python wheel build failed%s" % (": %s" % package if package else ""),
                "package": package,
                "requires": action["requires"],
                "rerun_from": "env_solve",
                "recommended_actions": [action],
            }
        if category == "torch_cuda_unavailable":
            return {
                "root_cause": "Installed torch wheel does not include CUDA support",
                "requires": {"dependency_install": True, "service_restart": True},
                "rerun_from": "env_solve",
                "recommended_actions": [
                    {
                        "type": "adjust_runtime",
                        "reason": "select CUDA-enabled torch wheel or CPU fallback",
                        "requires": {"dependency_install": True, "service_restart": True},
                        "payload": {"strategy": "select compatible torch wheel or CPU fallback"},
                    }
                ],
            }
        return {}

    def _pin_action(self, constraint: str, root_cause: str) -> Dict:
        return {
            "root_cause": root_cause,
            "package_constraints": [constraint],
            "requires": {"dependency_install": True, "network": True, "source_edit": False},
            "rerun_from": "env_deploy",
            "recommended_actions": [
                {
                    "type": "install_package",
                    "reason": "install compatible dependency constraint",
                    "requires": {"dependency_install": True, "network": True, "source_edit": False},
                    "payload": {"package": constraint},
                }
            ],
        }

    def _build_package(self, text: str) -> str:
        for pattern in self.BUILD_PACKAGE_PATTERNS:
            match = pattern.search(text or "")
            if match:
                return match.group(1)
        return ""
