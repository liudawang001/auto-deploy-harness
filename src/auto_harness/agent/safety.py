import re
from typing import Dict, List


class AgentInputSanitizer:
    SECRET_PATTERNS = (
        ("huggingface_token", re.compile(r"hf_[A-Za-z0-9_]{12,}")),
        ("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{16,}")),
        ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE)),
        ("api_key", re.compile(r"(?i)(api_key\s*=\s*)[^\s'\"\n]+")),
        ("api_secret", re.compile(r"(?i)(api_secret\s*=\s*)[^\s'\"\n]+")),
        ("password", re.compile(r"(?i)(password\s*=\s*)[^\s'\"\n]+")),
        ("env_secret", re.compile(r"(?m)^([A-Z][A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD)\s*=\s*)[^\s'\"\n]+")),
        ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
        ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}")),
    )
    INJECTION_PATTERNS = (
        ("prompt_injection", re.compile(r"ignore previous instructions", re.IGNORECASE)),
        ("prompt_injection", re.compile(r"disregard system prompt", re.IGNORECASE)),
        ("shell_request", re.compile(r"run shell|execute shell|rm -rf|delete files", re.IGNORECASE)),
        ("secret_exfiltration", re.compile(r"print secrets|exfiltrate token|leak token", re.IGNORECASE)),
        ("external_exfiltration", re.compile(r"curl\s+https?://|wget\s+https?://", re.IGNORECASE)),
        ("decode_execute", re.compile(r"base64\s+.*decode.*execute|base64\s+-d.*sh", re.IGNORECASE)),
    )
    DENYLIST_MARKERS = (".env", "secret", "credential", "token", "key")

    def __init__(self) -> None:
        self.risks: List[Dict] = []
        self.redactions: List[Dict] = []

    def sanitize_selected_files(self, files: Dict[str, str]) -> Dict[str, str]:
        self.risks = []
        self.redactions = []
        sanitized = {}
        for name, text in (files or {}).items():
            lowered = name.lower()
            if any(marker in lowered for marker in self.DENYLIST_MARKERS):
                self.risks.append({"file": name, "risk": "denied_file_name"})
                continue
            scan = self.scan_text(str(text))
            for risk in scan["risks"]:
                risk = dict(risk)
                risk["file"] = name
                self.risks.append(risk)
            for redaction in scan["redactions"]:
                item = dict(redaction)
                item["file"] = name
                self.redactions.append(item)
            sanitized[name] = scan["text"]
        return sanitized

    def scan_text(self, text: str) -> Dict:
        redactions = []
        risks = []
        sanitized = str(text)
        for kind, pattern in self.SECRET_PATTERNS:
            matches = list(pattern.finditer(sanitized))
            if matches:
                redactions.append({"type": kind, "count": len(matches)})
                if kind in {"api_key", "api_secret", "password", "env_secret"}:
                    sanitized = pattern.sub(lambda match: match.group(1) + "[REDACTED_SECRET]", sanitized)
                else:
                    sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
        for kind, pattern in self.INJECTION_PATTERNS:
            if pattern.search(sanitized):
                risks.append({"risk": kind})
        return {"text": sanitized, "risks": risks, "redactions": redactions}

    def redact_value(self, value):
        if isinstance(value, dict):
            return {key: self.redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, str):
            return self.scan_text(value)["text"]
        return value
