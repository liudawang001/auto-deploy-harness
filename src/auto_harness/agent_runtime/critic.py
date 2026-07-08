import json
from typing import Dict


class AgentCritic:
    """Single-agent quality gate. It never executes tools or marks success."""

    def critique(self, step: Dict) -> Dict:
        tool_call = step.get("tool_call") if isinstance(step.get("tool_call"), dict) else {}
        policy = step.get("policy_result") if isinstance(step.get("policy_result"), dict) else {}
        tool_name = str(tool_call.get("name") or "")
        payload_text = json.dumps(tool_call.get("input") or {}, ensure_ascii=False).lower()
        if policy.get("allowed") is False:
            return {"decision": "reject", "critique": "tool call rejected by policy", "safer_alternative": {"type": "inspect_log"}}
        if tool_name in {"install_environment", "start_service", "apply_repair", "resume_from_stage"} and not policy:
            return {"decision": "reject", "critique": "side-effect tool lacks explicit policy result", "safer_alternative": {"type": "inspect_log"}}
        if any(token in payload_text for token in ("api_key=", "token=", "password", "bearer ")):
            return {"decision": "reject", "critique": "tool input appears to contain secret value", "safer_alternative": {"type": "request_env_var_name_only"}}
        if tool_name == "verify_evidence" and "trace" not in payload_text:
            return {"decision": "revise", "critique": "verify tool should be tied to current trace evidence", "safer_alternative": {"type": "probe_http"}}
        return {"decision": "approve", "critique": "tool call is consistent with policy and evidence requirements", "safer_alternative": {}}
