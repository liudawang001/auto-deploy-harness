import os
import shlex
from pathlib import Path

from auto_harness.agents.base import AgentRequest, AgentResult
from auto_harness.utils.shell import run_command


class ClaudeCodeExecutor:
    def __init__(self, command_template: str = None) -> None:
        self.command_template = command_template or os.environ.get(
            "CLAUDE_CODE_CMD",
            "claude --print --output-format json",
        )

    def run(self, request: AgentRequest) -> AgentResult:
        prompt_path = request.workdir / ".auto_harness_prompt.md"
        prompt_path.write_text(request.prompt, encoding="utf-8")
        cmd = shlex.split(self.command_template) + [prompt_path.read_text(encoding="utf-8")]
        result = run_command(cmd, request.workdir, timeout_seconds=request.timeout_seconds)
        if result.exit_code != 0:
            return AgentResult(status="failed", text=result.stdout, error=result.stderr)
        return AgentResult(status="passed", text=result.stdout, raw={"stderr": result.stderr})

    def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        request.prompt = "Resume session %s.\n\n%s" % (session_id, request.prompt)
        return self.run(request)

