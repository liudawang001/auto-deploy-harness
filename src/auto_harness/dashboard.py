import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


class DashboardGenerator:
    """Builds a static dashboard from persisted task state and optional benchmark output."""

    STAGE_ORDER = [
        "analyze",
        "resource_plan",
        "env_solve",
        "env_deploy",
        "model_prepare",
        "runner",
        "verify",
        "report",
    ]

    def generate(self, runs_dir: Path, output_path: Path, benchmark_report: Optional[Path] = None) -> Dict:
        runs_dir = Path(runs_dir)
        output_path = Path(output_path)
        summary = self.build_summary(runs_dir, benchmark_report=benchmark_report)
        write_json(output_path.with_suffix(".json"), summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render_html(summary), encoding="utf-8")
        return {
            "status": "generated",
            "output_path": str(output_path),
            "summary_path": str(output_path.with_suffix(".json")),
            "task_count": summary["task_count"],
            "benchmark_status": summary["benchmark"].get("status", ""),
        }

    def build_summary(self, runs_dir: Path, benchmark_report: Optional[Path] = None) -> Dict:
        runs_dir = Path(runs_dir)
        tasks = self._tasks(runs_dir)
        benchmark = self._read_benchmark(benchmark_report)
        return {
            "generated_at": utc_now_iso(),
            "runs_dir": str(runs_dir),
            "task_count": len(tasks),
            "status_counts": self._status_counts(tasks),
            "stage_counts": self._stage_counts(tasks),
            "benchmark": self._benchmark_summary(benchmark),
            "tasks": tasks,
        }

    def render_html(self, summary: Dict) -> str:
        return self._html(summary)

    def _tasks(self, runs_dir: Path) -> List[Dict]:
        if not runs_dir.exists():
            return []
        tasks = []
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            state_path = run_dir / "state.json"
            task_path = run_dir / "task.json"
            if not state_path.exists():
                continue
            try:
                state = read_json(state_path)
                task = read_json(task_path) if task_path.exists() else {}
            except (OSError, ValueError):
                continue
            project = task.get("project") if isinstance(task.get("project"), dict) else {}
            stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
            tasks.append({
                "task_id": state.get("task_id") or run_dir.name,
                "name": project.get("name", ""),
                "repo": project.get("repo_url", ""),
                "status": state.get("status", ""),
                "current_stage": state.get("current_stage", ""),
                "last_safe_stage": state.get("last_safe_stage", ""),
                "report_path": state.get("report_path", ""),
                "updated_at": self._latest_stage_update(stages),
                "stage_statuses": {
                    stage: (stages.get(stage) or {}).get("status", "pending")
                    for stage in self.STAGE_ORDER
                },
            })
        return sorted(tasks, key=lambda item: item.get("updated_at") or "", reverse=True)

    def _read_benchmark(self, benchmark_report: Optional[Path]):
        if not benchmark_report:
            return {}
        path = Path(benchmark_report)
        if not path.exists():
            return {"status": "missing", "path": str(path)}
        try:
            data = read_json(path)
        except (OSError, ValueError):
            return {"status": "invalid", "path": str(path)}
        data["path"] = str(path)
        return data

    def _status_counts(self, tasks: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for task in tasks:
            status = task.get("status") or "unknown"
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _stage_counts(self, tasks: List[Dict]) -> Dict[str, Dict[str, int]]:
        counts: Dict[str, Dict[str, int]] = {stage: {} for stage in self.STAGE_ORDER}
        for task in tasks:
            for stage, status in task.get("stage_statuses", {}).items():
                bucket = counts.setdefault(stage, {})
                bucket[status] = bucket.get(status, 0) + 1
        return counts

    def _benchmark_summary(self, benchmark: Dict) -> Dict:
        if not benchmark:
            return {}
        cases = benchmark.get("cases") if isinstance(benchmark.get("cases"), list) else []
        failed = [case.get("id", "") for case in cases if case.get("status") != "passed"]
        return {
            "status": benchmark.get("status", ""),
            "path": benchmark.get("path", ""),
            "case_count": len(cases),
            "failed_case_ids": failed,
            "selected": bool(benchmark.get("selected")),
        }

    def _latest_stage_update(self, stages: Dict) -> str:
        values = []
        for stage in stages.values():
            if isinstance(stage, dict) and stage.get("updated_at"):
                values.append(stage["updated_at"])
        return max(values) if values else ""

    def _html(self, summary: Dict) -> str:
        status_cards = "\n".join(
            '<div class="metric"><strong>%s</strong><span>%s</span></div>' % (self._e(status), count)
            for status, count in sorted(summary.get("status_counts", {}).items())
        ) or '<div class="metric"><strong>none</strong><span>0</span></div>'
        benchmark = summary.get("benchmark") or {}
        benchmark_block = self._benchmark_html(benchmark)
        task_rows = "\n".join(self._task_row(task) for task in summary.get("tasks", []))
        if not task_rows:
            task_rows = '<tr><td colspan="7">No tasks found.</td></tr>'
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI-Auto-Harness Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2937; background: #f7f7f5; }
    header { padding: 24px 32px 16px; background: #ffffff; border-bottom: 1px solid #e5e7eb; }
    main { padding: 24px 32px 40px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    h2 { margin: 28px 0 12px; font-size: 18px; }
    .muted { color: #6b7280; font-size: 13px; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 16px; }
    .metric { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }
    .metric strong { display: block; font-size: 13px; color: #6b7280; }
    .metric span { display: block; margin-top: 6px; font-size: 24px; font-weight: 650; }
    table { width: 100%%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: top; }
    th { background: #f3f4f6; color: #374151; }
    tr:last-child td { border-bottom: 0; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .stage { display: inline-block; margin: 0 4px 4px 0; padding: 3px 6px; border-radius: 999px; background: #eef2ff; color: #3730a3; }
    .failed, .uncertain { background: #fff1f2; color: #9f1239; }
    .passed { background: #ecfdf5; color: #047857; }
    .pending { background: #f3f4f6; color: #4b5563; }
  </style>
</head>
<body>
  <header>
    <h1>AI-Auto-Harness Dashboard</h1>
    <div class="muted">Generated at %s from <code>%s</code></div>
    <div class="metrics">%s</div>
  </header>
  <main>
    %s
    <h2>Tasks</h2>
    <table>
      <thead><tr><th>Task</th><th>Project</th><th>Status</th><th>Current</th><th>Updated</th><th>Stages</th><th>Report</th></tr></thead>
      <tbody>%s</tbody>
    </table>
  </main>
</body>
</html>
""" % (
            self._e(summary.get("generated_at", "")),
            self._e(summary.get("runs_dir", "")),
            status_cards,
            benchmark_block,
            task_rows,
        )

    def _benchmark_html(self, benchmark: Dict) -> str:
        if not benchmark:
            return ""
        failed = benchmark.get("failed_case_ids") or []
        return """<h2>Benchmark</h2>
<div class="metrics">
  <div class="metric"><strong>Status</strong><span>%s</span></div>
  <div class="metric"><strong>Cases</strong><span>%s</span></div>
  <div class="metric"><strong>Failed</strong><span>%s</span></div>
</div>
<p class="muted">Source: <code>%s</code></p>
""" % (
            self._e(benchmark.get("status", "")),
            benchmark.get("case_count", 0),
            len(failed),
            self._e(benchmark.get("path", "")),
        )

    def _task_row(self, task: Dict) -> str:
        stages = " ".join(
            '<span class="stage %s">%s:%s</span>' % (self._css(status), self._e(stage), self._e(status))
            for stage, status in task.get("stage_statuses", {}).items()
        )
        report = task.get("report_path") or ""
        report_html = '<code>%s</code>' % self._e(report) if report else ""
        return "<tr><td><code>%s</code></td><td>%s<br><span class=\"muted\">%s</span></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            self._e(task.get("task_id", "")),
            self._e(task.get("name", "")),
            self._e(task.get("repo", "")),
            self._e(task.get("status", "")),
            self._e(task.get("current_stage", "")),
            self._e(task.get("updated_at", "")),
            stages,
            report_html,
        )

    def _css(self, value: str) -> str:
        value = value or "pending"
        if value in ("passed", "failed", "uncertain", "pending"):
            return value
        return ""

    def _e(self, value) -> str:
        return html.escape(str(value or ""), quote=True)


class DashboardServer:
    """Read-only HTTP dashboard server for local operations."""

    def __init__(self, generator: DashboardGenerator = None) -> None:
        self.generator = generator or DashboardGenerator()

    def create_server(self, runs_dir: Path, host: str = "127.0.0.1", port: int = 8765, benchmark_report: Optional[Path] = None):
        generator = self.generator
        runs_dir = Path(runs_dir)
        benchmark_report = Path(benchmark_report) if benchmark_report else None

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib handler API
                if self.path in ("/", "/index.html"):
                    summary = generator.build_summary(runs_dir, benchmark_report=benchmark_report)
                    body = generator.render_html(summary).encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", body)
                    return
                if self.path == "/dashboard.json":
                    summary = generator.build_summary(runs_dir, benchmark_report=benchmark_report)
                    body = json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", body)
                    return
                if self.path == "/healthz":
                    body = json.dumps({"status": "ok", "runs_dir": str(runs_dir)}, ensure_ascii=False).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", body)
                    return
                self._send(404, "application/json; charset=utf-8", b'{"status":"not_found"}')

            def log_message(self, format, *args):  # noqa: A002,N802 - stdlib handler API
                return

            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return ThreadingHTTPServer((host, port), Handler)
