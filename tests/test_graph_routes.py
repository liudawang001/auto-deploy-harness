"""Graph routes tests: verify all route functions are pure and correct.

Routes must be pure functions: no LLM, no shell, no file I/O, no state mutation.
"""
import pytest

from auto_harness.graph.routes import (
    route_after_parse,
    route_after_policy,
    route_after_stage,
    route_after_verify,
    route_after_replan,
    route_resume_stage,
)


class TestRouteAfterParse:
    def test_valid_plan_goes_to_policy(self):
        state = {"stop_reason": ""}
        assert route_after_parse(state) == "valid"

    def test_invalid_plan_goes_to_stop(self):
        state = {"stop_reason": "plan_parse_failed"}
        assert route_after_parse(state) == "invalid"

    def test_not_ok_status_goes_to_stop(self):
        state = {"stop_reason": "plan_not_ok"}
        assert route_after_parse(state) == "invalid"


class TestRouteAfterPolicy:
    def test_allowed_goes_to_compile(self):
        state = {"stop_reason": ""}
        assert route_after_policy(state) == "compile"

    def test_rejected_goes_to_stop(self):
        state = {"stop_reason": "policy_rejected"}
        assert route_after_policy(state) == "stop"


class TestRouteAfterStage:
    def test_passed_goes_to_continue(self):
        state = {
            "current_stage": "env_deploy",
            "stage_results": {"env_deploy": {"status": "passed"}},
            "replan_count": 0,
            "max_replans": 2,
        }
        assert route_after_stage(state) == "continue"

    def test_failed_goes_to_observe_failure(self):
        """Failed stages route to observe_failure for diagnosis."""
        state = {
            "current_stage": "runner",
            "stage_results": {"runner": {"status": "failed"}},
            "replan_count": 0,
            "max_replans": 2,
        }
        assert route_after_stage(state) == "observe_failure"

    def test_uncertain_goes_to_observe_failure(self):
        """Uncertain stages route to observe_failure for diagnosis."""
        state = {
            "current_stage": "env_deploy",
            "stage_results": {"env_deploy": {"status": "uncertain"}},
            "replan_count": 0,
            "max_replans": 1,
        }
        assert route_after_stage(state) == "observe_failure"

    def test_default_status_is_failed(self):
        """If stage result has no status, treat as failed -> observe_failure."""
        state = {
            "current_stage": "runner",
            "stage_results": {"runner": {}},
            "replan_count": 0,
            "max_replans": 2,
        }
        assert route_after_stage(state) == "observe_failure"


class TestRouteAfterVerify:
    def test_passed_goes_to_report(self):
        state = {"verify_status": "passed", "replan_count": 0, "max_replans": 2}
        assert route_after_verify(state) == "report"

    def test_pass_goes_to_report(self):
        state = {"verify_status": "pass", "replan_count": 0, "max_replans": 2}
        assert route_after_verify(state) == "report"

    def test_failed_goes_to_observe_failure(self):
        """Failed verify routes to observe_failure for diagnosis."""
        state = {"verify_status": "failed", "replan_count": 0, "max_replans": 2}
        assert route_after_verify(state) == "observe_failure"

    def test_uncertain_goes_to_observe_failure(self):
        """Uncertain verify routes to observe_failure for diagnosis."""
        state = {"verify_status": "uncertain", "replan_count": 0, "max_replans": 1}
        assert route_after_verify(state) == "observe_failure"


class TestRouteResumeStage:
    def test_analyze_is_allowed(self):
        state = {"resume_from_stage": "analyze"}
        assert route_resume_stage(state) == "analyze"

    def test_verify_is_allowed(self):
        state = {"resume_from_stage": "verify"}
        assert route_resume_stage(state) == "verify"

    def test_unknown_falls_back_to_analyze(self):
        state = {"resume_from_stage": "arbitrary_node"}
        assert route_resume_stage(state) == "analyze"

    def test_missing_falls_back_to_analyze(self):
        state = {}
        assert route_resume_stage(state) == "analyze"

    def test_all_pipeline_stages_allowed(self):
        for stage in ("analyze", "resource_plan", "env_solve", "env_deploy",
                       "model_prepare", "runner", "verify"):
            state = {"resume_from_stage": stage}
            assert route_resume_stage(state) == stage


class TestRouteAfterReplan:
    def test_new_plan_goes_to_parse(self):
        state = {"raw_plan_path": "/some/path", "replan_count": 1, "max_replans": 2}
        assert route_after_replan(state) == "parse"

    def test_exhausted_goes_to_stop(self):
        state = {"raw_plan_path": "/some/path", "replan_count": 3, "max_replans": 2}
        assert route_after_replan(state) == "stop"

    def test_no_raw_plan_goes_to_stop(self):
        state = {"raw_plan_path": "", "replan_count": 1, "max_replans": 2}
        assert route_after_replan(state) == "stop"

    def test_missing_raw_plan_goes_to_stop(self):
        state = {"replan_count": 1, "max_replans": 2}
        assert route_after_replan(state) == "stop"
