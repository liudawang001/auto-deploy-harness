---
name: verify-evidence
description: Verify automatic AI deployment with hard evidence. Use during verify stages for web UI/API services, especially Gradio/FastAPI/Flask/Streamlit projects, to send traceable requests, check response/body/artifact freshness, diagnose uncertainty, and prevent false positive HTTP 200 success.
---

# Verify Evidence

Goal: prove the deployed service handled this run's fresh trace, or explicitly return `uncertain`.

## Verification policy

1. Generate a unique trace id for every verify attempt.
2. Prefer service-specific API calls over only checking a browser page:
   - Gradio: POST `/api/predict` or discovered API endpoint with `{"data":["{{trace_id}}"]}`.
   - FastAPI/Flask: GET or POST an endpoint that should echo or process trace input.
   - Streamlit: HTTP readiness alone is weak; require artifact/log/DOM evidence if no API exists.
3. A live port or HTTP 200 is readiness evidence, not success evidence.
4. Pass only when at least one strong proof exists:
   - response contains current trace id;
   - a fresh output artifact was created after trace execution;
   - a framework-specific event/log proves current trace processing.
5. Store request, response tail, status code, body template, and evidence path.

## Diagnosis categories

- `service_unreachable`: no endpoint or port not ready.
- `api_shape_unknown`: service exists but callable API is unknown.
- `trace_not_observed`: request succeeded but response/artifact did not prove trace handling.
- `artifact_missing`: expected output file was not created or not fresh.
- `dry_run_missing_evidence`: execution was intentionally skipped.

## Repair guidance

When verify is uncertain, the next action is to inspect API shape and logs, then update `verify_hint` or add a framework-specific verify skill. Do not lower verification standards to make the pipeline green.
