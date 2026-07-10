# HTTP Trace Echo E2E Fixture

Minimal standard-library Python demo for deployment E2E verification.

The service exposes a GET endpoint that echoes the current `_auto_harness_trace`
query parameter. auto-deploy-harness must start the service and prove that verify
observed the same trace id in the response.
