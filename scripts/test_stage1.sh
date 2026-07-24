#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q \
  tests/test_default_agent_controller.py \
  tests/test_controller_contract.py \
  tests/test_graph_checkpoint_resume.py \
  tests/test_langgraph_recovery_integration.py \
  tests/test_langgraph_approval.py \
  tests/test_memory_evolution_e2e.py
