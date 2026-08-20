from auto_harness.cli import _apply_cli_overrides, build_parser
from auto_harness.config import HarnessConfig


def test_retrieval_cli_overrides_are_bounded():
    args = build_parser().parse_args([
        "deploy", "--repo", ".", "--retrieval", "--retrieval-mode", "hybrid",
        "--retrieval-embedding-provider", "fake", "--retrieval-top-k", "99",
        "--retrieval-max-context-tokens", "99999",
    ])
    config = HarnessConfig()
    _apply_cli_overrides(config, args)
    assert config.retrieval["enabled"] is True
    assert config.retrieval["mode"] == "hybrid"
    assert config.retrieval["dense_enabled"] is True
    assert config.retrieval["embedding_provider"] == "fake"
    assert config.retrieval["default_top_k"] == 12
    assert config.retrieval["max_context_tokens"] == 32000
