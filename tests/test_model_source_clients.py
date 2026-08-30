"""Phase A3 tests: source clients, immutable revision pinning, and TOCTOU."""
import json
import pytest

from auto_harness.model_runtime.resolver import ModelReferenceResolver
from auto_harness.model_runtime.schemas import ModelReferenceCandidate
from auto_harness.model_runtime.source_clients import (
    HuggingFaceSourceClient,
    ModelScopeSourceClient,
    SourceClientError,
    TransportResponse,
    source_client_for,
    source_metadata_hash,
)

SHA = "a" * 40
SHA2 = "b" * 40

HF_CONFIG = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "max_position_embeddings": 32768,
    "torch_dtype": "float16",
}

HF_TREE = [
    {"type": "file", "path": "config.json", "size": 100, "sha256": "x"},
    {"type": "file", "path": "model.safetensors.index.json", "size": 200, "sha256": "y"},
    {"type": "file", "path": "model-00001-of-00002.safetensors", "size": 4000, "sha256": "z"},
]


class FakeTransport:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def request(self, url, method="GET", headers=None, timeout=None):
        self.requests.append((url, dict(headers or {})))
        if url in self.routes:
            response = self.routes[url]
            if isinstance(response, Exception):
                raise response
            return response
        return TransportResponse(404, b"{}", url=url)


def _hf_transport(sha=SHA, config=None, tree=None, revision_extra=None):
    config = config if config is not None else HF_CONFIG
    tree = tree if tree is not None else HF_TREE
    extra = revision_extra or {}
    routes = {
        "https://huggingface.co/api/models/org/model/revision/main": TransportResponse(
            200, json.dumps({"sha": sha, **extra}).encode(), url="https://huggingface.co/api/models/org/model/revision/main"
        ),
        "https://huggingface.co/org/model/resolve/%s/config.json" % sha: TransportResponse(
            200, json.dumps(config).encode(), url="https://huggingface.co/org/model/resolve/%s/config.json" % sha
        ),
        "https://huggingface.co/api/models/org/model/tree/%s?recursive=true" % sha: TransportResponse(
            200, json.dumps(tree).encode(), url="https://huggingface.co/api/models/org/model/tree/%s?recursive=true" % sha
        ),
    }
    return FakeTransport(routes)


class TestResolveModel:
    def test_resolve_pins_commit(self):
        transport = _hf_transport()
        client = HuggingFaceSourceClient(transport=transport)
        candidate = ModelReferenceCandidate(source="huggingface", repo_id="org/model", requested_revision="main")
        spec = ModelReferenceResolver().resolve_model(candidate, client)
        assert spec.status == "resolved"
        assert spec.resolved_revision == SHA
        assert spec.model_identity == "huggingface:org/model@%s" % SHA
        assert spec.model_type == "qwen2"
        assert spec.architectures == ["Qwen2ForCausalLM"]
        assert spec.dtype == "float16"
        assert spec.requires_remote_code is False
        # config and tree requests used the frozen SHA, not the mutable ref
        urls = [url for url, _ in transport.requests]
        assert any(SHA in url for url in urls)
        assert not any("/resolve/main/" in url for url in urls)

    def test_remote_code_detected(self):
        config = dict(HF_CONFIG, auto_map={"AutoModel": "modeling_custom.py"})
        client = HuggingFaceSourceClient(transport=_hf_transport(config=config))
        candidate = ModelReferenceCandidate(source="huggingface", repo_id="org/model")
        spec = ModelReferenceResolver().resolve_model(candidate, client)
        assert spec.status == "remote_code_required"

    def test_quantization_detected(self):
        config = dict(HF_CONFIG, quantization_config={"quant_method": "gptq"})
        client = HuggingFaceSourceClient(transport=_hf_transport(config=config))
        candidate = ModelReferenceCandidate(source="huggingface", repo_id="org/model")
        spec = ModelReferenceResolver().resolve_model(candidate, client)
        assert spec.status == "unsupported_quantization"

    def test_not_found(self):
        transport = FakeTransport({})  # 404 default
        client = HuggingFaceSourceClient(transport=transport)
        candidate = ModelReferenceCandidate(source="huggingface", repo_id="org/model")
        spec = ModelReferenceResolver().resolve_model(candidate, client)
        assert spec.status == "not_found"

    def test_access_required(self):
        transport = FakeTransport({
            "https://huggingface.co/api/models/org/model/revision/main": TransportResponse(
                401, b'{"error": "unauthorized"}', url="https://huggingface.co/api/models/org/model/revision/main"
            ),
        })
        client = HuggingFaceSourceClient(transport=transport)
        candidate = ModelReferenceCandidate(source="huggingface", repo_id="org/model")
        spec = ModelReferenceResolver().resolve_model(candidate, client)
        assert spec.status == "access_required"


class TestSourceClient:
    def test_source_client_factory(self):
        assert isinstance(source_client_for("huggingface", transport=FakeTransport({})), HuggingFaceSourceClient)
        assert isinstance(source_client_for("modelscope", transport=FakeTransport({})), ModelScopeSourceClient)
        with pytest.raises(ValueError):
            source_client_for("s3")

    def test_auth_summary_no_token(self):
        client = HuggingFaceSourceClient(transport=FakeTransport({}), token=None)
        summary = client.auth_summary()
        assert summary == {"auth_env_name": "HF_TOKEN", "auth_present": False}

    def test_redirect_to_disallowed_host_rejected(self):
        transport = FakeTransport({
            "https://huggingface.co/api/models/org/model/revision/main": TransportResponse(
                200, json.dumps({"sha": SHA}).encode(), url="https://evil.example.com/revision/main"
            ),
        })
        client = HuggingFaceSourceClient(transport=transport)
        with pytest.raises(SourceClientError) as exc:
            client.resolve_revision("org/model", "main")
        assert exc.value.status == "network_failed"

    def test_explicit_mirror_endpoint_allowed(self):
        mirror = "https://hf-mirror.com"
        transport = FakeTransport({
            "%s/api/models/org/model/revision/main" % mirror: TransportResponse(
                200, json.dumps({"sha": SHA}).encode(), url="%s/api/models/org/model/revision/main" % mirror
            ),
        })
        client = HuggingFaceSourceClient(transport=transport, api_base=mirror)
        assert client.resolve_revision("org/model", "main")["resolved_revision"] == SHA

    def test_mirror_endpoint_does_not_allow_third_party_hosts(self):
        mirror = "https://hf-mirror.com"
        transport = FakeTransport({
            "%s/api/models/org/model/revision/main" % mirror: TransportResponse(
                200, json.dumps({"sha": SHA}).encode(), url="https://evil.example.com/revision/main"
            ),
        })
        client = HuggingFaceSourceClient(transport=transport, api_base=mirror)
        with pytest.raises(SourceClientError) as exc:
            client.resolve_revision("org/model", "main")
        assert exc.value.status == "network_failed"

    def test_source_client_factory_passes_api_base(self):
        client = source_client_for("huggingface", transport=FakeTransport({}), api_base="https://hf-mirror.com")
        assert client.api_base == "https://hf-mirror.com"
        client = source_client_for("huggingface", transport=FakeTransport({}))
        assert client.api_base == "https://huggingface.co"

    def test_lfs_oid_maps_to_content_sha256(self):
        tree = [
            {"type": "file", "path": "config.json", "size": 100, "oid": "a6344aac8c09"},
            {
                "type": "file", "path": "model.safetensors", "size": 4000,
                "oid": "9127f71e7314",
                "lfs": {"oid": "d" * 64, "size": 4000, "pointerSize": 135},
            },
        ]
        routes = {
            "https://huggingface.co/api/models/org/model/revision/main": TransportResponse(
                200, json.dumps({"sha": SHA}).encode(),
                url="https://huggingface.co/api/models/org/model/revision/main",
            ),
            "https://huggingface.co/org/model/resolve/%s/config.json" % SHA: TransportResponse(
                200, json.dumps(HF_CONFIG).encode(),
                url="https://huggingface.co/org/model/resolve/%s/config.json" % SHA,
            ),
            "https://huggingface.co/api/models/org/model/tree/%s?recursive=true" % SHA: TransportResponse(
                200, json.dumps(tree).encode(),
                url="https://huggingface.co/api/models/org/model/tree/%s?recursive=true" % SHA,
            ),
        }
        client = HuggingFaceSourceClient(transport=FakeTransport(routes))
        files = client.list_files("org/model", SHA)
        by_path = {f["path"]: f for f in files}
        # LFS oid is the content SHA-256; a plain git oid is a SHA-1 and is
        # only used as the etag.
        assert by_path["model.safetensors"]["sha256"] == "d" * 64
        assert by_path["model.safetensors"]["etag"] == "9127f71e7314"
        assert by_path["config.json"]["sha256"] is None

    def test_invalid_json_metadata_invalid(self):
        transport = FakeTransport({
            "https://huggingface.co/api/models/org/model/revision/main": TransportResponse(
                200, b"not-json", url="https://huggingface.co/api/models/org/model/revision/main"
            ),
        })
        client = HuggingFaceSourceClient(transport=transport)
        with pytest.raises(SourceClientError) as exc:
            client.resolve_revision("org/model", "main")
        assert exc.value.status == "metadata_invalid"

    def test_mutable_sha_rejected(self):
        transport = FakeTransport({
            "https://huggingface.co/api/models/org/model/revision/main": TransportResponse(
                200, json.dumps({"sha": "not-a-commit"}).encode(), url="https://huggingface.co/api/models/org/model/revision/main"
            ),
        })
        client = HuggingFaceSourceClient(transport=transport)
        with pytest.raises(SourceClientError) as exc:
            client.resolve_revision("org/model", "main")
        assert exc.value.status == "metadata_invalid"

    def test_5xx_retries_then_network_failed(self):
        transport = FakeTransport({
            "https://huggingface.co/api/models/org/model/revision/main": TransportResponse(
                500, b"boom", url="https://huggingface.co/api/models/org/model/revision/main"
            ),
        })
        client = HuggingFaceSourceClient(transport=transport, max_attempts=2)
        with pytest.raises(SourceClientError) as exc:
            client.resolve_revision("org/model", "main")
        assert exc.value.status == "network_failed"


class TestSourceMetadataHash:
    def test_stable(self):
        a = source_metadata_hash({"a": 1}, [{"path": "x", "size_bytes": 1}], SHA)
        b = source_metadata_hash({"a": 1}, [{"path": "x", "size_bytes": 1}], SHA)
        assert a == b

    def test_changes_with_commit(self):
        a = source_metadata_hash({"a": 1}, [], SHA)
        b = source_metadata_hash({"a": 1}, [], SHA2)
        assert a != b
