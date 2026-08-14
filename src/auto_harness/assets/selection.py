from pathlib import Path


class ModelFileSelector:
    WEIGHT_SUFFIXES = (
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".gguf",
        ".onnx",
    )
    ESSENTIAL_NAMES = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "sentencepiece.bpe.model",
    }
    ESSENTIAL_SUFFIXES = (
        ".tiktoken",
    )

    def should_download(self, path: str) -> bool:
        lower = path.lower()
        name = Path(lower).name
        if lower.endswith(self.WEIGHT_SUFFIXES):
            return True
        if name in self.ESSENTIAL_NAMES:
            return True
        return lower.endswith(self.ESSENTIAL_SUFFIXES)

    # ------------------------------------------------------------------
    # Role classification for the deterministic file closure (Document A).
    # Non-breaking extension: the legacy downloader still uses should_download.
    # ------------------------------------------------------------------

    def role(self, path: str) -> str:
        """Classify a model repo file path into a stable role string."""
        name = Path(path).name
        lower = path.lower()
        if lower.endswith(".safetensors.index.json"):
            return "weight_index"
        if lower.endswith(".safetensors"):
            return "weight_shard"
        if lower.endswith(".bin") or lower.endswith((".pt", ".pth", ".ckpt")):
            return "legacy_weight"
        if lower.endswith(".gguf"):
            return "gguf_weight"
        if name == "config.json":
            return "config"
        if name == "generation_config.json":
            return "generation_config"
        if name == "tokenizer_config.json":
            return "tokenizer_config"
        if name == "tokenizer.json":
            return "tokenizer"
        if name == "tokenizer.model":
            return "tokenizer"
        if name == "special_tokens_map.json":
            return "special_tokens_map"
        if name == "added_tokens.json":
            return "added_tokens"
        if name == "vocab.json":
            return "vocab"
        if name == "merges.txt":
            return "merges"
        if name == "chat_template.jinja":
            return "chat_template"
        if lower.endswith(".tiktoken"):
            return "tiktoken"
        return "other"
