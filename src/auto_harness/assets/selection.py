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
