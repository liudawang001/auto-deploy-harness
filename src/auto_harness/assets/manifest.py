from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelAsset:
    asset_id: str
    source: str
    repo_id: str = ""
    revision: str = "main"
    origin: str = ""
    expected_size_bytes: Optional[int] = None
    cache_key: str = ""
    cache_path: str = ""
    status: str = "planned"
    resume_supported: bool = True
    files: List[Dict] = field(default_factory=list)
    downloaded_bytes: int = 0
    last_error: Optional[str] = None


@dataclass
class AssetManifest:
    assets: List[ModelAsset] = field(default_factory=list)
    total_expected_size_bytes: Optional[int] = None
    cache_root: str = ""
    status: str = "planned"
