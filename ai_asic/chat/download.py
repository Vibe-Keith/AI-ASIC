"""Download known GGUF models into ``ai_models/llm/`` with SHA-256 verification.

Stdlib only (``urllib``). The file streams to ``<name>.part`` while being hashed, and is renamed
into place only if both the byte count and the SHA-256 match the published values, so an
interrupted or tampered download never appears as a usable model.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from ai_asic.chat.models_dir import llm_dir


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    size: int
    sha256: str
    description: str


KNOWN_MODELS: Dict[str, ModelSpec] = {
    "qwen2.5-0.5b": ModelSpec(
        key="qwen2.5-0.5b",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        url=("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
             "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        size=491400032,
        sha256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        description="Qwen2.5 0.5B Instruct, Q4_K_M (469 MB) - small, fast CPU chat model",
    ),
}


class DownloadError(RuntimeError):
    pass


def download(spec: ModelSpec, dest_dir: Optional[Path] = None,
             progress: Optional[Callable[[int, int], None]] = None,
             chunk_size: int = 1 << 20) -> Path:
    dest_dir = Path(dest_dir) if dest_dir else llm_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / spec.filename
    if final.is_file() and final.stat().st_size == spec.size:
        return final
    part = dest_dir / (spec.filename + ".part")
    h = hashlib.sha256()
    done = 0
    req = urllib.request.Request(spec.url, headers={"User-Agent": "ai-asic/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, spec.size)
    except OSError as exc:
        raise DownloadError(f"download failed: {exc}") from exc
    if done != spec.size:
        part.unlink(missing_ok=True)
        raise DownloadError(f"size mismatch: got {done} bytes, expected {spec.size}")
    if h.hexdigest() != spec.sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(f"SHA-256 mismatch: got {h.hexdigest()}, expected {spec.sha256}")
    os.replace(part, final)
    return final
