"""The ``ai_models/`` directory: where every AI file the app uses lives.

Layout (created on first use)::

    ai_models/
      config.json   chat defaults (model, system prompt, sampling, accelerator options)
      llm/          GGUF language-model files (*.gguf) for the chat engine
      hasher/       trained HASHER split-models (*.json) from the Workload tab / CLI
      cache/        response cache + sealed chat transcripts

The root is ``<project>/ai_models`` by default; set ``AI_ASIC_MODELS`` to move it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "default_llm": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "system_prompt": "You are a helpful, concise assistant running on an AI-ASIC host.",
    # With speculative drafts, llama-cpp-python keeps logits for every context position
    # (n_ctx x vocab float32: ~1.2 GB at 2048 for Qwen's 151k vocab), so keep this modest.
    "n_ctx": 2048,
    "max_tokens": 512,
    "temperature": 0.7,
    "use_draft": True,
    # Prompt-lookup draft shape: longest n-gram matched against the context, and the most
    # tokens proposed per lookup. Longer drafts make each CPU verification pass costlier and
    # their tail is accepted less often, so tune these on your own prompts.
    "draft_max_ngram": 3,
    "draft_num_pred": 10,
    "use_cache": True,
    "seal_difficulty": 10,
    "seal_timeout": 30,  # seconds to wait for a real miner's share before sealing on the host
}

_README = """\
# ai_models

Every AI file AI-ASIC uses lives here.

| Folder | Contents |
|--------|----------|
| `llm/` | GGUF language models for the chat engine (`*.gguf`). Drop any GGUF here, or run `python -m ai_asic.cli models download qwen2.5-0.5b`. |
| `hasher/` | Trained HASHER split-models (`*.json`) saved from the Workload tab or `ai-asic workload --save`. |
| `cache/` | `responses.json` (SHA-256-addressed response cache) and `transcripts.jsonl` (chat turns sealed with mined proof-of-work). Safe to delete. |
| `config.json` | Chat defaults: model file, system prompt, context size, sampling, accelerator options. |

Model weights and caches are git-ignored. Set the `AI_ASIC_MODELS` environment variable to
keep this folder somewhere else.
"""


def models_root() -> Path:
    env = os.environ.get("AI_ASIC_MODELS", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "ai_models"


def ensure_layout(root: Optional[Path] = None) -> Path:
    root = Path(root) if root else models_root()
    for sub in ("llm", "hasher", "cache"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(_README, encoding="utf-8")
    cfg = root / "config.json"
    if not cfg.exists():
        cfg.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    return root


def llm_dir(root: Optional[Path] = None) -> Path:
    return ensure_layout(root) / "llm"


def hasher_dir(root: Optional[Path] = None) -> Path:
    return ensure_layout(root) / "hasher"


def cache_dir(root: Optional[Path] = None) -> Path:
    return ensure_layout(root) / "cache"


def list_llm_models(root: Optional[Path] = None) -> List[Path]:
    return sorted(llm_dir(root).glob("*.gguf"))


def load_config(root: Optional[Path] = None) -> Dict[str, Any]:
    path = ensure_layout(root) / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {**DEFAULT_CONFIG, **data}


def resolve_llm(name: Optional[str] = None, root: Optional[Path] = None) -> Optional[Path]:
    """Find a GGUF by explicit path, by file name in ``llm/``, by the configured default, or
    else the first GGUF present. Returns ``None`` when there is none."""
    if name:
        p = Path(name)
        if p.is_file():
            return p
        cand = llm_dir(root) / name
        if cand.is_file():
            return cand
        return None
    default = llm_dir(root) / load_config(root)["default_llm"]
    if default.is_file():
        return default
    found = list_llm_models(root)
    return found[0] if found else None
