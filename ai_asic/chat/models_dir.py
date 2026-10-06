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
    # llama.cpp CPU settings (null = llama-cpp-python's default). n_threads: decode threads
    # (default: half the logical cores). On a 4-vCPU VM, Qwen2.5-0.5B-shaped Q4_K_M decoded at
    # 21.6 tok/s with the default 2 threads and 34.6 with 4; on CPUs with SMT/hyperthreading the
    # physical core count is usually best - try both (chat --threads N). n_threads_batch: prompt
    # threads (default: all logical cores). flash_attn: off by default, it cut decode to 17 tok/s
    # on that VM. kv_cache_type "q8_0" halves KV-cache memory on long chats (needs flash_attn).
    "n_threads": None,
    "n_threads_batch": None,
    "n_batch": 512,
    "n_ubatch": 512,
    "flash_attn": None,
    "kv_cache_type": "f16",
    "max_tokens": 512,
    "temperature": 0.7,
    "use_draft": True,
    # Speculative drafts (see ai_asic/chat/draft.py). Shape: longest context n-gram matched and
    # the most tokens proposed per verification pass. "consensus" pools every earlier occurrence
    # (and retrieved past replies) and stops where they disagree; "lookup" is the old
    # single-occurrence behaviour. draft_len_by_match caps drafts backed by 1- and 2-token
    # matches, where most rejected (wasted) draft tokens come from.
    "draft_max_ngram": 4,
    "draft_num_pred": 16,
    "draft_mode": "consensus",
    "draft_len_by_match": [3, 8],
    "draft_max_candidates": 8,
    "draft_adaptive": False,
    # Retrieval drafts: past replies to similar prompts (matched by ASIC LSH buckets) become
    # extra draft candidates. Stored in cache/draft_store.jsonl.
    "draft_retrieval": True,
    "draft_retrieval_k": 2,
    "draft_retrieval_min_sim": 0.65,
    # KV-block routing (ai_asic/chat/routing.py): once the prompt would exceed
    # route_budget_tokens, keep the system prompt, the newest exchanges and the older exchanges
    # whose ASIC LSH buckets match the new message, so the CPU attends over less context.
    "route_context": True,
    "route_budget_tokens": 768,
    "route_keep_recent": 2,
    "route_low_water": 0.6,       # a compaction keeps this fraction of the budget
    "route_recall_sim": 0.8,      # re-admit a dropped exchange this similar to the new message
    # LSH bucket IDs by native BM1387 nonce search (ai_asic/chat/lsh.py).
    "lsh_bands": 10,
    "lsh_band_bits": 6,
    "bucket_difficulty": 6,
    "bucket_max_nonces": 4096,
    # Search the whole bucket table (lsh_bands x 2**lsh_band_bits headers) on the ASIC in the
    # background at startup, so no reply waits on the device for a bucket ID.
    "lsh_prefetch_table": True,
    "use_cache": True,
    "seal_difficulty": 10,
    "seal_timeout": 30,  # seconds to wait for a real miner's share before sealing on the host
    "seal_async": True,  # seal in the background; the reply returns without waiting for it
}

_README = """\
# ai_models

Every AI file AI-ASIC uses lives here.

| Folder | Contents |
|--------|----------|
| `llm/` | GGUF language models for the chat engine (`*.gguf`). Drop any GGUF here, or run `python -m ai_asic.cli models download qwen2.5-0.5b`. |
| `hasher/` | Trained HASHER split-models (`*.json`) saved from the Workload tab or `ai-asic workload --save`. |
| `cache/` | `responses.jsonl` (SHA-256-addressed response cache, append-only), `transcripts.jsonl` (chat turns sealed with mined proof-of-work) and `draft_store.jsonl` (past replies used as speculative-draft candidates). Safe to delete. |
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
