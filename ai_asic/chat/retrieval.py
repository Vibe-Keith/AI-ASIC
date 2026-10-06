"""Retrieval drafts: past replies, indexed by the ASIC bucket IDs of the prompts behind them.

Prompt-lookup drafting can only copy from the current context. When a new prompt resembles an
earlier one - in this chat, an earlier chat, or an earlier session - the earlier reply is a good
source of draft continuations too (retrieval-based speculative decoding). This store keeps
recent replies keyed by the LSH buckets of their prompts; at the start of a turn the prompt's
buckets (one batched nonce search, usually cache hits) pull the most similar past replies, which
the draft index then uses as extra candidate token sequences.

Persisted as JSONL (``ai_models/cache/draft_store.jsonl``) and capped at ``max_entries``.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ai_asic.chat.lsh import Buckets, similarity


@dataclass
class _Entry:
    sig: int
    buckets: Buckets
    text: str


class DraftStore:
    def __init__(self, path: Optional[Path] = None, max_entries: int = 500):
        self.path = Path(path) if path else None
        self.max_entries = max_entries
        self._entries: List[Optional[_Entry]] = []
        self._index: Dict[Tuple[int, int], List[int]] = {}
        self._lock = threading.Lock()
        if self.path is not None:
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    try:
                        rec = json.loads(line)
                        self._insert(_Entry(int(rec["sig"]), list(rec["buckets"]),
                                            str(rec["text"])))
                    except (ValueError, KeyError, TypeError):
                        continue
            except OSError:
                pass
            self._entries = self._entries[-max_entries:]
            self._reindex()

    def __len__(self) -> int:
        return sum(1 for e in self._entries if e is not None)

    def _insert(self, entry: _Entry) -> None:
        idx = len(self._entries)
        self._entries.append(entry)
        for band, b in enumerate(entry.buckets):
            if b is not None:
                self._index.setdefault((band, b), []).append(idx)

    def _reindex(self) -> None:
        entries = [e for e in self._entries if e is not None]
        self._entries, self._index = [], {}
        for e in entries:
            self._insert(e)

    def add(self, sig: int, buckets: Buckets, text: str) -> None:
        text = text.strip()
        if not text or not any(b is not None for b in buckets):
            return
        entry = _Entry(sig, list(buckets), text)
        with self._lock:
            self._insert(entry)
            rewrite = len(self._entries) > self.max_entries
            if rewrite:
                self._entries = self._entries[-self.max_entries:]
                self._reindex()
            if self.path is not None:
                try:
                    if rewrite:
                        self.path.write_text("".join(
                            json.dumps({"sig": e.sig, "buckets": e.buckets, "text": e.text}) + "\n"
                            for e in self._entries if e is not None), encoding="utf-8")
                    else:
                        with open(self.path, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"sig": sig, "buckets": entry.buckets,
                                                 "text": text}) + "\n")
                except OSError:
                    pass

    def lookup(self, sig: int, buckets: Sequence[Optional[int]], k: int = 2,
               min_similarity: float = 0.65) -> List[Tuple[float, str]]:
        """Up to ``k`` past replies whose prompt shares an LSH bucket with this one and whose
        full signature similarity is at least ``min_similarity``, best first."""
        with self._lock:
            cand = set()
            for band, b in enumerate(buckets):
                if b is not None:
                    cand.update(self._index.get((band, b), ()))
            scored = sorted(((similarity(sig, self._entries[i].sig), -i) for i in cand
                             if self._entries[i] is not None), reverse=True)
            out: List[Tuple[float, str]] = []
            seen = set()
            for s, neg_i in scored:
                if s < min_similarity or len(out) >= k:
                    break
                text = self._entries[-neg_i].text
                if text not in seen:
                    seen.add(text)
                    out.append((s, text))
            return out
