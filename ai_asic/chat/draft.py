"""Speculative-decoding drafts from a SHA-256-addressed n-gram index on the ASIC.

This is prompt-lookup decoding: when the last few generated tokens already appeared earlier
in the context, the tokens that followed them are a cheap guess for what comes next. llama.cpp
verifies the whole guess in one batched forward pass and keeps the prefix it agrees with, so
correct guesses yield several tokens per pass and wrong ones cost little.

The index keys every n-gram window by its SHA-256 digest, computed in batches on the hash
accelerator. That moves the lookup's hashing onto the ASIC; the speedup itself comes from
speculative decoding (largest when a reply repeats or quotes its context), not from SHA-256.

llama.cpp calls the draft model with the full token sequence each step and rewinds the
sequence when it rejects a draft, so the index updates incrementally from the common prefix and
drops windows past any rewind point.

Acceptance is measured from those same calls: each one passes the previous sequence plus the
draft tokens llama.cpp kept plus one newly sampled token, so the common prefix of the new tail
with the previous draft is exactly how many draft tokens were accepted.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Sequence

from ai_asic.chat.accelerator import HashAccelerator


class HashDraftIndex:
    def __init__(self, accelerator: HashAccelerator, max_ngram: int = 3, num_pred: int = 10):
        self.acc = accelerator
        self.max_ngram = max_ngram
        self.num_pred = num_pred
        self.tokens: List[int] = []
        self._index: Dict[bytes, List[int]] = {}   # digest -> end positions (exclusive)
        self._by_end: Dict[int, List[bytes]] = {}  # end -> digests, ordered by n = 1..max
        self.calls = 0
        self.proposed = 0
        self.scored = 0     # draft tokens whose fate llama.cpp has revealed
        self.accepted = 0   # ... of which it kept
        self._pending: List[int] = []

    def reset(self) -> None:
        self.tokens = []
        self._index.clear()
        self._by_end.clear()
        self._pending = []

    def discard_pending(self) -> None:
        """Forget the last draft unscored; call when a generation ends, since the next call
        will start a new generation rather than reveal that draft's fate."""
        self._pending = []

    def __call__(self, input_ids: Sequence[int]) -> List[int]:
        seq = [int(t) for t in input_ids]
        n = len(self.tokens)
        pending, self._pending = self._pending, []
        if seq[:n] == self.tokens:
            common = n
            if pending and len(seq) > n:
                k = 0
                while k < len(pending) and n + k < len(seq) and seq[n + k] == pending[k]:
                    k += 1
                self.accepted += k
                self.scored += len(pending)
        else:
            common = 0
            limit = min(n, len(seq))
            while common < limit and seq[common] == self.tokens[common]:
                common += 1
            self._truncate(common)
        if len(seq) > common:
            self._extend(seq[common:])
        self.calls += 1
        draft = self._lookup()
        self.proposed += len(draft)
        self._pending = draft
        return draft

    def _truncate(self, length: int) -> None:
        for end in range(len(self.tokens), length, -1):
            for digest in self._by_end.pop(end, []):
                ends = self._index.get(digest)
                if ends:
                    ends.remove(end)
                    if not ends:
                        del self._index[digest]
        del self.tokens[length:]

    def _extend(self, new_tokens: List[int]) -> None:
        start = len(self.tokens)
        self.tokens.extend(new_tokens)
        windows: List[bytes] = []
        ends: List[int] = []
        for end in range(start + 1, len(self.tokens) + 1):
            for size in range(1, self.max_ngram + 1):
                if end - size < 0:
                    break
                windows.append(struct.pack(f"<{size}i", *self.tokens[end - size:end]))
                ends.append(end)
        if not windows:
            return
        for end, digest in zip(ends, self.acc.hash_batch(windows, stage="draft")):
            self._index.setdefault(digest, []).append(end)
            self._by_end.setdefault(end, []).append(digest)

    def _lookup(self) -> List[int]:
        total = len(self.tokens)
        suffix = self._by_end.get(total, [])
        for size in range(min(self.max_ngram, total - 1, len(suffix)), 0, -1):
            for end in reversed(self._index.get(suffix[size - 1], [])):
                if end < total:  # most recent earlier occurrence
                    cont = self.tokens[end:end + self.num_pred]
                    if cont:
                        return cont
        return []


def make_llama_draft_model(index: HashDraftIndex):
    """Wrap ``index`` as a llama-cpp-python ``LlamaDraftModel`` (imports llama_cpp lazily)."""
    import numpy as np
    from llama_cpp.llama_speculative import LlamaDraftModel

    class _HashDraftModel(LlamaDraftModel):
        def __call__(self, input_ids, /, **kwargs):
            return np.array(index(input_ids.tolist()), dtype=np.intc)

    return _HashDraftModel()
