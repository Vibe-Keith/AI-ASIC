"""Speculative-decoding drafts from a SHA-256-addressed n-gram index.

This is prompt-lookup decoding, extended. When the last few tokens already appeared earlier,
the tokens that followed them are a cheap guess for what comes next; llama.cpp verifies the
whole guess in one batched forward pass and keeps the prefix it agrees with, so good guesses
yield several tokens per pass and wrong ones cost little.

Every n-gram window (n = 1..``max_ngram``) is keyed by its SHA-256 digest - a compact context
hash - computed in batches on the host with ``hashlib`` (a mining ASIC cannot hash arbitrary
data, and keeping it local avoids a round trip per token).

What makes it propose better drafts than single-occurrence lookup (``mode="lookup"``):

  * **Candidate batches** (``mode="consensus"``): every earlier occurrence of the matched suffix
    contributes its continuation, up to ``max_candidates`` - from the live context and from
    *reference sequences* (past replies retrieved for similar prompts, see ``retrieval.py``).
    Longer matches weigh more (n squared); references weigh half.
  * **Consensus with early stop**: the draft follows the weighted majority token by token and
    stops where the surviving candidates disagree, so the tail that would likely be rejected is
    never sent for verification.
  * **Length by match strength** (``len_by_match``): a draft backed only by a 1-token match
    is capped at 3 tokens, a 2-token match at 8, longer matches at ``num_pred``. Single-token
    matches ("the", "and") are where most rejected draft tokens - wasted verification and
    attention work - came from in testing.
  * **Adaptive length** (optional, off by default): after a fully accepted draft the next may
    be 2 tokens longer; after a rejection it shrinks to what was accepted plus 2. It did not
    raise accepted tokens per pass in testing, so it is not the default.

llama.cpp calls the draft model with the full token sequence each step and rewinds the sequence
when it rejects a draft, so the live index updates incrementally from the common prefix and
drops windows past any rewind point.

Acceptance is measured from those same calls: each one passes the previous sequence plus the
draft tokens llama.cpp kept plus one newly sampled token, so the common prefix of the new tail
with the previous draft is exactly how many draft tokens were accepted.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Sequence, Tuple

from ai_asic.chat.accelerator import HashAccelerator

MODES = ("lookup", "consensus")


class HashDraftIndex:
    def __init__(self, accelerator: HashAccelerator, max_ngram: int = 3, num_pred: int = 10,
                 mode: str = "consensus", adaptive: bool = False, max_candidates: int = 8,
                 min_pred: int = 2, min_agree: float = 0.5,
                 len_by_match: Sequence[int] = (3, 8)):
        if mode not in MODES:
            raise ValueError(f"draft mode must be one of {MODES}, not {mode!r}")
        self.acc = accelerator
        self.max_ngram = max_ngram
        self.num_pred = num_pred
        self.mode = mode
        self.adaptive = adaptive
        self.max_candidates = max(1, max_candidates)
        self.min_pred = max(1, min(min_pred, num_pred))
        self.min_agree = min_agree
        self.len_by_match = [int(x) for x in len_by_match]
        self.tokens: List[int] = []
        self._index: Dict[bytes, List[int]] = {}   # digest -> end positions (exclusive)
        self._by_end: Dict[int, List[bytes]] = {}  # end -> digests, ordered by n = 1..max
        self._refs: List[List[int]] = []
        self._ref_index: Dict[bytes, List[Tuple[int, int]]] = {}
        self._cur_len = num_pred
        self.calls = 0
        self.proposed = 0
        self.scored = 0     # draft tokens whose fate llama.cpp has revealed
        self.accepted = 0   # ... of which it kept
        self.candidates = 0  # candidate continuations considered across lookups
        self.ref_drafts = 0  # drafts that drew on a reference sequence
        self._pending: List[int] = []

    def reset(self) -> None:
        self.tokens = []
        self._index.clear()
        self._by_end.clear()
        self._pending = []
        self._cur_len = self.num_pred
        self.set_references([])

    def discard_pending(self) -> None:
        """Forget the last draft unscored; call when a generation ends, since the next call
        will start a new generation rather than reveal that draft's fate."""
        self._pending = []

    def set_references(self, refs: Sequence[Sequence[int]]) -> None:
        """Extra token sequences (e.g. retrieved past replies) to draft from this turn."""
        self._refs = [[int(t) for t in r] for r in refs if r]
        self._ref_index = {}
        windows: List[bytes] = []
        where: List[Tuple[int, int]] = []
        for r, toks in enumerate(self._refs):
            for end in range(1, len(toks)):  # a match at the very end has no continuation
                for size in range(1, self.max_ngram + 1):
                    if end - size < 0:
                        break
                    windows.append(struct.pack(f"<{size}i", *toks[end - size:end]))
                    where.append((r, end))
        if windows:
            for loc, digest in zip(where, self.acc.hash_batch(windows, stage="draft")):
                self._ref_index.setdefault(digest, []).append(loc)

    @property
    def references(self) -> int:
        return len(self._refs)

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
                if self.adaptive:
                    self._adapt(k, len(pending))
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

    def _adapt(self, accepted: int, proposed: int) -> None:
        if accepted >= proposed:
            self._cur_len = min(self.num_pred, max(self._cur_len, proposed) + 2)
        else:
            self._cur_len = max(self.min_pred, min(self._cur_len, accepted + 2))

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

    def _candidates(self, length: int) -> List[Tuple[float, List[int], bool]]:
        """(weight, continuation, from_reference), longest match first, most recent first."""
        total = len(self.tokens)
        suffix = self._by_end.get(total, [])
        seen = set()
        out: List[Tuple[float, List[int], bool]] = []
        for size in range(min(self.max_ngram, len(suffix)), 0, -1):
            digest = suffix[size - 1]
            w = float(size * size)
            for end in reversed(self._index.get(digest, [])):
                if end < total and ("L", end) not in seen:
                    seen.add(("L", end))
                    cont = self.tokens[end:end + length]
                    if cont:
                        out.append((w, cont, False))
            for r, end in self._ref_index.get(digest, []):
                if (r, end) not in seen:
                    seen.add((r, end))
                    cont = self._refs[r][end:end + length]
                    if cont:
                        out.append((w * 0.5, cont, True))
            if len(out) >= self.max_candidates:
                break
        return out[:self.max_candidates]

    def _match_size(self) -> int:
        """Length of the longest suffix n-gram with an earlier occurrence (live or reference)."""
        total = len(self.tokens)
        suffix = self._by_end.get(total, [])
        for size in range(min(self.max_ngram, len(suffix)), 0, -1):
            digest = suffix[size - 1]
            if self._ref_index.get(digest) or any(e < total for e in self._index.get(digest, ())):
                return size
        return 0

    def _lookup(self) -> List[int]:
        size = self._match_size()
        if size == 0:
            return []
        length = self._cur_len if self.adaptive else self.num_pred
        if size <= len(self.len_by_match):
            length = min(length, self.len_by_match[size - 1])
        if length <= 0:
            return []
        cands = self._candidates(length)
        self.candidates += len(cands)
        if not cands:
            return []
        if self.mode == "lookup":
            w, cont, ref = cands[0]
            self.ref_drafts += int(ref)
            return cont[:length]
        draft: List[int] = []
        alive = cands
        used_ref = False
        for pos in range(length):
            votes: Dict[int, float] = {}
            for w, cont, _ in alive:
                if pos < len(cont):
                    votes[cont[pos]] = votes.get(cont[pos], 0.0) + w
            if not votes:
                break
            tok, v = max(votes.items(), key=lambda kv: kv[1])  # ties: most recent candidate
            if pos > 0 and v / sum(votes.values()) < self.min_agree:
                break
            draft.append(tok)
            alive = [c for c in alive if pos < len(c[1]) and c[1][pos] == tok]
            used_ref = used_ref or any(c[2] for c in alive)
        self.ref_drafts += int(used_ref)
        return draft


def make_llama_draft_model(index: HashDraftIndex):
    """Wrap ``index`` as a llama-cpp-python ``LlamaDraftModel`` (imports llama_cpp lazily)."""
    import numpy as np
    from llama_cpp.llama_speculative import LlamaDraftModel

    class _HashDraftModel(LlamaDraftModel):
        def __call__(self, input_ids, /, **kwargs):
            return np.array(index(input_ids.tolist()), dtype=np.intc)

    return _HashDraftModel()
