"""KV-block routing: attend over the history blocks that matter, not the whole conversation.

Every past exchange (a user message and its reply) is a block of KV-cache entries llama.cpp
must hold and attend over at every forward pass: attention work per token grows with context
length. Routing is sticky, because llama.cpp only reuses the KV cache for the unchanged *prefix* of
the prompt. Re-selecting blocks every turn re-prefills everything after the first change, and in
testing that doubled total attention work in a live chat even though each step attended over
less. So the router compacts rarely and appends in between:

  * While the prompt fits ``budget_tokens``, nothing is dropped.
  * When it first exceeds the budget, a **compaction** rebuilds the history down to a low-water
    mark (``low_water`` x budget) from
      - the ``keep_recent`` latest exchanges (always: the local thread of the conversation),
      - older exchanges whose LSH buckets collide with the new message's buckets (the ASIC
        part: bucket IDs come from batched BM1387 nonce search), best similarity first,
      - then the most recent remaining exchanges, while the target allows.
  * Later turns keep that selection and **append** new exchanges, so the KV prefix is reused,
    until the budget is exceeded again.
  * **Recall**: if the new message strongly matches a dropped exchange (bucket collision and
    signature similarity >= ``recall_similarity``), compact early so it comes back.

Blocks left out are never prefilled into the KV cache, so every prefill and decode step attends
over a shorter context. Selection is at exchange granularity (what the chat template can
express); per-token, per-step KV page selection would need changes to llama.cpp's attention
kernels. A dropped block is information the model no longer sees: check reply quality with the
routing benchmark.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ai_asic.chat.lsh import BucketHasher, BucketStats, Buckets, bands, collide, similarity, simhash

Message = Dict[str, str]


@dataclass
class RouteResult:
    messages: List[Message]
    routed: bool                       # True when blocks were dropped
    tokens_full: int                   # prompt tokens with the whole history
    tokens_routed: int                 # prompt tokens actually sent
    blocks_total: int = 0              # history exchanges
    blocks_kept: int = 0
    candidates: int = 0                # older exchanges whose buckets collided with the query
    kept: List[int] = field(default_factory=list)
    compacted: bool = False            # this turn rebuilt the selection (KV prefix not reused)
    recalled: int = 0                  # dropped exchanges that triggered an early compaction
    query_sig: Optional[int] = None
    query_buckets: Optional[Buckets] = None
    buckets: BucketStats = field(default_factory=BucketStats)
    select_seconds: float = 0.0
    reason: str = ""


def exchanges(history: Sequence[Message]) -> List[List[Message]]:
    """Group history into blocks: each user message with the replies that follow it."""
    blocks: List[List[Message]] = []
    for m in history:
        if m.get("role") == "user" or not blocks:
            blocks.append([m])
        else:
            blocks[-1].append(m)
    return blocks


class ContextRouter:
    def __init__(self, hasher: BucketHasher, budget_tokens: int = 768, keep_recent: int = 2,
                 n_bands: int = 10, band_bits: int = 6, low_water: float = 0.6,
                 recall_similarity: float = 0.8):
        self.hasher = hasher
        self.budget_tokens = budget_tokens
        self.keep_recent = keep_recent
        self.n_bands = n_bands
        self.band_bits = band_bits
        self.low_water = low_water
        self.recall_similarity = recall_similarity
        self._sigs: Dict[str, int] = {}
        self._pinned: Optional[List[int]] = None  # kept exchanges at the last compaction
        self._pinned_upto = 0                     # exchanges that existed at that compaction

    def reset(self) -> None:
        """Forget the current selection (call when a new conversation starts)."""
        self._pinned = None
        self._pinned_upto = 0

    def signature(self, text: str) -> int:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sig = self._sigs.get(key)
        if sig is None:
            sig = simhash(text)
            if len(self._sigs) > 4096:
                self._sigs.clear()
            self._sigs[key] = sig
        return sig

    def bands(self, sig: int) -> List[int]:
        return bands(sig, self.n_bands, self.band_bits)

    @staticmethod
    def block_text(block: Sequence[Message]) -> str:
        return "\n".join(m.get("content", "") for m in block)

    def prefetch(self, text: str) -> int:
        """Queue bucket searches for a finished exchange in the background, so routing the next
        turn finds them cached. Returns how many searches were queued."""
        return self.hasher.prefetch([self.bands(self.signature(text))])

    def query(self, text: str) -> Tuple[int, Buckets, BucketStats]:
        sig = self.signature(text)
        (qb,), stats = self.hasher.buckets([self.bands(sig)])
        return sig, qb, stats

    def route(self, msgs: Sequence[Message], count: Callable[[Message], int],
              enabled: bool = True, want_query_buckets: bool = False) -> RouteResult:
        """Choose the history blocks for this turn. ``msgs`` is system + history + the new user
        message; ``count`` gives a message's prompt-token cost."""
        t0 = time.perf_counter()
        msgs = list(msgs)
        system, body, query = msgs[:1], msgs[1:-1], msgs[-1:]
        blocks = exchanges(body)
        sizes = [sum(count(m) for m in b) for b in blocks]
        fixed = sum(count(m) for m in system + query)
        full = fixed + sum(sizes)
        qtext = query[0].get("content", "") if query else ""
        qsig = self.signature(qtext)
        result = RouteResult(messages=msgs, routed=False, tokens_full=full, tokens_routed=full,
                             blocks_total=len(blocks), blocks_kept=len(blocks),
                             kept=list(range(len(blocks))), query_sig=qsig)

        recent_start = max(0, len(blocks) - self.keep_recent)
        if not enabled or full <= self.budget_tokens or recent_start == 0:
            self._pinned = None
            result.reason = ("routing off" if not enabled else
                             "fits budget" if full <= self.budget_tokens else
                             "only recent exchanges")
            if want_query_buckets:
                (qb,), result.buckets = self.hasher.buckets([self.bands(qsig)])
                result.query_buckets = qb
            result.select_seconds = time.perf_counter() - t0 - result.buckets.seconds
            return result

        older = list(range(recent_start))
        sigs = {i: self.signature(self.block_text(blocks[i])) for i in older}
        # One batched request covers the query and any older block not bucketed yet.
        lists, result.buckets = self.hasher.buckets(
            [self.bands(qsig)] + [self.bands(sigs[i]) for i in older])
        qb = lists[0]
        older_b = dict(zip(older, lists[1:]))
        result.query_buckets = qb
        matches = {i: similarity(qsig, sigs[i]) for i in older if collide(qb, older_b[i])}
        result.candidates = len(matches)

        if self._pinned is not None and self._pinned_upto > len(blocks):
            self._pinned = None  # history was replaced; start over
        keep: Optional[List[int]] = None
        if self._pinned is not None:
            sticky = self._pinned + list(range(self._pinned_upto, len(blocks)))
            dropped = set(older) - set(sticky)
            recalled = [i for i in dropped if matches.get(i, 0.0) >= self.recall_similarity]
            if recalled:
                result.recalled = len(recalled)
            elif fixed + sum(sizes[i] for i in sticky) <= self.budget_tokens:
                keep = sticky
        if keep is None:
            keep = self._compact(blocks, sizes, fixed, recent_start, matches)
            result.compacted = True
            self._pinned, self._pinned_upto = keep, len(blocks)

        result.messages = system + [m for i in keep for m in blocks[i]] + query
        result.kept = keep
        result.blocks_kept = len(keep)
        result.tokens_routed = fixed + sum(sizes[i] for i in keep)
        result.routed = len(keep) < len(blocks)
        what = ("compacted" + (f" to recall {result.recalled} exchange(s)" if result.recalled
                               else "") if result.compacted else "kept selection (KV reused)")
        result.reason = (f"{what}: {len(keep)}/{len(blocks)} exchanges, "
                         f"{len(matches)} bucket match(es)")
        result.select_seconds = time.perf_counter() - t0 - result.buckets.seconds
        return result

    def _compact(self, blocks, sizes, fixed: int, recent_start: int,
                 matches: Dict[int, float]) -> List[int]:
        target = max(int(self.budget_tokens * self.low_water), 0)
        keep = set(range(recent_start, len(blocks)))
        used = fixed + sum(sizes[i] for i in keep)
        for i in sorted(matches, key=lambda i: (matches[i], i), reverse=True):
            if used + sizes[i] <= target:
                keep.add(i)
                used += sizes[i]
        for i in range(recent_start - 1, -1, -1):
            if i not in keep and used + sizes[i] <= target:
                keep.add(i)
                used += sizes[i]
        return sorted(keep)
