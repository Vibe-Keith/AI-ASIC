"""Compact context hashes: SimHash signatures (host) and ASIC bucket IDs (native nonce search).

The similarity-preserving part runs on the host. A text's word and word-bigram features are
folded into a 64-bit SimHash, so related texts get signatures that differ in few bits; the
signature is cut into bands, and two texts whose band ``b`` is identical are LSH candidates.

The ASIC's part is the bucket function. Each ``(band index, band value)`` pair is written into an
80-byte header and the BM1387 nonce search maps it to a bucket ID - the double SHA-256 at the
first nonce meeting the bucket difficulty (HASHER's "first valid nonce" signature). Equal bands
give equal buckets. The bucket is a keyed hash of the band and nothing more: it adds no
similarity information, and a host SHA-256 would serve equally well. It is the step where the
chip does genuine, native work for routing and retrieval.

Buckets are cached. There are only ``n_bands * 2**band_bits`` possible headers (640 by
default), so after warm-up most lookups are cache hits and the chip is asked only for headers it
has not searched before - in one batched request.

Determinism on real silicon: the virtual chip and the host fallback both search upward from
nonce 0, so "first valid nonce" is well defined. A multi-core chip reports whichever core finds
a valid nonce first; for stable buckets the device must return the lowest valid nonce in the
searched range.
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ai_asic.workloads.split_model import DEVICE_HOST

SIG_BITS = 64
_WORD = re.compile(r"[a-z0-9_']+")
_STOP = frozenset(
    "a an and are as at be but by can do does for from had has have he her his how i if in "
    "into is it its me my no not of on or our she so than that the their them then there "
    "these they this to too us was we were what when where which who why will with would you "
    "your".split())
_NBITS = 0x1D00FFFF

Buckets = List[Optional[int]]


def features(text: str) -> Dict[str, float]:
    """Stemmed words (first 5 letters) plus within-word character 4-grams. Plain word and
    bigram features left short related texts no closer than unrelated ones; these separate
    them (related ~0.7-0.9 similarity, unrelated ~0.4-0.55 on short prompts)."""
    words = [w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOP]
    counts = Counter("s:" + w[:5] for w in words)
    for w in words:
        padded = f" {w} "
        counts.update("c:" + padded[i:i + 4] for i in range(len(padded) - 3))
    return {f: 1.0 + math.log(c) for f, c in counts.items()}


def simhash(text: str, bits: int = SIG_BITS) -> int:
    acc = [0.0] * bits
    for feat, w in features(text).items():
        h = int.from_bytes(hashlib.sha256(feat.encode("utf-8")).digest()[:8], "big")
        for i in range(bits):
            if (h >> i) & 1:
                acc[i] += w
            else:
                acc[i] -= w
    sig = 0
    for i in range(bits):
        if acc[i] > 0:
            sig |= 1 << i
    return sig


def similarity(a: int, b: int, bits: int = SIG_BITS) -> float:
    return 1.0 - bin(a ^ b).count("1") / bits


def bands(sig: int, n_bands: int = 10, band_bits: int = 6) -> List[int]:
    mask = (1 << band_bits) - 1
    return [(sig >> (i * band_bits)) & mask for i in range(n_bands)]


def band_header(index: int, value: int) -> bytes:
    """The 80-byte header whose nonce search defines the bucket of ``(index, value)``."""
    header = bytearray(80)
    struct.pack_into("<I", header, 0, 0x20000000)
    header[4:20] = b"AI-ASIC LSH band"
    struct.pack_into("<II", header, 20, index & 0xFFFFFFFF, value & 0xFFFFFFFF)
    struct.pack_into("<I", header, 72, _NBITS)
    return bytes(header)


def collide(a: Buckets, b: Buckets) -> bool:
    """True when two bucket lists share a bucket in the same band."""
    return any(x is not None and x == y for x, y in zip(a, b))


@dataclass
class BucketStats:
    searches: int = 0      # nonce searches sent to the device this call
    hashes: int = 0        # hashes those searches took
    cache_hits: int = 0
    round_trips: int = 0
    seconds: float = 0.0
    device: str = DEVICE_HOST


class BucketHasher:
    """Maps band values to bucket IDs by native nonce search, batched and cached."""

    def __init__(self, accelerator, difficulty_bits: int = 6, max_nonces: int = 4096):
        self.acc = accelerator
        self.difficulty_bits = difficulty_bits
        self.max_nonces = max_nonces
        self._cache: Dict[Tuple[int, int], Optional[int]] = {}
        self._inflight: Dict[Tuple[int, int], Future] = {}
        # Re-entrant: a host-inline search completes inside _submit, firing _store at once.
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._cache)

    def _submit(self, pairs: Sequence[Tuple[int, int]], stage: str) -> Dict[Tuple[int, int], Future]:
        """Queue searches for pairs not cached or already in flight (one batch)."""
        with self._lock:
            fresh = [p for p in dict.fromkeys(pairs)
                     if p not in self._cache and p not in self._inflight]
            futures = self.acc.submit_mine_batch(
                [band_header(i, v) for i, v in fresh], self.difficulty_bits,
                max_nonces=self.max_nonces, stage=stage) if fresh else []
            for p, f in zip(fresh, futures):
                self._inflight[p] = f
                f.add_done_callback(lambda fut, p=p: self._store(p, fut))
            return dict(zip(fresh, futures))

    def _store(self, pair: Tuple[int, int], fut: Future) -> None:
        try:
            r = fut.result()
            value = int(r.hash_hex[:16], 16) if r.found else None
        except Exception:
            value = None
        with self._lock:
            self._cache[pair] = value
            self._inflight.pop(pair, None)

    def buckets(self, band_lists: Sequence[Sequence[int]],
                stage: str = "route.buckets") -> Tuple[List[Buckets], BucketStats]:
        """Bucket IDs for each band list, waiting for any uncached searches (one batch)."""
        t0 = time.perf_counter()
        pairs = [(i, v) for bl in band_lists for i, v in enumerate(bl)]
        stats = BucketStats(device=self.acc.nonce_device)
        submitted = self._submit(pairs, stage)
        with self._lock:
            waiting = {p: f for p, f in self._inflight.items() if p in set(pairs)}
        waiting.update(submitted)
        for p, f in waiting.items():
            try:
                r = f.result()
                if p in submitted:
                    stats.hashes += r.hashes_tried
                    stats.device = r.device
            except Exception:
                pass
        stats.searches = len(submitted)
        stats.round_trips = (len(submitted) + 255) // 256 if submitted else 0
        stats.cache_hits = len(set(pairs)) - len(waiting)
        with self._lock:
            out = [[self._cache.get((i, v)) for i, v in enumerate(bl)] for bl in band_lists]
        stats.seconds = time.perf_counter() - t0
        return out, stats

    def prefetch(self, band_lists: Sequence[Sequence[int]], stage: str = "route.prefetch") -> int:
        """Start searches for these bands in the background; returns how many were queued."""
        pairs = [(i, v) for bl in band_lists for i, v in enumerate(bl)]
        return len(self._submit(pairs, stage))

    def cached(self, band_list: Sequence[int]) -> Optional[Buckets]:
        """Bucket IDs if every band is already cached, else None."""
        with self._lock:
            if all((i, v) in self._cache for i, v in enumerate(band_list)):
                return [self._cache[(i, v)] for i, v in enumerate(band_list)]
        return None
