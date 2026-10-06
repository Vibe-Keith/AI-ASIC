"""Host-side nonce search: the one hot loop every software mining path shares.

Finds the first nonce in ``[start, start + count)`` whose double SHA-256 over an 80-byte header
has at least ``difficulty_bits`` leading zero bits (the digest read big-endian, as everywhere
else in this package). Three things make it faster than re-hashing the header per nonce:

  * **Midstate resume** - header bytes 0-75 are absorbed into a ``hashlib`` object once and
    copied per nonce, so each roll only feeds the 4 nonce bytes (the same shortcut a BM1387
    takes with the host's midstate).
  * **Byte-compare target** - ``digest < target`` on 32-byte strings replaces converting every
    digest to a Python int and counting bits.
  * **Process parallelism for long searches** - ``hashlib`` keeps the GIL for inputs this small,
    so threads do not help. When the expected work is at least ``PARALLEL_MIN_WORK`` hashes,
    the range is split into chunks searched by a persistent process pool. Chunks are consumed
    in nonce order, so the result (first valid nonce, its hash, hashes tried) is identical to a
    serial search. :func:`search_many` (a batch of searches) also spreads shorter searches one
    per process when together they are worth it. Set ``AI_ASIC_WORKERS=1`` to disable; any pool
    failure falls back to serial.

Pure Python, standard library only.
"""
from __future__ import annotations

import atexit
import hashlib
import os
import struct
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Sequence, Tuple

_MASK = 0xFFFFFFFF
_PACK = struct.Struct("<I").pack

# Expected hashes (min(count, 2**difficulty_bits)) at which a search goes to the process pool.
PARALLEL_MIN_WORK = 1 << 20
# Nonces per pool task (~0.1-0.2 s each in CPython).
CHUNK = 1 << 17

SearchResult = Tuple[Optional[int], bytes, int]  # (nonce or None, digest, hashes tried)
Job = Tuple[bytes, int, int, int]                # (header, difficulty_bits, start, count)


def target_for_bits(difficulty_bits: int) -> bytes:
    """A byte string ``t`` such that ``digest < t`` exactly when the 32-byte digest has at least
    ``difficulty_bits`` leading zero bits."""
    bits = int(difficulty_bits)
    if bits <= 0:
        return b"\xff" * 32 + b"\x00"  # every 32-byte digest compares lower
    if bits > 256:
        return b""                     # nothing compares lower
    return (1 << (256 - bits)).to_bytes(32, "big")


def search_serial(header: bytes, difficulty_bits: int, start: int, count: int) -> SearchResult:
    """Single-threaded search. Returns ``(nonce, digest, tried)``; ``nonce`` is ``None`` (and
    ``digest`` empty) when no nonce in the range qualifies."""
    if len(header) != 80:
        raise ValueError("mining header must be exactly 80 bytes")
    target = target_for_bits(difficulty_bits)
    prefix = hashlib.sha256(header[:76])
    copy = prefix.copy
    sha = hashlib.sha256
    pack = _PACK
    end = start + count
    if start >= 0 and end <= _MASK + 1:
        for nonce in range(start, end):
            h = copy()
            h.update(pack(nonce))
            d = sha(h.digest()).digest()
            if d < target:
                return nonce, d, nonce - start + 1
    else:
        for nonce in range(start, end):
            h = copy()
            h.update(pack(nonce & _MASK))
            d = sha(h.digest()).digest()
            if d < target:
                return nonce, d, nonce - start + 1
    return None, b"", max(0, count)


# --- process pool -------------------------------------------------------------------------

_pool: Optional[ProcessPoolExecutor] = None
_pool_lock = threading.Lock()
_pool_broken = False


def workers() -> int:
    env = os.environ.get("AI_ASIC_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return os.cpu_count() or 1


def _get_pool(n: int) -> Optional[ProcessPoolExecutor]:
    global _pool
    with _pool_lock:
        if _pool is None and not _pool_broken:
            import multiprocessing

            # spawn: safe from threaded parents (servers, the accelerator worker) on every OS.
            _pool = ProcessPoolExecutor(max_workers=n,
                                        mp_context=multiprocessing.get_context("spawn"))
        return _pool


def _shutdown_pool() -> None:
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # Python 3.8: no cancel_futures
            pool.shutdown(wait=False)


atexit.register(_shutdown_pool)


def _search_parallel(header: bytes, difficulty_bits: int, start: int, count: int,
                     n: int) -> SearchResult:
    global _pool_broken
    pool = _get_pool(n)
    if pool is None:
        return search_serial(header, difficulty_bits, start, count)
    end = start + count
    pending = []
    nxt = start
    try:
        while nxt < end or pending:
            while nxt < end and len(pending) < 2 * n:
                size = min(CHUNK, end - nxt)
                pending.append(pool.submit(search_serial, header, difficulty_bits, nxt, size))
                nxt += size
            nonce, digest, _ = pending.pop(0).result()  # in nonce order
            if nonce is not None:
                for f in pending:
                    f.cancel()
                return nonce, digest, nonce - start + 1
        return None, b"", count
    except Exception:
        # Broken pool (e.g. an unguarded __main__ under spawn): never fail the search.
        for f in pending:
            f.cancel()
        _pool_broken = True
        _shutdown_pool()
        return search_serial(header, difficulty_bits, start, count)


def search(header: bytes, difficulty_bits: int, start: int = 0,
           count: int = 1 << 24) -> SearchResult:
    """First nonce in ``[start, start + count)`` meeting ``difficulty_bits``; serial for short
    searches, process-parallel for long ones (same result either way)."""
    header = bytes(header)
    if len(header) != 80:
        raise ValueError("mining header must be exactly 80 bytes")
    count = max(0, int(count))
    expected = _expected(difficulty_bits, count)
    n = workers()
    if n > 1 and expected >= PARALLEL_MIN_WORK and count > CHUNK:
        return _search_parallel(header, difficulty_bits, start, count, n)
    return search_serial(header, difficulty_bits, start, count)


def _expected(difficulty_bits: int, count: int) -> int:
    return min(max(0, int(count)), 1 << max(0, min(int(difficulty_bits), 62)))


def _search_job(job: Job) -> SearchResult:
    return search_serial(*job)


def search_many(jobs: Sequence[Job]) -> List[SearchResult]:
    """Several independent searches (a ``MineBatch``). Long ones are each split across the pool
    (:func:`search`); short ones run one per worker process when together they are worth it,
    else serially. Results are in job order and identical to searching each job alone."""
    global _pool_broken
    jobs = [(bytes(h), int(b), int(s), max(0, int(c))) for h, b, s, c in jobs]
    for h, _, _, _ in jobs:
        if len(h) != 80:
            raise ValueError("mining header must be exactly 80 bytes")
    out: List[Optional[SearchResult]] = [None] * len(jobs)
    small = []
    for i, (h, b, s, c) in enumerate(jobs):
        if _expected(b, c) >= PARALLEL_MIN_WORK:
            out[i] = search(h, b, s, c)
        else:
            small.append(i)
    n = workers()
    if (n > 1 and len(small) > 1
            and sum(_expected(jobs[i][1], jobs[i][3]) for i in small) >= PARALLEL_MIN_WORK):
        pool = _get_pool(n)
        if pool is not None:
            try:
                for i, r in zip(small, pool.map(_search_job, [jobs[i] for i in small])):
                    out[i] = r
                small = []
            except Exception:
                _pool_broken = True
                _shutdown_pool()
    for i in small:
        out[i] = search_serial(*jobs[i])
    return out  # type: ignore[return-value]


def leading_zero_bits(digest: bytes) -> int:
    value = int.from_bytes(digest, "big")
    return 256 - value.bit_length() if value else 256
