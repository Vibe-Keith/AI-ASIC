"""The hash accelerator: routes the chat pipeline's SHA-256 work to the ASIC.

Like a GPU for matrix maths, the ASIC here is a co-processor for one class of operation -
SHA-256 and nonce search. :class:`HashAccelerator` sends those ops to an on-device
hasher-server when one is reachable and falls back to ``hashlib`` on the host otherwise, so the
pipeline always works and the trace always says honestly which device did the work.
Per-stage meters let the chat engine report time and op counts for each offloaded stage.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ai_asic.server.device import MAX_BATCH_SIZE
from ai_asic.workloads.split_model import DEVICE_ASIC, DEVICE_HOST, _software_mine


@dataclass
class Meter:
    ops: int = 0
    calls: int = 0
    seconds: float = 0.0


class HashAccelerator:
    """SHA-256 co-processor: hasher-server if reachable, else host ``hashlib``.

    With ``pool`` (a running :class:`~ai_asic.stratum.pool.StratumPool`) it targets a real miner
    instead: a stock miner only takes work from a pool and only double-hashes block headers, so
    hashing of arbitrary data (fingerprint, draft) stays on the host and proof-of-work seals go
    to the chips via :meth:`pool_seal`.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, pool=None):
        self._client = None
        self._client_label = ""
        self._lost = False
        self._pool = pool
        self.meters: Dict[str, Meter] = {}
        if host and port and pool is None:
            try:
                from ai_asic.server.client import HasherClient

                client = HasherClient(host, int(port))
                if client.is_available():
                    chip = client.device_info().get("chip", "?")
                    self._client = client
                    self._client_label = f"ASIC via hasher-server @ {host}:{port} ({chip})"
            except Exception:
                self._client = None

    @property
    def has_pool(self) -> bool:
        return self._pool is not None

    @property
    def is_hardware(self) -> bool:
        return self._client is not None or (self._pool is not None and self._pool.has_miner)

    @property
    def label(self) -> str:
        if self._client is not None:
            return self._client_label
        if self._pool is not None:
            miners = self._pool.miners()
            who = ", ".join(f"{m.address} {m.user_agent}".strip() for m in miners)
            return (f"miner via private stratum pool :{self._pool.port} "
                    + (f"({who}); hashing on host CPU" if miners else
                       "(no miner connected; seals on host CPU)"))
        if self._lost:
            return "host CPU (hashlib; hasher-server unreachable)"
        return "host CPU (hashlib)"

    @property
    def device(self) -> str:
        """Where :meth:`hash` / :meth:`hash_batch` run."""
        return DEVICE_ASIC if self._client is not None else DEVICE_HOST

    @property
    def mine_device(self) -> str:
        """Where proof-of-work runs."""
        return DEVICE_ASIC if self.is_hardware else DEVICE_HOST

    # -- metering -----------------------------------------------------------
    def reset_meters(self) -> None:
        self.meters = {}

    def meter(self, stage: str) -> Meter:
        return self.meters.setdefault(stage, Meter())

    def _record(self, stage: str, ops: int, t0: float) -> None:
        m = self.meter(stage)
        m.ops += ops
        m.calls += 1
        m.seconds += time.perf_counter() - t0

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _lost_device(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._lost = True

    # -- ops ------------------------------------------------------------------
    def hash(self, data: bytes, stage: str = "misc") -> bytes:
        t0 = time.perf_counter()
        if self._client is not None:
            try:
                out = self._client.compute_hash(data)
                self._record(stage, 1, t0)
                return out
            except Exception:
                self._lost_device()
        out = hashlib.sha256(data).digest()
        self._record(stage, 1, t0)
        return out

    def hash_batch(self, data: Sequence[bytes], stage: str = "misc") -> List[bytes]:
        t0 = time.perf_counter()
        out: List[bytes] = []
        if self._client is not None:
            try:
                for i in range(0, len(data), MAX_BATCH_SIZE):
                    out.extend(self._client.compute_batch(data[i:i + MAX_BATCH_SIZE]))
                self._record(stage, len(data), t0)
                return out
            except Exception:
                self._lost_device()
                out = []
        out = [hashlib.sha256(d).digest() for d in data]
        self._record(stage, len(data), t0)
        return out

    def mine(self, header: bytes, difficulty_bits: int, max_nonces: int = 1 << 24,
             stage: str = "misc"):
        """Native nonce search. Returns an object with nonce/found/hash_hex/leading_zeros/
        hashes_tried."""
        t0 = time.perf_counter()
        if self._client is not None:
            try:
                out = self._client.mine(header, difficulty_bits, max_nonces=max_nonces)
                self._record(stage, out.hashes_tried, t0)
                return out
            except Exception:
                self._lost_device()
        out = _software_mine(header, difficulty_bits, max_nonces)
        self._record(stage, out.hashes_tried, t0)
        return out

    def pool_seal(self, prev: bytes, digest: bytes, ntime: int, difficulty_bits: int,
                  timeout: float = 30.0, stage: str = "misc"):
        """Have the pool's miner mine a header committing to ``digest`` (chained to ``prev``).
        Returns a :class:`~ai_asic.stratum.pool.PoolShare`, or ``None`` when there is no pool,
        no miner connected, or no share within ``timeout`` (callers then seal on the host)."""
        if self._pool is None or not self._pool.has_miner:
            return None
        t0 = time.perf_counter()
        share = self._pool.mine(prev, digest, ntime, min_bits=difficulty_bits, timeout=timeout)
        if share is not None:
            # Expected hashes behind one share at the pool's difficulty.
            self._record(stage, int(self._pool.share_difficulty * 2 ** 32), t0)
        return share
