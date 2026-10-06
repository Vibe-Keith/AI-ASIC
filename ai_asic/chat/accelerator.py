"""The hash accelerator: native BM1387 nonce search on the ASIC, every other hash on the host.

A BM1387 does exactly one thing: roll a 32-bit nonce over an 80-byte block header and report
the nonces whose double SHA-256 meets a target. That is the only work this module sends to, or
counts against, the ASIC (``DEVICE_ASIC``). SHA-256 of arbitrary data - prompt fingerprints,
draft n-gram keys, seal digests - is not something a mining chip can do (on a real control
board the hasher-server runs it on the board's CPU), so :meth:`HashAccelerator.hash` and
:meth:`HashAccelerator.hash_batch` always run locally with ``hashlib`` and are counted as host
work. That also removes a network round trip from every draft step.

Nonce searches are asynchronous and batched. :meth:`HashAccelerator.submit_mine` and
:meth:`HashAccelerator.submit_mine_batch` return futures immediately; one worker thread drains
the queue and sends everything waiting as a single ``MineBatch`` request (up to 256 searches per
round trip). While idle it keeps the device path warm with a tiny search every
``warm_interval`` seconds, so a real request never pays a reconnect or a cold start. Keep-warm
searches are metered separately (:attr:`HashAccelerator.warm`) and never attributed to a turn.
"""
from __future__ import annotations

import hashlib
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

from ai_asic.server.device import MAX_BATCH_SIZE
from ai_asic.workloads.split_model import DEVICE_ASIC, DEVICE_HOST, _software_mine

WARM_STAGE = "warm"
_WARM_HEADER = b"AI-ASIC keep-warm".ljust(80, b"\x00")


@dataclass
class Meter:
    ops: int = 0          # hashes: nonce-search hashes (ASIC stages) or SHA-256 calls (host)
    calls: int = 0        # round trips (ASIC) or calls (host)
    seconds: float = 0.0
    jobs: int = 0         # nonce searches (ASIC stages)
    device: str = DEVICE_HOST


@dataclass
class NonceResult:
    """One nonce search, and the device that actually ran it."""

    nonce: int
    found: bool
    hash_hex: str
    leading_zeros: int
    hashes_tried: int
    device: str


@dataclass
class _Job:
    header: bytes
    difficulty_bits: int
    max_nonces: int
    stage: str
    future: Future


class HashAccelerator:
    """Native nonce search on a hasher-server (the ASIC) when reachable, else on the host.

    With ``pool`` (a running :class:`~ai_asic.stratum.pool.StratumPool`) it targets a real miner
    instead: a stock miner only takes work from a pool, so proof-of-work seals go to the chips
    via :meth:`pool_seal` while other nonce searches run on the host.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, pool=None,
                 keep_warm: bool = True, warm_interval: float = 20.0,
                 batch_window: float = 0.001):
        self._client = None
        self._client_label = ""
        self._lost = False
        self._pool = pool
        self.keep_warm = keep_warm
        self.warm_interval = warm_interval
        self.batch_window = batch_window
        self.meters: Dict[str, Meter] = {}
        self.warm = Meter(device=DEVICE_ASIC)  # keep-warm searches, never part of a turn
        self.round_trips = 0                   # MineBatch requests sent to the device
        self._mlock = threading.Lock()
        self._queue: "queue.Queue[Optional[List[_Job]]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._wlock = threading.Lock()
        self._closed = False
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
        if self._client is not None and keep_warm:
            self._ensure_worker()
            self._queue.put([self._warm_job()])  # warm the path before the first real request

    # -- device ---------------------------------------------------------------
    @property
    def has_pool(self) -> bool:
        return self._pool is not None

    @property
    def is_hardware(self) -> bool:
        return self._client is not None or (self._pool is not None and self._pool.has_miner)

    @property
    def label(self) -> str:
        if self._client is not None:
            return f"{self._client_label}; nonce search on the ASIC, other hashing on host CPU"
        if self._pool is not None:
            miners = self._pool.miners()
            who = ", ".join(f"{m.address} {m.user_agent}".strip() for m in miners)
            return (f"miner via private stratum pool :{self._pool.port} "
                    + (f"({who}); seals on the miner, other hashing on host CPU" if miners else
                       "(no miner connected; seals on host CPU)"))
        if self._lost:
            return "host CPU (hashlib; hasher-server unreachable)"
        return "host CPU (hashlib)"

    @property
    def pool_share_hashes(self) -> int:
        """Expected hashes behind one share from the pool's miner."""
        return int(getattr(self._pool, "share_difficulty", 0) * 2 ** 32) if self._pool else 0

    @property
    def device(self) -> str:
        """Where :meth:`hash` / :meth:`hash_batch` run: always the host."""
        return DEVICE_HOST

    @property
    def nonce_device(self) -> str:
        """Where :meth:`mine` / :meth:`submit_mine` searches run."""
        return DEVICE_ASIC if self._client is not None else DEVICE_HOST

    @property
    def mine_device(self) -> str:
        """Where proof-of-work seals run (the hasher-server's chip or the pool's miner)."""
        return DEVICE_ASIC if self.is_hardware else DEVICE_HOST

    # -- metering -----------------------------------------------------------
    def reset_meters(self) -> None:
        with self._mlock:
            self.meters = {}

    def meter(self, stage: str) -> Meter:
        with self._mlock:
            return self.meters.setdefault(stage, Meter())

    def _record(self, stage: str, ops: int, seconds: float, device: str,
                jobs: int = 0, calls: int = 1) -> None:
        with self._mlock:
            m = self.warm if stage == WARM_STAGE else self.meters.setdefault(stage, Meter())
            m.ops += ops
            m.calls += calls
            m.jobs += jobs
            m.seconds += seconds
            m.device = device

    def close(self) -> None:
        """Stop the worker and drop the device connection. Searches submitted afterwards (for
        example by a seal still running in the background) run inline on the host."""
        self._closed = True
        worker = self._worker
        if worker is not None:
            self._queue.put(None)
            if worker is not threading.current_thread():
                worker.join(timeout=5.0)
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def _lost_device(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.close()
        self._lost = True

    # -- host hashing (not ASIC work) ---------------------------------------
    def hash(self, data: bytes, stage: str = "misc") -> bytes:
        t0 = time.perf_counter()
        out = hashlib.sha256(data).digest()
        self._record(stage, 1, time.perf_counter() - t0, DEVICE_HOST)
        return out

    def hash_batch(self, data: Sequence[bytes], stage: str = "misc") -> List[bytes]:
        t0 = time.perf_counter()
        sha = hashlib.sha256
        out = [sha(d).digest() for d in data]
        self._record(stage, len(data), time.perf_counter() - t0, DEVICE_HOST)
        return out

    # -- native nonce search (ASIC work) --------------------------------------
    def submit_mine_batch(self, headers: Sequence[bytes], difficulty_bits: int,
                          max_nonces: int = 1 << 24, stage: str = "misc") -> List[Future]:
        """Queue nonce searches; returns one future per header, resolving to
        :class:`NonceResult`. Searches submitted together go to the device together."""
        jobs = [_Job(bytes(h), int(difficulty_bits), int(max_nonces), stage, Future())
                for h in headers]
        if not jobs:
            return []
        if self._closed or (self._client is None and self._worker is None):
            self._execute(jobs)  # no device and no worker (or closed): run inline
        else:
            self._ensure_worker()
            for i in range(0, len(jobs), MAX_BATCH_SIZE):
                self._queue.put(jobs[i:i + MAX_BATCH_SIZE])
        return [j.future for j in jobs]

    def submit_mine(self, header: bytes, difficulty_bits: int, max_nonces: int = 1 << 24,
                    stage: str = "misc") -> Future:
        return self.submit_mine_batch([header], difficulty_bits, max_nonces, stage)[0]

    def mine_batch(self, headers: Sequence[bytes], difficulty_bits: int,
                   max_nonces: int = 1 << 24, stage: str = "misc") -> List[NonceResult]:
        return [f.result() for f in
                self.submit_mine_batch(headers, difficulty_bits, max_nonces, stage)]

    def mine(self, header: bytes, difficulty_bits: int, max_nonces: int = 1 << 24,
             stage: str = "misc") -> NonceResult:
        return self.submit_mine(header, difficulty_bits, max_nonces, stage).result()

    def _warm_job(self) -> _Job:
        return _Job(_WARM_HEADER, 1, 64, WARM_STAGE, Future())

    def _ensure_worker(self) -> None:
        with self._wlock:
            if self._worker is None and not self._closed:
                self._worker = threading.Thread(target=self._run, name="asic-worker",
                                                daemon=True)
                self._worker.start()

    def _run(self) -> None:
        while True:
            warming = self.keep_warm and self._client is not None
            try:
                group = self._queue.get(timeout=self.warm_interval if warming else None)
            except queue.Empty:
                if not self._closed and self._client is not None:
                    self._execute([self._warm_job()])
                continue
            if group is None:
                self._drain()
                return
            batch = list(group)
            stop = False
            deadline = time.perf_counter() + self.batch_window
            while len(batch) < MAX_BATCH_SIZE:
                try:
                    nxt = self._queue.get(timeout=max(0.0, deadline - time.perf_counter()))
                except queue.Empty:
                    break
                if nxt is None:
                    stop = True
                    break
                if len(batch) + len(nxt) > MAX_BATCH_SIZE:
                    self._execute(batch)
                    batch = []
                batch.extend(nxt)
            if batch:
                self._execute(batch)
            if stop:
                self._drain()
                return

    def _drain(self) -> None:
        """Run anything queued after the stop sentinel, so no caller waits forever."""
        while True:
            try:
                group = self._queue.get_nowait()
            except queue.Empty:
                return
            if group:
                self._execute(list(group))

    def _execute(self, batch: List[_Job]) -> None:
        try:
            t0 = time.perf_counter()
            device = DEVICE_HOST
            raw = None
            client = self._client
            if client is not None:
                try:
                    raw = client.mine_batch(
                        [(j.header, j.difficulty_bits, j.max_nonces) for j in batch])
                    device = DEVICE_ASIC
                    self.round_trips += 1
                except Exception:
                    self._lost_device()
                    raw = None
            if raw is None:
                raw = [_software_mine(j.header, j.difficulty_bits, j.max_nonces) for j in batch]
            elapsed = time.perf_counter() - t0
            total = sum(r.hashes_tried for r in raw) or 1
            per_stage: Dict[str, List] = {}
            for j, r in zip(batch, raw):
                res = NonceResult(int(r.nonce), bool(r.found), str(r.hash_hex),
                                  int(r.leading_zeros), int(r.hashes_tried), device)
                acc = per_stage.setdefault(j.stage, [0, 0, 0.0])
                acc[0] += res.hashes_tried
                acc[1] += 1
                acc[2] += elapsed * res.hashes_tried / total
                j.future.set_result(res)
            for stage, (ops, jobs, secs) in per_stage.items():
                self._record(stage, ops, secs, device, jobs=jobs, calls=1)
        except Exception as exc:  # never leave a caller waiting forever
            for j in batch:
                if not j.future.done():
                    j.future.set_exception(exc)

    # -- real miner via the private pool --------------------------------------
    def pool_seal(self, prev: bytes, digest: bytes, ntime: int, difficulty_bits: int,
                  timeout: float = 30.0, stage: str = "misc"):
        """Have the pool's miner mine a header committing to ``digest`` (chained to ``prev``).
        Returns a :class:`~ai_asic.stratum.pool.PoolShare`, or ``None`` when there is no pool,
        no miner connected, or no share within ``timeout`` (callers then seal elsewhere)."""
        if self._pool is None or not self._pool.has_miner:
            return None
        t0 = time.perf_counter()
        share = self._pool.mine(prev, digest, ntime, min_bits=difficulty_bits, timeout=timeout)
        if share is not None:
            # Expected hashes behind one share at the pool's difficulty.
            self._record(stage, self.pool_share_hashes, time.perf_counter() - t0,
                         DEVICE_ASIC, jobs=1)
        return share


def copy_meters(meters: Dict[str, Meter]) -> Dict[str, Meter]:
    return {k: replace(v) for k, v in meters.items()}
