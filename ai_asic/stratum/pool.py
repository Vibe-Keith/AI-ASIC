"""A private, host-side Stratum v1 pool: the closed-loop path from this host to a real miner.

Point the miner's pool URL at ``stratum+tcp://<this host's LAN IP>:<port>`` (any worker name and
password) and it mines the jobs this process hands it - nothing leaves the LAN, no outside pool
or internet is involved. :meth:`StratumPool.mine` commits a 32-byte payload into a job and
returns the first share the miner finds for it, as a verifiable proof-of-work.

Between requests the miner keeps hashing an idle job (a stock miner cannot be paused over
stratum, and a pool that stops sending work gets failed over); idle shares are acknowledged and
discarded. Stdlib only.
"""
from __future__ import annotations

import collections
import itertools
import json
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

from ai_asic.stratum import protocol as P

DEFAULT_STRATUM_PORT = 3333
# S9 ~13.5 TH/s: difficulty 256 is ~12 shares/s, so a seal lands in ~0.1 s on average without
# flooding the link. Scale it up with the miner's hashrate.
DEFAULT_SHARE_DIFFICULTY = 256.0
EXTRANONCE1_SIZE = 4
EXTRANONCE2_SIZE = 4
_JOB_HISTORY = 32


@dataclass
class Job:
    job_id: str
    prev: bytes
    payload: bytes
    coinb1: bytes
    coinb2: bytes
    ntime: int
    version: int = P.VERSION
    nbits: int = P.NBITS
    idle: bool = False

    def notify_params(self, clean: bool) -> list:
        return [self.job_id, P.swap_words(self.prev).hex(), self.coinb1.hex(),
                self.coinb2.hex(), [], f"{self.version:08x}", f"{self.nbits:08x}",
                f"{self.ntime:08x}", clean]


@dataclass
class PoolShare:
    """An accepted share for a :meth:`StratumPool.mine` request: the seal proof."""
    job_id: str
    coinbase: bytes
    header: bytes
    hash: bytes
    leading_zeros: int
    nonce: int
    miner: str
    seconds: float  # job pushed -> share received (includes the miner's work-restart latency)

    @property
    def pow_hash_hex(self) -> str:
        return self.hash[::-1].hex()  # display order: leading zeros first


@dataclass
class MinerSession:
    address: str
    extranonce1: bytes
    difficulty: float
    target: int
    user_agent: str = ""
    worker: str = ""
    authorized: bool = False
    version_mask: int = 0
    accepted: int = 0
    rejected: int = 0
    connected_at: float = field(default_factory=time.time)
    last_share_at: float = 0.0
    last_reject: str = ""


@dataclass
class _Waiter:
    job_id: str
    min_bits: int
    event: threading.Event = field(default_factory=threading.Event)
    share: Optional[PoolShare] = None
    started: float = field(default_factory=time.monotonic)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.server.pool._serve(self)  # type: ignore[attr-defined]


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class StratumPool:
    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_STRATUM_PORT,
                 share_difficulty: float = DEFAULT_SHARE_DIFFICULTY,
                 allow: Optional[Sequence[str]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.host = host
        self.port = port
        self.share_difficulty = float(share_difficulty)
        self.allow: Optional[Set[str]] = set(allow) if allow else None
        self.log = log or (lambda msg: None)
        self._server: Optional[_Server] = None
        self._lock = threading.Lock()
        self._sessions: Dict[_Handler, MinerSession] = {}
        self._writers: Dict[_Handler, threading.Lock] = {}
        self._jobs: "collections.OrderedDict[str, Job]" = collections.OrderedDict()
        self._current: Optional[Job] = None
        self._waiters: Dict[str, _Waiter] = {}
        self._seen: Set[Tuple] = set()
        self._ids = itertools.count(1)
        self._en1 = itertools.count(1)
        self._recent: Deque[Tuple[float, float]] = collections.deque()  # (time, difficulty)
        self._miner_joined = threading.Condition(self._lock)
        self._height = itertools.count(1)

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "StratumPool":
        if self._server is not None:
            return self
        server = _Server((self.host, self.port), _Handler)
        server.pool = self  # type: ignore[attr-defined]
        self.port = server.server_address[1]
        self._server = server
        self._set_job(self._make_job(b"\x00" * 32, b"\x00" * 32, int(time.time()), idle=True),
                      broadcast=False)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.log(f"stratum pool listening on {self.host}:{self.port} "
                 f"(share difficulty {self.share_difficulty:g})")
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        with self._lock:
            handlers = list(self._sessions)
            for w in self._waiters.values():
                w.event.set()
        for h in handlers:
            try:
                h.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._server = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def __enter__(self) -> "StratumPool":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- status -------------------------------------------------------------
    def miners(self) -> List[MinerSession]:
        with self._lock:
            return [s for s in self._sessions.values() if s.authorized]

    @property
    def has_miner(self) -> bool:
        return bool(self.miners())

    def wait_for_miner(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._lock:
            while not any(s.authorized for s in self._sessions.values()):
                left = deadline - time.monotonic()
                if left <= 0 or self._server is None:
                    return False
                self._miner_joined.wait(left)
            return True

    def hashrate(self, window: float = 60.0) -> float:
        """Hashes/s implied by accepted shares over the last ``window`` seconds."""
        now = time.time()
        with self._lock:
            while self._recent and self._recent[0][0] < now - window:
                self._recent.popleft()
            if len(self._recent) < 2:
                return 0.0
            span = now - self._recent[0][0]
            work = sum(d for _, d in self._recent) * 2 ** 32
        return work / span if span > 0 else 0.0

    # -- work ---------------------------------------------------------------
    def mine(self, prev: bytes, payload: bytes, ntime: int, min_bits: int = 0,
             timeout: float = 30.0) -> Optional[PoolShare]:
        """Commit ``payload`` into a fresh job, push it to every miner, and return the first
        share whose header hash has at least ``min_bits`` leading zero bits (every share already
        has :func:`~ai_asic.stratum.protocol.bits_for_difficulty` of them). ``None`` on timeout
        or when no miner is connected."""
        if not self.has_miner:
            return None
        job = self._make_job(prev, payload, ntime)
        waiter = _Waiter(job.job_id, min_bits)
        with self._lock:
            self._waiters[job.job_id] = waiter
        self._set_job(job)
        try:
            waiter.event.wait(timeout)
        finally:
            with self._lock:
                self._waiters.pop(job.job_id, None)
        return waiter.share

    def _make_job(self, prev: bytes, payload: bytes, ntime: int, idle: bool = False) -> Job:
        coinb1, coinb2 = P.build_coinbase(payload, next(self._height),
                                          EXTRANONCE1_SIZE + EXTRANONCE2_SIZE)
        return Job(f"{next(self._ids):x}", prev, payload, coinb1, coinb2, ntime & 0xFFFFFFFF,
                   idle=idle)

    def _set_job(self, job: Job, broadcast: bool = True) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            while len(self._jobs) > _JOB_HISTORY:
                self._jobs.popitem(last=False)
            self._current = job
            targets = [h for h, s in self._sessions.items() if s.authorized]
        if broadcast:
            for h in targets:
                self._send(h, {"id": None, "method": "mining.notify",
                               "params": job.notify_params(clean=True)})

    # -- connection handling ------------------------------------------------
    def _send(self, handler: _Handler, msg: dict) -> None:
        lock = self._writers.get(handler)
        if lock is None:
            return
        data = (json.dumps(msg) + "\n").encode("utf-8")
        with lock:
            try:
                handler.wfile.write(data)
                handler.wfile.flush()
            except OSError:
                pass

    def _serve(self, h: _Handler) -> None:
        addr = h.client_address[0]
        if self.allow is not None and addr not in self.allow:
            self.log(f"refused connection from {addr} (not in allow list)")
            return
        try:
            h.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        h.connection.settimeout(900.0)
        en1 = struct.pack(">I", next(self._en1) & 0xFFFFFFFF)
        session = MinerSession(addr, en1, self.share_difficulty,
                               P.target_for_difficulty(self.share_difficulty))
        with self._lock:
            self._sessions[h] = session
            self._writers[h] = threading.Lock()
        self.log(f"miner connected from {addr}")
        try:
            while True:
                try:
                    line = h.rfile.readline()
                except OSError:
                    break
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except ValueError:
                    continue
                if isinstance(msg, dict) and msg.get("method"):
                    self._handle(h, session, msg)
        finally:
            with self._lock:
                self._sessions.pop(h, None)
                self._writers.pop(h, None)
            self.log(f"miner {addr} disconnected ({session.accepted} shares accepted)")

    def _reply(self, h: _Handler, msg_id, result=None, error=None) -> None:
        if msg_id is not None:
            self._send(h, {"id": msg_id, "result": result, "error": error})

    def _handle(self, h: _Handler, s: MinerSession, msg: dict) -> None:
        method = str(msg["method"])
        params = msg.get("params") or []
        mid = msg.get("id")
        if method == "mining.subscribe":
            if params and isinstance(params[0], str):
                s.user_agent = params[0]
            self._reply(h, mid, [[["mining.set_difficulty", "1"], ["mining.notify", "1"]],
                                 s.extranonce1.hex(), EXTRANONCE2_SIZE])
        elif method == "mining.configure":
            exts = params[0] if params else []
            opts = params[1] if len(params) > 1 and isinstance(params[1], dict) else {}
            result = {}
            if "version-rolling" in exts:
                asked = int(str(opts.get("version-rolling.mask", "ffffffff")), 16)
                s.version_mask = asked & P.DEFAULT_VERSION_MASK
                result["version-rolling"] = bool(s.version_mask)
                result["version-rolling.mask"] = f"{s.version_mask:08x}"
            self._reply(h, mid, result)
        elif method == "mining.authorize":
            s.worker = str(params[0]) if params else ""
            self.log(f"miner {s.address} authorized as {s.worker!r} ({s.user_agent or '?'})")
            s.authorized = True
            self._reply(h, mid, True)
            self._send(h, {"id": None, "method": "mining.set_difficulty",
                           "params": [s.difficulty]})
            with self._lock:
                job = self._current
                self._miner_joined.notify_all()
            if job is not None:
                self._send(h, {"id": None, "method": "mining.notify",
                               "params": job.notify_params(clean=True)})
        elif method == "mining.submit":
            ok, err = self._submit(s, params)
            self._reply(h, mid, ok, err)
        elif method in ("mining.extranonce.subscribe", "mining.suggest_difficulty",
                        "mining.suggest_target"):
            self._reply(h, mid, True)
        else:
            self._reply(h, mid, None, [20, f"unsupported method {method}", None])

    def _reject(self, s: MinerSession, code: int, reason: str):
        s.rejected += 1
        s.last_reject = reason
        return False, [code, reason, None]

    def _submit(self, s: MinerSession, params: list):
        if not s.authorized:
            return self._reject(s, 24, "Unauthorized worker")
        try:
            _, job_id, en2_hex, ntime_hex, nonce_hex = (str(p) for p in params[:5])
            version_bits = str(params[5]) if len(params) > 5 and params[5] else None
            en2 = bytes.fromhex(en2_hex)
            if len(en2) != EXTRANONCE2_SIZE or len(bytes.fromhex(ntime_hex)) != 4 \
                    or len(bytes.fromhex(nonce_hex)) != 4:
                raise ValueError
        except (ValueError, TypeError):
            return self._reject(s, 20, "Malformed submit")
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return self._reject(s, 21, "Job not found (stale)")
        version = P.rolled_version(job.version, version_bits, s.version_mask)
        key = (job_id, s.extranonce1, en2, ntime_hex, nonce_hex, version)
        with self._lock:
            if key in self._seen:
                return self._reject(s, 22, "Duplicate share")
            self._seen.add(key)
            if len(self._seen) > 100_000:
                self._seen.clear()
        coinbase = job.coinb1 + s.extranonce1 + en2 + job.coinb2
        header = P.header_from_share(version, job.prev, P.dsha256(coinbase), ntime_hex,
                                     job.nbits, nonce_hex)
        digest = P.dsha256(header)
        if P.hash_value(digest) > s.target:
            return self._reject(s, 23, "Low difficulty share")
        now = time.time()
        s.accepted += 1
        s.last_share_at = now
        with self._lock:
            self._recent.append((now, s.difficulty))
            waiter = self._waiters.get(job_id)
        lz = P.leading_zero_bits(digest)
        if waiter is not None and not waiter.event.is_set() and lz >= waiter.min_bits:
            waiter.share = PoolShare(job_id, coinbase, header, digest, lz,
                                     struct.unpack("<I", header[76:80])[0],
                                     f"{s.address} {s.user_agent}".strip(),
                                     time.monotonic() - waiter.started)
            waiter.event.set()
        return True, None
