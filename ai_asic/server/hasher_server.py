"""The on-device hasher-server, ported to pure Python.

Original: ``cmd/hasher-server`` - a Go gRPC service (proto3 ``hasher.v1.HasherService``) cross
-compiled to a MIPS binary, deployed onto the miner by the host, talking to ``/dev/bitmain
-asic``. gRPC + protobuf would pull in third-party deps, so this port keeps the service's
methods and semantics but speaks the project's own transport: NUL-terminated JSON over TCP
(the same framing ``ai_asic/cgminer/client.py`` uses), stdlib only.

Methods (mirroring the proto ``HasherService``):

  * ``ComputeHash``   -> {hash, latency_us}
  * ``ComputeBatch``  -> {hashes, total_latency_us, processed_count}   (<=256)
  * ``StreamCompute`` -> collapsed onto ComputeBatch for this request/response transport
  * ``GetMetrics``    -> the counters the original gathered via eBPF trace points
  * ``GetDeviceInfo`` -> device/profile capabilities
  * ``Mine``          -> native nonce search (new; the primitive the split-model ASIC layer uses)
  * ``MineBatch``     -> up to 256 native nonce searches in one round trip (new)

Only ``Mine``/``MineBatch`` are native BM1387 work. ``ComputeHash``/``ComputeBatch`` hash
arbitrary data, which a mining chip cannot do; on a real control board they run on the board's
CPU. ``GetMetrics`` therefore reports ``native_hashes`` (nonce-search hashes) and ``cpu_hashes``
separately; ``total_hashes`` is their sum.

Backed by any :class:`~ai_asic.server.device.AsicDevice`.
"""
from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Set

from ai_asic.server.device import AsicDevice, VirtualAsicDevice, auto_device

DEFAULT_PORT = 8888  # the asic-driver default (docs/ASIC_DRIVER_README.md)


@dataclass
class Metrics:
    """The eBPF-trace counters the original server exposed via GetMetrics."""

    total_requests: int = 0
    total_bytes: int = 0
    total_hashes: int = 0
    native_hashes: int = 0   # nonce-search hashes (Mine / MineBatch): the chip's own work
    cpu_hashes: int = 0      # ComputeHash / ComputeBatch: arbitrary-data SHA-256, board CPU
    error_count: int = 0
    _latency_sum_us: float = 0.0
    peak_latency_us: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, nbytes: int, nhashes: int, latency_us: float, error: bool = False,
               native: bool = False) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_bytes += nbytes
            self.total_hashes += nhashes
            if native:
                self.native_hashes += nhashes
            else:
                self.cpu_hashes += nhashes
            self._latency_sum_us += latency_us
            if latency_us > self.peak_latency_us:
                self.peak_latency_us = latency_us
            if error:
                self.error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg = self._latency_sum_us / self.total_requests if self.total_requests else 0.0
            return {
                "total_requests": self.total_requests,
                "total_bytes": self.total_bytes,
                "total_hashes": self.total_hashes,
                "native_hashes": self.native_hashes,
                "cpu_hashes": self.cpu_hashes,
                "avg_latency_us": round(avg, 2),
                "peak_latency_us": round(self.peak_latency_us, 2),
                "error_count": self.error_count,
            }


IDLE_TIMEOUT_S = 300.0  # a persistent connection idle this long is closed; clients reconnect


class _Handler(socketserver.BaseRequestHandler):
    """Serves NUL-framed requests on one connection until the client closes it, so a client
    can keep a single socket open instead of paying a TCP connect per call."""

    def handle(self) -> None:
        sock = self.request
        sock.settimeout(IDLE_TIMEOUT_S)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        buf = bytearray()
        while True:
            try:
                while b"\x00" not in buf:
                    chunk = sock.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
            except OSError:
                return
            raw, _, rest = bytes(buf).partition(b"\x00")
            buf = bytearray(rest)
            try:
                req = json.loads(raw.decode("utf-8")) if raw else {}
            except ValueError:
                req = {}
            resp = self.server.dispatch(req)  # type: ignore[attr-defined]
            try:
                sock.sendall(json.dumps(resp).encode("utf-8") + b"\x00")
            except OSError:
                return


class _TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self._conns: Set[socket.socket] = set()
        self._conns_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def process_request_thread(self, request, client_address):
        with self._conns_lock:
            self._conns.add(request)
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._conns_lock:
                self._conns.discard(request)

    def close_connections(self) -> None:
        """Drop open client connections, so a stopped server is unreachable rather than
        still answering on sockets accepted before ``shutdown``."""
        with self._conns_lock:
            conns = list(self._conns)
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class HasherServer:
    """Runs an :class:`AsicDevice` behind the ported HasherService API."""

    def __init__(self, device: Optional[AsicDevice] = None, model: str = "Antminer S9",
                 host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 difficulty_bits: int = 16):
        self.device = device or auto_device(model)
        self.host = host
        self.port = port
        self.difficulty_bits = difficulty_bits
        self.metrics = Metrics()
        self._server: Optional[_TCPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "HasherServer":
        if self._server is not None:
            return self
        server = _TCPServer((self.host, self.port), _Handler)
        server.dispatch = self.dispatch  # type: ignore[attr-defined]
        self.port = server.server_address[1]
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server.close_connections()
            self._server = None
            self._thread = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def __enter__(self) -> "HasherServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- dispatch -----------------------------------------------------------
    def dispatch(self, req: dict) -> dict:
        method = str(req.get("method", "")).strip()
        params = req.get("params") or {}
        try:
            if method in ("ComputeHash", "compute_hash"):
                return self._compute_hash(params)
            if method in ("ComputeBatch", "compute_batch", "StreamCompute", "stream_compute"):
                return self._compute_batch(params)
            if method in ("Mine", "mine"):
                return self._mine(params)
            if method in ("MineBatch", "mine_batch"):
                return self._mine_batch(params)
            if method in ("GetMetrics", "get_metrics"):
                return {"ok": True, "metrics": self.metrics.snapshot()}
            if method in ("GetDeviceInfo", "get_device_info", "info"):
                return {"ok": True, "device": self.device.info(),
                        "server_difficulty_bits": self.difficulty_bits}
            if method in ("Health", "health", "ping"):
                return {"ok": True, "available": self.device.available,
                        "device": self.device.name()}
            return {"ok": False, "error": f"unknown method: {method!r}"}
        except Exception as exc:
            self.metrics.record(0, 0, 0.0, error=True)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _compute_hash(self, params: dict) -> dict:
        data = bytes.fromhex(params["data"])
        t0 = time.perf_counter()
        digest = self.device.compute_hash(data)
        us = (time.perf_counter() - t0) * 1e6
        self.metrics.record(len(data), 1, us)
        return {"ok": True, "hash": digest.hex(), "latency_us": round(us, 2)}

    def _compute_batch(self, params: dict) -> dict:
        items = [bytes.fromhex(h) for h in params.get("data", [])]
        t0 = time.perf_counter()
        hashes = self.device.compute_batch(items)
        us = (time.perf_counter() - t0) * 1e6
        self.metrics.record(sum(len(d) for d in items), len(items), us)
        return {"ok": True, "hashes": [h.hex() for h in hashes],
                "total_latency_us": round(us, 2), "processed_count": len(hashes)}

    def _mine(self, params: dict) -> dict:
        header = bytes.fromhex(params["header"])
        difficulty = int(params.get("difficulty_bits", self.difficulty_bits))
        max_nonces = int(params.get("max_nonces", 1 << 20))
        start = int(params.get("start", 0))
        t0 = time.perf_counter()
        out = self.device.mine(header, difficulty, max_nonces=max_nonces, start=start)
        us = (time.perf_counter() - t0) * 1e6
        self.metrics.record(80, out.hashes_tried, us, native=True)
        return {"ok": True, **_mine_fields(out), "latency_us": round(us, 2)}

    def _mine_batch(self, params: dict) -> dict:
        jobs = []
        for j in params.get("jobs", []):
            jobs.append({
                "header": bytes.fromhex(j["header"]),
                "difficulty_bits": int(j.get("difficulty_bits", self.difficulty_bits)),
                "max_nonces": int(j.get("max_nonces", 1 << 20)),
                "start": int(j.get("start", 0)),
            })
        t0 = time.perf_counter()
        outs = self.device.mine_batch(jobs)
        us = (time.perf_counter() - t0) * 1e6
        self.metrics.record(80 * len(jobs), sum(o.hashes_tried for o in outs), us, native=True)
        return {"ok": True, "results": [_mine_fields(o) for o in outs],
                "total_latency_us": round(us, 2)}


def _mine_fields(out) -> dict:
    return {"found": out.found, "nonce": out.nonce, "nonce_hex": f"0x{out.nonce:08x}",
            "hash": out.hash_hex, "leading_zeros": out.leading_zeros,
            "hashes_tried": out.hashes_tried, "midstate_index": out.midstate_index}
