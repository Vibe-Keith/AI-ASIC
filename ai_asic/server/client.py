"""Client for the ported on-device hasher-server.

The host-side counterpart to :class:`~ai_asic.server.hasher_server.HasherServer`, mirroring the
original ``hasher-host`` -> ``hasher-server`` gRPC client. Same NUL-terminated JSON-over-TCP
transport as the cgminer client; stdlib only.
"""
from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_asic.server.device import MAX_BATCH_SIZE
from ai_asic.server.hasher_server import DEFAULT_PORT


@dataclass
class RemoteMineResult:
    found: bool
    nonce: int
    hash_hex: str
    leading_zeros: int
    hashes_tried: int
    midstate_index: int = 0
    latency_us: float = 0.0


class HasherClient:
    """Keeps one persistent connection to the server (reconnecting when it drops), so each
    call costs a request/response round trip rather than a fresh TCP handshake."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 timeout: float = 60.0):
        self.host = host or "127.0.0.1"
        self.port = port or DEFAULT_PORT
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._batch_supported = True

    def close(self) -> None:
        with self._lock:
            self._drop()

    def _drop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _connect(self) -> socket.socket:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return sock

    def _exchange(self, sock: socket.socket, payload: bytes) -> bytes:
        sock.sendall(payload)
        buf = bytearray()
        while b"\x00" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError(
                    f"hasher-server at {self.host}:{self.port} closed the connection")
            buf += chunk
        return bytes(buf).partition(b"\x00")[0]

    def _rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = json.dumps({"method": method, "params": params or {}}).encode("utf-8") + b"\x00"
        with self._lock:
            reused = self._sock is not None
            try:
                if self._sock is None:
                    self._sock = self._connect()
                raw = self._exchange(self._sock, payload)
            except OSError:  # includes ConnectionError and socket.timeout
                self._drop()
                if not reused:
                    raise
                # The kept-alive socket went stale (server idle timeout, restart, or a server
                # that closes after each reply); retry once on a fresh connection.
                try:
                    self._sock = self._connect()
                    raw = self._exchange(self._sock, payload)
                except OSError:
                    self._drop()
                    raise
        if not raw:
            raise ConnectionError(f"empty response from hasher-server at {self.host}:{self.port}")
        resp = json.loads(raw.decode("utf-8"))
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("error", "hasher-server error"))
        return resp

    # -- service methods ----------------------------------------------------
    def is_available(self) -> bool:
        try:
            self._rpc("Health")
            return True
        except Exception:
            return False

    def device_info(self) -> Dict[str, Any]:
        return self._rpc("GetDeviceInfo")["device"]

    def metrics(self) -> Dict[str, Any]:
        return self._rpc("GetMetrics")["metrics"]

    def compute_hash(self, data: bytes) -> bytes:
        return bytes.fromhex(self._rpc("ComputeHash", {"data": data.hex()})["hash"])

    def compute_batch(self, data: Sequence[bytes]) -> List[bytes]:
        resp = self._rpc("ComputeBatch", {"data": [d.hex() for d in data]})
        return [bytes.fromhex(h) for h in resp["hashes"]]

    def mine(self, header: bytes, difficulty_bits: int, max_nonces: int = 1 << 20,
             start: int = 0) -> RemoteMineResult:
        resp = self._rpc("Mine", {
            "header": header.hex(), "difficulty_bits": difficulty_bits,
            "max_nonces": max_nonces, "start": start,
        })
        return _mine_result(resp)

    def mine_batch(self, jobs: Sequence[Tuple[bytes, int, int]]) -> List[RemoteMineResult]:
        """Native nonce searches for ``(header, difficulty_bits, max_nonces)`` jobs in one round
        trip (``MineBatch``). Falls back to one ``Mine`` per job against a server that predates
        ``MineBatch``."""
        if not jobs:
            return []
        if len(jobs) > MAX_BATCH_SIZE:
            out: List[RemoteMineResult] = []
            for i in range(0, len(jobs), MAX_BATCH_SIZE):
                out.extend(self.mine_batch(jobs[i:i + MAX_BATCH_SIZE]))
            return out
        if self._batch_supported:
            try:
                resp = self._rpc("MineBatch", {"jobs": [
                    {"header": h.hex(), "difficulty_bits": int(b), "max_nonces": int(n)}
                    for h, b, n in jobs]})
                return [_mine_result(r) for r in resp["results"]]
            except RuntimeError as exc:
                if "unknown method" not in str(exc):
                    raise
                self._batch_supported = False
        return [self.mine(h, int(b), max_nonces=int(n)) for h, b, n in jobs]


def _mine_result(resp: Dict[str, Any]) -> RemoteMineResult:
    return RemoteMineResult(
        found=bool(resp["found"]), nonce=int(resp["nonce"]),
        hash_hex=str(resp["hash"]), leading_zeros=int(resp["leading_zeros"]),
        hashes_tried=int(resp["hashes_tried"]),
        midstate_index=int(resp.get("midstate_index", 0)),
        latency_us=float(resp.get("latency_us", 0.0)),
    )
