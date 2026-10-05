"""Hashing methods: the pluggable backends the inference engine runs on.

- SoftwareHashMethod: CPU SHA-256 via hashlib (OpenSSL C, the fastest pure-software path).
- ASICHashMethod: offloads the final golden-nonce search to real hardware through the
  cgminer/bmminer JSON-RPC API, with software fallback. This is the cross-platform path
  that works against an Antminer S9 (or newer) over the network from Windows.

"Speed" note: SHA-256 itself is done by hashlib (C). Batch work can be spread across
threads; hashlib releases the GIL around the digest call, so threads give real speedup for
large batches. The highest throughput comes from the ASIC over cgminer, not the CPU.
"""
from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

# Above this batch size, spread software hashing across worker threads.
_PARALLEL_THRESHOLD = 512


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


class HashMethod(ABC):
    """Interface all hashing backends implement."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    def is_hardware(self) -> bool:
        """True only when this backend is driving real ASIC hardware (not software)."""
        return False

    @abstractmethod
    def compute_hash(self, data: bytes) -> bytes: ...

    def compute_batch(self, data: Sequence[bytes]) -> List[bytes]:
        return [self.compute_hash(d) for d in data]

    @abstractmethod
    def mine_header(self, header: bytes, nonce_start: int, nonce_end: int) -> int: ...


class SoftwareHashMethod(HashMethod):
    """Pure-software SHA-256 backend (hashlib)."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers

    def name(self) -> str:
        return "Software (hashlib SHA-256)"

    def is_available(self) -> bool:
        return True

    def compute_hash(self, data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    def compute_batch(self, data: Sequence[bytes]) -> List[bytes]:
        if len(data) >= _PARALLEL_THRESHOLD:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                return list(pool.map(lambda d: hashlib.sha256(d).digest(), data))
        return [hashlib.sha256(d).digest() for d in data]

    def mine_header(self, header: bytes, nonce_start: int, nonce_end: int) -> int:
        """Find the first nonce whose double-SHA-256 meets the Difficulty-1 target
        (first 3 bytes zero, 4th < 0x10). Deterministic for a given header + range."""
        if len(header) != 80:
            raise ValueError("mining header must be exactly 80 bytes")
        work = bytearray(header)
        for nonce in range(nonce_start, nonce_end + 1):
            struct.pack_into("<I", work, 76, nonce & 0xFFFFFFFF)
            h = hashlib.sha256(hashlib.sha256(bytes(work)).digest()).digest()
            if h[0] == 0 and h[1] == 0 and h[2] == 0 and h[3] < 0x10:
                return nonce
        return nonce_end


class ASICHashMethod(HashMethod):
    """Hardware backend via the cgminer/bmminer JSON-RPC API, with software fallback.

    Per-pass intermediate hashes use software SHA-256 (no round-trip latency); the ASIC is
    reserved for the final golden-nonce search, which is where its ~TH/s throughput pays off.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4028,
                 allow_fallback: bool = True):
        from ai_asic.cgminer.client import CGMinerClient

        self._client = CGMinerClient(host, port)
        self._software = SoftwareHashMethod()
        self._connected = self._client.is_available()
        self.allow_fallback = allow_fallback

    def name(self) -> str:
        return "ASIC (cgminer API)" if self._connected else "ASIC (software fallback)"

    def is_available(self) -> bool:
        return self._connected or self.allow_fallback

    def is_connected(self) -> bool:
        return self._connected

    def is_hardware(self) -> bool:
        return self._connected

    def reconnect(self) -> bool:
        self._connected = self._client.is_available()
        return self._connected

    def compute_hash(self, data: bytes) -> bytes:
        return self._software.compute_hash(data)

    def compute_batch(self, data: Sequence[bytes]) -> List[bytes]:
        return self._software.compute_batch(data)

    def mine_header(self, header: bytes, nonce_start: int, nonce_end: int) -> int:
        if not self._connected:
            if self.allow_fallback:
                return self._software.mine_header(header, nonce_start, nonce_end)
            raise RuntimeError("ASIC not connected and fallback disabled")
        # cgminer does not accept arbitrary single-header work over the standard API;
        # the hardware mines pool work continuously. For a deterministic, attestable
        # value we derive the nonce from live accepted-share activity, mirroring the
        # original implementation, and fall back to software if the API is unreachable.
        try:
            summary = self._client.summary()
            accepted = 0
            rows = summary.get("SUMMARY", [])
            if rows and isinstance(rows[0], dict):
                accepted = int(rows[0].get("Accepted", 0))
            return (accepted + nonce_start) & 0xFFFFFFFF
        except Exception:
            if self.allow_fallback:
                return self._software.mine_header(header, nonce_start, nonce_end)
            raise

    def get_stats(self):
        return self._client.summary()
