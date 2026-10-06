"""ASIC device backends for the on-device hasher-server.

The original HASHER on-device server (``cmd/hasher-server``, a MIPS gRPC binary deployed onto
the miner) drove the chips through ``/dev/bitmain-asic``. This is the portable equivalent: an
``AsicDevice`` interface with two backends -

  * :class:`VirtualAsicDevice` - the simulated BM1387 chip from ``ai_asic.hardware.simulator``;
    runs everywhere, including Windows, with no hardware.
  * :class:`ChainAsicDevice` - the direct ``/dev/bitmain-asic`` chain path; live only on a
    miner's own Linux control board, dormant (``available == False``) elsewhere.

A BM13xx *is* a SHA-256 engine, so ``compute_hash``/``compute_batch`` (the service's documented
ops) map onto the chip's SHA-256 cores, while ``mine`` is the native nonce-search primitive.
Pure Python, standard library only.
"""
from __future__ import annotations

import hashlib
import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ai_asic.hardware import bm1387
from ai_asic.hashing import nonce_search
from ai_asic.hardware.miner_profiles import MinerProfile, default_profile, detect_profile

MAX_BATCH_SIZE = 256  # mirrors the original ComputeBatch cap
_MASK = 0xFFFFFFFF


@dataclass
class MineOutcome:
    nonce: int
    found: bool
    hash_hex: str
    leading_zeros: int
    hashes_tried: int
    midstate_index: int = 0


class AsicDevice(ABC):
    """What the hasher-server needs from whatever silicon (or model) is behind it."""

    profile: MinerProfile

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute_hash(self, data: bytes) -> bytes:
        """Single SHA-256 (the chip's hash core)."""

    def compute_batch(self, data: Sequence[bytes]) -> List[bytes]:
        if len(data) > MAX_BATCH_SIZE:
            raise ValueError(f"batch exceeds {MAX_BATCH_SIZE}")
        return [self.compute_hash(d) for d in data]

    @abstractmethod
    def mine(self, header: bytes, difficulty_bits: int, max_nonces: int = 1 << 20,
             start: int = 0) -> MineOutcome:
        """Native nonce search: find a nonce whose double-SHA-256 of ``header`` has at least
        ``difficulty_bits`` leading zero bits."""

    def mine_batch(self, jobs: Sequence[dict]) -> List[MineOutcome]:
        """Several native nonce searches in one request. Each job is a dict with ``header``
        (bytes) and optional ``difficulty_bits``, ``max_nonces``, ``start``."""
        if len(jobs) > MAX_BATCH_SIZE:
            raise ValueError(f"batch exceeds {MAX_BATCH_SIZE}")
        return [self.mine(j["header"], int(j.get("difficulty_bits", 16)),
                          max_nonces=int(j.get("max_nonces", 1 << 20)),
                          start=int(j.get("start", 0))) for j in jobs]

    def info(self) -> dict:
        p = self.profile
        return {
            "name": self.name(), "available": self.available,
            "model": p.model, "chip": p.chip, "process": p.process,
            "chip_count": p.chip_count, "board_count": p.board_count,
            "nominal_hashrate": p.nominal_hashrate,
            "frequency_mhz": p.default_frequency_mhz,
            "protocol": p.protocol.value,
        }


class VirtualAsicDevice(AsicDevice):
    """The simulated BM1387 chip.

    ``mine`` takes the work exactly as the chip does - a CRC-checked BM1387 chain frame
    carrying only the midstate and the 12-byte tail - and rolls the nonce from that midstate.
    By default the roll uses ``hashlib`` resumed from the midstate (``fast=True``), and the
    winning nonce is then re-derived through the cycle-honest midstate-only model
    (:func:`~ai_asic.hardware.simulator._finish_double_sha_from_midstate`); a mismatch raises.
    Results are identical to the pure-Python model (same first nonce, same hash, same count)
    at a few hundred times the speed. ``fast=False`` rolls every nonce through the model.
    """

    def __init__(self, model: str = "Antminer S9", fast: bool = True):
        self.profile = detect_profile(model) or default_profile()
        self.fast = fast

    @property
    def available(self) -> bool:
        return True

    def name(self) -> str:
        return f"virtual {self.profile.chip} ({self.profile.model})"

    def compute_hash(self, data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    def mine(self, header: bytes, difficulty_bits: int, max_nonces: int = 1 << 20,
             start: int = 0) -> MineOutcome:
        from ai_asic.hardware.simulator import SimConfig, VirtualBM1387

        if len(header) != 80:
            raise ValueError("header must be 80 bytes")
        if self.fast:
            return self._mine_fast(header, difficulty_bits, max_nonces, start)
        cfg = SimConfig(
            difficulty_bits=difficulty_bits,
            chip_count=self.profile.chip_count or 63,
            nominal_hashrate=float(self.profile.nominal_hashrate),
            max_nonces=max_nonces, start_nonce=start,
        )
        chip = VirtualBM1387(cfg)
        chip.load_work(bm1387.new_work_from_header(header, work_id=1).encode())
        r = chip.mine()
        return MineOutcome(r.nonce, r.found, r.hash_hex, r.leading_zeros,
                           r.hashes_tried, r.midstate_index)

    def _mine_fast(self, header: bytes, difficulty_bits: int, max_nonces: int,
                   start: int) -> MineOutcome:
        from ai_asic.hardware.simulator import _finish_double_sha_from_midstate

        # The chip's view of the work: a CRC-checked frame with the midstate and 12-byte tail.
        work = bm1387.decode_work(bm1387.new_work_from_header(header, work_id=1).encode(),
                                  verify_crc=True)
        tail = bytes(work.data)
        # hashlib resumed from header[:64] has exactly the frame's midstate as its state.
        nonce, digest, tried = nonce_search.search(header, difficulty_bits, start, max_nonces)
        if nonce is not None:
            if _finish_double_sha_from_midstate(work.midstates[0], tail, nonce) != digest:
                raise RuntimeError("midstate model disagrees with the nonce search")
            return MineOutcome(nonce & _MASK, True, digest.hex(),
                               nonce_search.leading_zero_bits(digest), tried, 0)
        end = start + max_nonces
        return MineOutcome(end & _MASK, False, (b"\x00" * 32).hex(), 0, tried, 0)


class ChainAsicDevice(AsicDevice):
    """Direct ``/dev/bitmain-asic`` chain backend (a miner's own Linux board).

    This is the real-hardware path the original embedded driver used. It is present so the
    hasher-server can run on an actual control board; off such a board the device node does
    not exist, ``available`` is ``False``, and the server reports the device as unavailable
    rather than pretending to mine.
    """

    def __init__(self, model: str = "Antminer S9",
                 device_path: str = "/dev/bitmain-asic", read_timeout: float = 5.0):
        self.profile = detect_profile(model) or default_profile()
        self.device_path = device_path
        self.read_timeout = read_timeout

    @property
    def available(self) -> bool:
        return os.path.exists(self.device_path)

    def name(self) -> str:
        return f"chain {self.profile.chip} via {self.device_path}"

    def _require(self) -> None:
        if not self.available:
            raise RuntimeError(
                f"ASIC chain device not present at {self.device_path}; this backend runs only "
                "on a miner's control board. Use VirtualAsicDevice off-board.")

    def compute_hash(self, data: bytes) -> bytes:
        # On the control board the SHA-256 is done by the board CPU (the chip cores are
        # dedicated to the nonce search); hashlib is that CPU path.
        self._require()
        return hashlib.sha256(data).digest()

    def mine(self, header: bytes, difficulty_bits: int, max_nonces: int = 1 << 20,
             start: int = 0) -> MineOutcome:
        self._require()
        if len(header) != 80:
            raise ValueError("header must be 80 bytes")
        # Push a BM1387 work frame to the chain and read back a nonce response. The chip-side
        # target is fixed by firmware; difficulty_bits is verified host-side on the returned
        # nonce. Real UART framing is firmware-specific; this is the canonical driver path.
        frame = bm1387.new_work_from_header(header, work_id=1).encode()
        with open(self.device_path, "r+b", buffering=0) as dev:
            dev.write(frame)
            resp = dev.read(6)
        res = bm1387.parse_nonce_response(resp)
        full = bytearray(header)
        struct.pack_into("<I", full, 76, res.nonce & _MASK)
        digest = hashlib.sha256(hashlib.sha256(bytes(full)).digest()).digest()
        value = int.from_bytes(digest, "big")
        lz = 256 - value.bit_length() if value else 256
        return MineOutcome(res.nonce, lz >= difficulty_bits, digest.hex(), lz, 0,
                           res.midstate_index)


def auto_device(model: str = "Antminer S9",
                device_path: str = "/dev/bitmain-asic") -> AsicDevice:
    """Pick the real chain device when its node exists, else the virtual chip."""
    chain = ChainAsicDevice(model, device_path)
    return chain if chain.available else VirtualAsicDevice(model)
