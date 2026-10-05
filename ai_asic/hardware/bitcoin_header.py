"""80-byte Bitcoin block-header construction (the "camouflage" strategy).

Builds the headers pushed to BM1382-class ASICs, packing neural "slots" into the
prev-hash and merkle-root fields. Shared by the mining neuron and the direct-drive paths.
"""
from __future__ import annotations

import struct
import time
from typing import List, Sequence

BITCOIN_VERSION = 0x00000002
BITCOIN_BITS = 0x1D00FFFF  # Difficulty 1


def prepare_asic_job(slots: Sequence[int], candidate_nonce: int, timestamp: int = None) -> bytes:
    """Create an 80-byte Bitcoin header from 12 neural slots and a candidate nonce.

    Layout (matching the Go HardwarePrep.PrepareAsicJob):
      - bytes 0-3   version (LE)
      - bytes 4-35  slots 0-7 -> prev block hash (BE words)
      - bytes 36-67 slots 8-11 + zero padding -> merkle root (BE words)
      - bytes 68-71 timestamp (LE)
      - bytes 72-75 nBits (LE)
      - bytes 76-79 nonce (LE)
    """
    if len(slots) < 12:
        slots = list(slots) + [0] * (12 - len(slots))
    header = bytearray(80)
    struct.pack_into("<I", header, 0, BITCOIN_VERSION)
    for i in range(8):
        struct.pack_into(">I", header, 4 + i * 4, slots[i] & 0xFFFFFFFF)
    for i in range(4):
        struct.pack_into(">I", header, 36 + i * 4, slots[i + 8] & 0xFFFFFFFF)
    # bytes 52-67 already zero
    ts = int(time.time()) if timestamp is None else timestamp
    struct.pack_into("<I", header, 68, ts & 0xFFFFFFFF)
    struct.pack_into("<I", header, 72, BITCOIN_BITS)
    struct.pack_into("<I", header, 76, candidate_nonce & 0xFFFFFFFF)
    return bytes(header)


def extract_nonce(header: bytes) -> int:
    if len(header) < 80:
        return 0
    return struct.unpack_from("<I", header, 76)[0]


def extract_slots(header: bytes) -> List[int]:
    slots = [0] * 12
    if len(header) < 80:
        return slots
    for i in range(8):
        slots[i] = struct.unpack_from(">I", header, 4 + i * 4)[0]
    for i in range(4):
        slots[i + 8] = struct.unpack_from(">I", header, 36 + i * 4)[0]
    return slots


def validate_header(header: bytes) -> bool:
    if len(header) != 80:
        return False
    if struct.unpack_from("<I", header, 0)[0] != BITCOIN_VERSION:
        return False
    if struct.unpack_from("<I", header, 72)[0] != BITCOIN_BITS:
        return False
    return True
