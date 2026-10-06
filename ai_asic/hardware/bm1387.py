"""BM1387 (Antminer S9 / S9i / S9j / T9+) work protocol.

Unlike the BM1382 (S2/S3) path, which pushes a full 80-byte Bitcoin header to the chip,
the BM1387 chain protocol pushes *pre-hashed* work:

  - The host computes the SHA-256 midstate over the first 64 bytes of the header
    (version + prev-hash + the first 28 bytes of the merkle root).
  - It sends that 32-byte midstate plus the remaining 12 bytes of pre-nonce header data
    (the 4-byte merkle-root tail + 4-byte ntime + 4-byte nbits).
  - The chip rolls the 4-byte nonce internally and reports nonces that meet target.

This mirrors how cgminer/bmminer frame work for the S9. The outer FPGA framing on the S9
control board is firmware-specific; this encoder produces the canonical, verifiable
payload (midstate + data + CRC). For most S9-class deployments work is driven through the
cgminer JSON-RPC API; this encoder exists for direct-drive / repurposing and for correct
work generation and nonce interpretation.

Pure Python, cross-platform (Windows/macOS/Linux).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

MIDSTATE_BYTES = 32
WORK_DATA_BYTES = 12  # merkle tail (4) + ntime (4) + nbits (4)
MAX_MIDSTATES = 4  # 4-way AsicBoost version rolling

# BM1387 chip-chain command tokens (CRC5-protected 5-byte command frames).
CMD_SET_ADDRESS = 0x00
CMD_SET_CONFIG = 0x08
CMD_CHAIN_INACTIVE = 0x05
CMD_READ_REG = 0x04

# --- SHA-256 single-block compression (for midstate) -------------------------
# Python's hashlib does not expose the intermediate state after a single block, so a
# standalone one-block compression function is implemented here to compute midstates.

_SHA256_INIT = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)

_SHA256_K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)

_MASK = 0xFFFFFFFF


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _MASK


def _sha256_compress(state: List[int], block: bytes) -> List[int]:
    """Compress one 64-byte block into state (SHA-256 round function)."""
    w = list(struct.unpack(">16I", block)) + [0] * 48
    for i in range(16, 64):
        s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & _MASK

    a, b, c, d, e, f, g, h = state
    for i in range(64):
        s1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = (h + s1 + ch + _SHA256_K[i] + w[i]) & _MASK
        s0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (s0 + maj) & _MASK
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & _MASK, c, b, a, (t1 + t2) & _MASK

    return [(x + y) & _MASK for x, y in zip(state, (a, b, c, d, e, f, g, h))]


def compute_midstate(block: bytes) -> bytes:
    """SHA-256 midstate: the 8-word state after compressing exactly one 64-byte block
    from the standard initial hash values, returned as 32 big-endian bytes.

    Some BM1387 firmwares expect the midstate words in reversed byte order; use
    ``swap_midstate_word_endianness`` to convert if your transport requires it.
    """
    if len(block) != 64:
        raise ValueError(f"midstate block must be 64 bytes, got {len(block)}")
    state = _sha256_compress(list(_SHA256_INIT), block)
    return struct.pack(">8I", *state)


def swap_midstate_word_endianness(ms: bytes) -> bytes:
    """Reverse the byte order within each 4-byte word of a 32-byte midstate."""
    if len(ms) != 32:
        raise ValueError("midstate must be 32 bytes")
    words = struct.unpack(">8I", ms)
    return struct.pack("<8I", *words)


# --- CRC implementations -----------------------------------------------------

def crc5(data: bytes, num_bits: int) -> int:
    """5-bit CRC used by Bitmain BM13xx command frames, ported from bmminer.

    ``num_bits`` is the number of leading bits of ``data`` to include (BM1387 commands
    CRC the first (len*8 - 5) bits, i.e. everything except the CRC field itself).
    """
    crcin = [1, 1, 1, 1, 1]
    for bit in range(num_bits):
        din = 1 if (data[bit // 8] & (0x80 >> (bit % 8))) else 0
        crcout = [
            crcin[4] ^ din,
            crcin[0],
            crcin[1] ^ crcin[4] ^ din,
            crcin[2],
            crcin[3],
        ]
        crcin = crcout
    crc = 0
    if crcin[4]:
        crc |= 0x10
    if crcin[3]:
        crc |= 0x08
    if crcin[2]:
        crc |= 0x04
    if crcin[1]:
        crc |= 0x02
    if crcin[0]:
        crc |= 0x01
    return crc


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def bm1387_command(cmd: int, param: int, addr: int, broadcast: bool = False) -> bytes:
    """Build a CRC5-protected 5-byte command frame for the BM1387 chain.

    Returns ``[cmd|type][len][param][addr][crc5]`` where the final byte's low 5 bits hold
    the CRC over the preceding 35 bits.
    """
    typ = cmd | (0x80 if broadcast else 0x00)
    frame = bytearray([typ, 0x05, param & 0xFF, addr & 0xFF, 0x00])
    frame[4] = crc5(bytes(frame), len(frame) * 8 - 5) & 0x1F
    return bytes(frame)


# --- Work item ---------------------------------------------------------------

@dataclass
class BM1387Work:
    """A single unit of pre-hashed work for the BM1387 chain."""

    work_id: int
    midstates: List[bytes] = field(default_factory=list)
    data: bytes = b"\x00" * WORK_DATA_BYTES

    def encode(self) -> bytes:
        """Serialize into a chain frame:

            [work_id:4 LE][midstate_0:32]...[midstate_n:32][data:12][crc16:2 BE]

        The CRC16 covers every preceding byte.
        """
        if not self.midstates:
            raise ValueError("work has no midstates")
        if len(self.midstates) > MAX_MIDSTATES:
            raise ValueError(f"too many midstates: {len(self.midstates)}")
        buf = bytearray(struct.pack("<I", self.work_id & _MASK))
        for ms in self.midstates:
            if len(ms) != MIDSTATE_BYTES:
                raise ValueError("each midstate must be 32 bytes")
            buf += ms
        if len(self.data) != WORK_DATA_BYTES:
            raise ValueError("data must be 12 bytes")
        buf += self.data
        buf += struct.pack(">H", crc16(bytes(buf)))
        return bytes(buf)


def decode_work(frame: bytes, *, verify_crc: bool = True) -> BM1387Work:
    """Inverse of :meth:`BM1387Work.encode`.

    Parses a chain frame ``[work_id:4 LE][midstate:32]*N[data:12][crc16:2 BE]`` back into a
    :class:`BM1387Work`. The number of midstates is inferred from the frame length. This is
    what receiving firmware (or the virtual chip in ``simulator.py``) does to recover the
    work it was handed. With ``verify_crc`` set, a corrupted frame raises ``ValueError`` -
    the same rejection a real chip performs before it will mine.
    """
    # 4 (work_id) + 32*N (midstates) + 12 (data) + 2 (crc16)
    body = len(frame) - 4 - WORK_DATA_BYTES - 2
    if body < MIDSTATE_BYTES or body % MIDSTATE_BYTES != 0:
        raise ValueError(f"malformed work frame: {len(frame)} bytes")
    n_midstates = body // MIDSTATE_BYTES
    if n_midstates > MAX_MIDSTATES:
        raise ValueError(f"too many midstates in frame: {n_midstates}")
    if verify_crc:
        want = struct.unpack(">H", frame[-2:])[0]
        got = crc16(frame[:-2])
        if want != got:
            raise ValueError(f"work frame CRC16 mismatch: got 0x{got:04x}, want 0x{want:04x}")
    work_id = struct.unpack("<I", frame[0:4])[0]
    midstates = [
        frame[4 + i * MIDSTATE_BYTES: 4 + (i + 1) * MIDSTATE_BYTES]
        for i in range(n_midstates)
    ]
    data = frame[4 + body: 4 + body + WORK_DATA_BYTES]
    return BM1387Work(work_id=work_id, midstates=midstates, data=data)


def new_work_from_header(header: bytes, work_id: int) -> BM1387Work:
    """Build a single-midstate work item from an 80-byte Bitcoin header. The nonce field
    (bytes 76-79) is ignored; the chip supplies the nonce."""
    if len(header) != 80:
        raise ValueError(f"header must be exactly 80 bytes, got {len(header)}")
    return BM1387Work(
        work_id=work_id,
        midstates=[compute_midstate(header[0:64])],
        data=header[64:76],
    )


def new_work_asicboost(header: bytes, work_id: int, versions: List[int]) -> BM1387Work:
    """Build a multi-midstate work item for AsicBoost version rolling. ``versions`` holds
    up to ``MAX_MIDSTATES`` block-version values; a midstate is computed for each by
    overwriting the header version field (bytes 0-3, little-endian) before hashing."""
    if len(header) != 80:
        raise ValueError(f"header must be exactly 80 bytes, got {len(header)}")
    if not versions:
        raise ValueError("at least one version is required")
    if len(versions) > MAX_MIDSTATES:
        raise ValueError(f"at most {MAX_MIDSTATES} versions supported, got {len(versions)}")
    block = bytearray(header[0:64])
    midstates = []
    for v in versions:
        block[0:4] = struct.pack("<I", v & _MASK)
        midstates.append(compute_midstate(bytes(block)))
    return BM1387Work(work_id=work_id, midstates=midstates, data=header[64:76])


@dataclass
class BM1387NonceResult:
    nonce: int
    work_id: int = 0
    midstate_index: int = 0


def encode_nonce_response(nonce: int, work_id: int = 0, midstate_index: int = 0) -> bytes:
    """Build the 6-byte nonce response a BM1387 chip returns up the chain:

        [nonce:4 BE][work_id:1][midstate_index/chip:1]

    Inverse of :func:`parse_nonce_response`. The virtual chip in ``simulator.py`` uses this
    to report a golden nonce exactly as real firmware frames it.
    """
    return struct.pack(">IBB", nonce & _MASK, work_id & 0xFF, midstate_index & 0x03)


def parse_nonce_response(frame: bytes) -> BM1387NonceResult:
    """Decode a BM1387 nonce response frame:

        [nonce:4 BE][work_id:1][midstate_index/chip:1][crc5-in-low-bits:1]

    Extracts the nonce and the work/midstate identifiers needed to match a nonce back to
    its work item. The exact high bits of the trailing bytes are firmware-specific.
    """
    if len(frame) < 5:
        raise ValueError(f"nonce response too short: {len(frame)} bytes")
    res = BM1387NonceResult(nonce=struct.unpack(">I", frame[0:4])[0])
    if len(frame) >= 6:
        res.work_id = frame[4]
        res.midstate_index = frame[5] & 0x03
    return res
