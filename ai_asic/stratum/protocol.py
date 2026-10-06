"""Stratum v1 wire conventions and the seal proof format.

A stock miner (S9 bmminer, and every later Bitmain/cgminer-derived firmware) only takes work
from a pool. It never sees an 80-byte header; it gets a *job* and builds headers itself:

    coinbase    = coinb1 || extranonce1 || extranonce2 || coinb2
    merkle_root = SHA256d(coinbase)                  (no other transactions: no branches)
    header      = version | prevhash | merkle_root | ntime | nbits | nonce

So the host commits data to the chips through the coinbase: the turn's 32-byte digest is a push
in the coinbase script, the header's merkle root commits to the coinbase, and a share - a nonce
whose header hash meets the share target - is proof-of-work over that digest. ``prevhash`` is
the previous transcript digest, so the header also commits to the chain.

Byte orders follow cgminer/bmminer exactly (the de-facto standard every pool implements):

  * version / ntime / nbits are sent as big-endian hex; the header holds them little-endian.
  * prevhash is sent with each 4-byte word byte-reversed relative to the header's bytes.
  * the submitted nonce and ntime hex are the header bytes reversed.
  * a hash meets a target when ``int.from_bytes(hash, "little") <= target``.
"""
from __future__ import annotations

import hashlib
import struct
from fractions import Fraction
from typing import Optional, Tuple

DIFF1_TARGET = 0xFFFF << 208          # "difficulty 1" in pool (bdiff) terms
MAX_TARGET = (1 << 256) - 1
VERSION = 0x20000000
# Real-network-like bits so firmware never mistakes a share for a "found block". Seals are
# judged against the share target and the requested bits, never against nbits.
NBITS = 0x17034219
DEFAULT_VERSION_MASK = 0x1FFFE000      # BIP 320 general-purpose bits (AsicBoost rolling)
SCRIPT_TAG = b"ai-asic seal"


def dsha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def hash_value(digest: bytes) -> int:
    return int.from_bytes(digest, "little")


def leading_zero_bits(digest: bytes) -> int:
    """Leading zero bits of a header hash read the way miners and pools read it."""
    v = hash_value(digest)
    return 256 - v.bit_length() if v else 256


def target_for_difficulty(difficulty: float) -> int:
    """Share target for a (possibly fractional) pool difficulty."""
    if difficulty <= 0:
        raise ValueError("difficulty must be positive")
    f = Fraction(difficulty)
    return min(MAX_TARGET, DIFF1_TARGET * f.denominator // f.numerator)


def bits_for_difficulty(difficulty: float) -> int:
    """Leading zero bits every share at this difficulty is guaranteed to have."""
    return 256 - target_for_difficulty(difficulty).bit_length()


def swap_words(data: bytes) -> bytes:
    """Byte-reverse each 4-byte word (the prevhash and cgminer ``flip`` convention)."""
    return b"".join(data[i:i + 4][::-1] for i in range(0, len(data), 4))


def _push(data: bytes) -> bytes:
    if len(data) > 75:
        raise ValueError("push too long")
    return bytes([len(data)]) + data


def _varint(n: int) -> bytes:
    if n < 0xFD:
        return bytes([n])
    return b"\xfd" + struct.pack("<H", n)


def build_coinbase(payload: bytes, height: int, extranonce_size: int,
                   tag: bytes = SCRIPT_TAG) -> Tuple[bytes, bytes]:
    """A well-formed coinbase transaction committing to ``payload`` (32 bytes), split around
    the extranonce: returns ``(coinb1, coinb2)``."""
    if len(payload) != 32:
        raise ValueError("payload must be 32 bytes")
    script_head = (_push(struct.pack("<I", height & 0xFFFFFF)[:3])   # BIP 34-style height
                   + _push(payload) + _push(tag) + bytes([extranonce_size]))
    script_len = len(script_head) + extranonce_size
    coinb1 = (struct.pack("<I", 1) + b"\x01" + b"\x00" * 32 + b"\xff\xff\xff\xff"
              + _varint(script_len) + script_head)
    out_script = b"\x6a" + _push(tag)                                 # OP_RETURN, 0 value
    coinb2 = (b"\xff\xff\xff\xff" + b"\x01" + struct.pack("<q", 0)
              + _varint(len(out_script)) + out_script + struct.pack("<I", 0))
    return coinb1, coinb2


def coinbase_commits_to(coinbase: bytes, payload: bytes) -> bool:
    return _push(payload) in coinbase


def header_from_share(version: int, prev: bytes, merkle_root: bytes, ntime_hex: str,
                      nbits: int, nonce_hex: str) -> bytes:
    """Rebuild the 80-byte header a miner hashed for a submitted share."""
    return (struct.pack("<I", version & 0xFFFFFFFF) + prev + merkle_root
            + bytes.fromhex(ntime_hex)[::-1] + struct.pack("<I", nbits)
            + bytes.fromhex(nonce_hex)[::-1])


def rolled_version(job_version: int, version_bits_hex: Optional[str], mask: int) -> int:
    if not version_bits_hex or not mask:
        return job_version
    return (job_version & ~mask) | (int(version_bits_hex, 16) & mask)


def verify_seal(prev: bytes, digest: bytes, coinbase: bytes, header: bytes,
                difficulty_bits: int) -> Optional[str]:
    """Check a stratum seal proof with ``hashlib`` alone. Returns a problem, or ``None``."""
    if len(header) != 80:
        return "header is not 80 bytes"
    if not coinbase_commits_to(coinbase, digest):
        return "coinbase does not commit to the turn digest"
    if header[36:68] != dsha256(coinbase):
        return "header merkle root does not match the coinbase"
    if header[4:36] != prev:
        return "header does not chain to the previous digest"
    if leading_zero_bits(dsha256(header)) < difficulty_bits:
        return "proof-of-work does not meet its difficulty"
    return None
