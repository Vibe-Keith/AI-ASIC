"""Hash-based neurons.

- HashNeuron: SHA-256(input || seed) normalized to [0, 1], the activation function.
- MiningNeuron: activation is the first valid nonce found by mining a header built from
  input projections (deterministic for a given header + nonce range).
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import List, Optional, Sequence

from ai_asic.hashing.methods import HashMethod, mine_difficulty1

DIFFICULTY1_NBITS = 0x1D00FFFF
GOLDEN_RATIO = 2654435769
_MASK = 0xFFFFFFFF
_U64 = (1 << 64) - 1


class HashNeuron:
    def __init__(self, seed: bytes, output_mode: str = "float"):
        if len(seed) != 32:
            raise ValueError("seed must be 32 bytes")
        self.seed = seed
        self.output_mode = output_mode or "float"

    def forward(self, input_bytes: bytes) -> float:
        h = hashlib.sha256(input_bytes + self.seed).digest()
        val = struct.unpack(">Q", h[0:8])[0]
        return val / _U64

    def __repr__(self) -> str:
        return "HashNeuron"


class MiningNeuron:
    """Hash neuron whose activation is a mined nonce. Weights project the input into the
    header's prev-hash / merkle-root fields."""

    def __init__(self, input_dim: int, output_dim: int, salt: int = 0,
                 nonce_start: int = 0, nonce_end: int = 0xFFFF,
                 hash_method: Optional[HashMethod] = None):
        # Xavier-ish deterministic init (matches the Go reference).
        self.weights: List[List[float]] = [
            [((i + j) * 0.01) - 0.005 for j in range(input_dim)]
            for i in range(output_dim)
        ]
        self.bias: List[float] = [0.0] * output_dim
        self.salt = salt & _MASK
        self.nonce_start = nonce_start & _MASK
        self.nonce_end = nonce_end & _MASK
        self.hash_method = hash_method

    def set_hash_method(self, method: HashMethod) -> None:
        self.hash_method = method

    def forward(self, input_vec: Sequence[float]) -> int:
        projections = self._compute_projections(input_vec)
        header = self._build_header(projections)
        return self._mine(header)

    def build_header(self, input_vec: Sequence[float]) -> bytes:
        """Project ``input_vec`` into the 80-byte header this neuron would mine, without
        mining it. Lets an external backend (e.g. an ASIC) perform the nonce search."""
        return self._build_header(self._compute_projections(input_vec))

    def _compute_projections(self, input_vec: Sequence[float]) -> List[float]:
        out = []
        for i, row in enumerate(self.weights):
            s = self.bias[i]
            for j in range(min(len(input_vec), len(row))):
                s += row[j] * input_vec[j]
            out.append(s)
        return out

    def _build_header(self, projections: Sequence[float]) -> bytes:
        header = bytearray(80)
        struct.pack_into("<I", header, 0, self.salt)
        for i in range(min(8, len(projections))):
            struct.pack_into("<I", header, 4 + i * 4, _float32_bits(projections[i]))
        for i in range(8):
            if i + 8 < len(projections):
                struct.pack_into("<I", header, 36 + i * 4, _float32_bits(projections[i + 8]))
        struct.pack_into("<I", header, 68, self.salt)
        struct.pack_into("<I", header, 72, DIFFICULTY1_NBITS)
        struct.pack_into("<I", header, 76, self.nonce_start)
        return bytes(header)

    def _mine(self, header: bytes) -> int:
        if self.hash_method is not None and self.hash_method.is_available():
            try:
                return self.hash_method.mine_header(header, self.nonce_start, self.nonce_end)
            except Exception:
                pass
        return self._mine_software(header)

    def _mine_software(self, header: bytes) -> int:
        return mine_difficulty1(header, self.nonce_start, self.nonce_end)


def _float32_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


def normalize_nonce(nonce: int, nonce_range: int) -> float:
    return nonce / nonce_range if nonce_range else 0.0


def nonce_to_activation(nonce: int, num_outputs: int) -> List[float]:
    acts = []
    for i in range(num_outputs):
        shift = i % 32
        rotated = ((nonce >> shift) | (nonce << (32 - shift))) & _MASK if shift else nonce & _MASK
        mixed = rotated ^ ((i * GOLDEN_RATIO) & _MASK)
        acts.append(mixed / _MASK)
    return acts
