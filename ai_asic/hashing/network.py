"""Hash-based neural network: input -> hidden1 -> hidden2 -> output, each neuron a
SHA-256 hash neuron with a cryptographic seed "weight"."""
from __future__ import annotations

import json
import os
import struct
from typing import List, Tuple

from ai_asic.hashing.neuron import HashNeuron

_U64 = (1 << 64) - 1


def _float64_to_uint64(f: float) -> int:
    f = min(1.0, max(0.0, f))
    return int(f * _U64)


def float_slice_to_bytes(floats: List[float]) -> bytes:
    return b"".join(struct.pack(">Q", _float64_to_uint64(f)) for f in floats)


class HashNetwork:
    def __init__(self, input_size: int, hidden1: int, hidden2: int, output_size: int,
                 seeds1=None, seeds2=None, seeds_out=None):
        if min(input_size, hidden1, hidden2, output_size) <= 0:
            raise ValueError("all network dimensions must be positive")
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        self.seeds1 = seeds1 or [os.urandom(32) for _ in range(hidden1)]
        self.seeds2 = seeds2 or [os.urandom(32) for _ in range(hidden2)]
        self.seeds_out = seeds_out or [os.urandom(32) for _ in range(output_size)]
        self.neurons1 = [HashNeuron(s) for s in self.seeds1]
        self.neurons2 = [HashNeuron(s) for s in self.seeds2]
        self.neurons_out = [HashNeuron(s) for s in self.seeds_out]

    def forward(self, input_bytes: bytes) -> List[float]:
        layer1 = [n.forward(input_bytes) for n in self.neurons1]
        layer2_in = float_slice_to_bytes(layer1)
        layer2 = [n.forward(layer2_in) for n in self.neurons2]
        out_in = float_slice_to_bytes(layer2)
        return [n.forward(out_in) for n in self.neurons_out]

    def predict(self, input_bytes: bytes) -> Tuple[int, float]:
        output = self.forward(input_bytes)
        max_idx = 0
        max_val = output[0]
        for i, v in enumerate(output[1:], start=1):
            if v > max_val:
                max_val = v
                max_idx = i
        return max_idx, max_val

    def serialize(self) -> str:
        return json.dumps({
            "input_size": self.input_size,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "output_size": self.output_size,
            "seeds1": [s.hex() for s in self.seeds1],
            "seeds2": [s.hex() for s in self.seeds2],
            "seeds_out": [s.hex() for s in self.seeds_out],
        })

    @classmethod
    def deserialize(cls, data: str) -> "HashNetwork":
        d = json.loads(data)
        return cls(
            d["input_size"], d["hidden1"], d["hidden2"], d["output_size"],
            seeds1=[bytes.fromhex(s) for s in d["seeds1"]],
            seeds2=[bytes.fromhex(s) for s in d["seeds2"]],
            seeds_out=[bytes.fromhex(s) for s in d["seeds_out"]],
        )
