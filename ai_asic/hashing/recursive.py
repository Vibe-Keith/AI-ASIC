"""Recursive single-ASIC inference engine.

Runs the network over N temporal passes (default 21) with per-pass input jitter and
optional seed rotation, then forms a temporal consensus by majority vote. When a
HashMethod is supplied the per-layer hashing is batched through it (hardware or software).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ai_asic.hashing.methods import HashMethod
from ai_asic.hashing.network import HashNetwork, float_slice_to_bytes

_U64 = (1 << 64) - 1


@dataclass
class InferencePass:
    pass_number: int
    prediction: int
    confidence: float
    pass_latency_s: float


@dataclass
class ConsensusResult:
    prediction: int
    confidence: float
    average_confidence: float
    vote_count: int


@dataclass
class RecursiveResult:
    passes: List[InferencePass]
    consensus: ConsensusResult
    latency_s: float
    valid_passes: int
    total_passes: int

    def statistical_summary(self) -> Dict[str, object]:
        confs = [p.confidence for p in self.passes]
        n = len(confs) or 1
        mean = sum(confs) / n
        var = sum((c - mean) ** 2 for c in confs) / n
        dist: Dict[int, int] = {}
        for p in self.passes:
            dist[p.prediction] = dist.get(p.prediction, 0) + 1
        return {"mean_confidence": mean, "confidence_variance": var, "class_distribution": dist}


class RecursiveEngine:
    def __init__(self, network: HashNetwork, passes: int = 21, jitter: float = 0.01,
                 seed_rotation: bool = False, hash_method: Optional[HashMethod] = None):
        if network is None:
            raise ValueError("invalid network configuration")
        self.network = network
        self.passes = passes if passes > 0 else 21
        self.jitter = jitter if 0 <= jitter <= 1 else 0.01
        self.seed_rotation = seed_rotation
        self.hash_method = hash_method

    def set_hash_method(self, method: HashMethod) -> None:
        self.hash_method = method

    def is_using_hardware(self) -> bool:
        try:
            return self.hash_method is not None and self.hash_method.is_hardware()
        except Exception:
            return False

    def infer(self, input_bytes: bytes) -> RecursiveResult:
        start = time.monotonic()
        results: List[InferencePass] = []
        for i in range(self.passes):
            try:
                results.append(self._run_pass(input_bytes, i))
            except Exception:
                continue
        if not results:
            raise RuntimeError("no valid passes completed")
        consensus = self._aggregate(results)
        return RecursiveResult(
            passes=results,
            consensus=consensus,
            latency_s=time.monotonic() - start,
            valid_passes=len(results),
            total_passes=self.passes,
        )

    def _run_pass(self, input_bytes: bytes, pass_num: int) -> InferencePass:
        start = time.monotonic()
        jittered = _apply_jitter(input_bytes, self.jitter, pass_num)

        if self.hash_method is not None and self.hash_method.is_available():
            prediction, confidence = self._run_hardware(jittered, pass_num)
        else:
            net = self._rotate_seeds(pass_num) if self.seed_rotation else self.network
            prediction, confidence = net.predict(jittered)

        return InferencePass(pass_num, prediction, confidence, time.monotonic() - start)

    def _run_hardware(self, input_bytes: bytes, pass_num: int):
        net = self._rotate_seeds(pass_num) if self.seed_rotation else self.network

        l1 = self.hash_method.compute_batch(_layer_inputs(input_bytes, net.seeds1))
        l1f = _hashes_to_floats(l1)
        l2 = self.hash_method.compute_batch(_layer_inputs(float_slice_to_bytes(l1f), net.seeds2))
        l2f = _hashes_to_floats(l2)
        out = self.hash_method.compute_batch(_layer_inputs(float_slice_to_bytes(l2f), net.seeds_out))
        outf = _hashes_to_floats(out)

        max_idx, max_val = 0, outf[0]
        for i, v in enumerate(outf[1:], start=1):
            if v > max_val:
                max_val, max_idx = v, i
        return max_idx, max_val

    def _rotate_seeds(self, pass_num: int) -> HashNetwork:
        def rot(seeds):
            return [bytes((b ^ ((pass_num + i) % 256)) for i, b in enumerate(s)) for s in seeds]

        return HashNetwork(
            self.network.input_size, self.network.hidden1, self.network.hidden2,
            self.network.output_size,
            seeds1=rot(self.network.seeds1),
            seeds2=rot(self.network.seeds2),
            seeds_out=rot(self.network.seeds_out),
        )

    def _aggregate(self, passes: List[InferencePass]) -> ConsensusResult:
        votes: Dict[int, int] = {}
        mode, max_votes = -1, 0
        for p in passes:
            votes[p.prediction] = votes.get(p.prediction, 0) + 1
            if votes[p.prediction] > max_votes:
                max_votes, mode = votes[p.prediction], p.prediction
        confidence = max_votes / len(passes)
        avg = sum(p.confidence for p in passes) / len(passes)
        return ConsensusResult(mode, confidence, avg, len(passes))


def _apply_jitter(input_bytes: bytes, jitter: float, seed: int) -> bytes:
    if jitter == 0:
        return input_bytes
    rng = random.Random(seed)
    out = bytearray(input_bytes)
    for i in range(len(out)):
        delta = int(rng.random() * jitter * 255) - int(rng.random() * jitter * 255)
        out[i] = min(255, max(0, out[i] + delta))
    return bytes(out)


def _layer_inputs(input_bytes: bytes, seeds: List[bytes]) -> List[bytes]:
    return [input_bytes + s for s in seeds]


def _hashes_to_floats(hashes: List[bytes]) -> List[float]:
    out = []
    for h in hashes:
        val = (h[0] << 56 | h[1] << 48 | h[2] << 40 | h[3] << 32
               | h[4] << 24 | h[5] << 16 | h[6] << 8 | h[7])
        out.append(val / _U64)
    return out
