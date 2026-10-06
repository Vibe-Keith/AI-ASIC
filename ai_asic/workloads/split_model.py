"""A hash-based classifier split across the host CPU and an ASIC miner.

Pipeline (three stages, two devices):

    input bytes
      -> [HOST]  encoder     : HashNetwork, dense SHA-256 hash layers on the CPU
      -> [ASIC]  mining layer : project features into block headers, mine a golden nonce
                                for each on the chip (the one op an ASIC accelerates)
      -> [HOST]  head         : SHA-256 hash layer -> class logits -> argmax

The mining layer is dispatched through :class:`MiningBackend`, which uses a running virtual
miner (``simulator.VirtualMinerServer``) when one is reachable - performing a real,
header-specific nonce search on the (simulated) chip - and otherwise falls back to an
identical software search on the host. Either way the result is the same; only the *device*
that did the nonce search changes, which is the whole point of the split. Every run returns
an :class:`ExecutionTrace` recording what ran where.

On genuinely stock hardware the plain cgminer API cannot mine an arbitrary header, so the
ASIC path here targets the virtual miner (the stand-in for the original project's on-device
mining server). Pure Python, standard library only.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ai_asic.hashing import nonce_search
from ai_asic.hashing.network import HashNetwork, float_slice_to_bytes
from ai_asic.hashing.neuron import HashNeuron, MiningNeuron, nonce_to_activation

_MASK = 0xFFFFFFFF
DEVICE_HOST = "host-cpu"
DEVICE_ASIC = "asic"


# --- execution trace ---------------------------------------------------------

@dataclass
class Stage:
    name: str
    device: str          # DEVICE_HOST or DEVICE_ASIC
    ops: int             # SHA-256 ops / nonce searches performed in this stage
    seconds: float
    detail: str = ""


@dataclass
class ExecutionTrace:
    stages: List[Stage] = field(default_factory=list)
    asic_label: str = ""          # human label for the device that ran the mining layer
    asic_is_hardware: bool = False  # True when a (simulated) miner, not software, mined

    def add(self, stage: Stage) -> None:
        self.stages.append(stage)

    @property
    def host_ops(self) -> int:
        return sum(s.ops for s in self.stages if s.device == DEVICE_HOST)

    @property
    def asic_ops(self) -> int:
        return sum(s.ops for s in self.stages if s.device == DEVICE_ASIC)

    @property
    def host_seconds(self) -> float:
        return sum(s.seconds for s in self.stages if s.device == DEVICE_HOST)

    @property
    def asic_seconds(self) -> float:
        return sum(s.seconds for s in self.stages if s.device == DEVICE_ASIC)


@dataclass
class WorkloadResult:
    prediction: int
    confidence: float
    logits: List[float]
    nonces: List[int]
    trace: ExecutionTrace


# --- mining backend (device selection for the ASIC layer) --------------------

@dataclass
class MineOutcome:
    nonce: int
    found: bool
    hash_hex: str
    leading_zeros: int
    hashes_tried: int


def _software_mine(header: bytes, difficulty_bits: int, max_nonces: int,
                   start: int = 0) -> MineOutcome:
    """Host-side nonce search to the same target the virtual chip uses (leading zero bits of
    the double-SHA-256), so host and ASIC paths are directly comparable."""
    nonce, digest, tried = nonce_search.search(header, difficulty_bits, start, max_nonces)
    if nonce is not None:
        return MineOutcome(nonce, True, digest.hex(), nonce_search.leading_zero_bits(digest),
                           tried)
    end = start + max_nonces
    return MineOutcome(end & _MASK, False, "", 0, tried)


class MiningBackend:
    """Chooses where the mining layer runs: a reachable on-device hasher-server (ASIC), else
    the host.

    Construct with a host/port to target a running
    :class:`~ai_asic.server.hasher_server.HasherServer`; omit them (or pass
    ``prefer_asic=False``) to force the software path.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 difficulty_bits: int = 16, max_nonces: int = 1 << 18,
                 prefer_asic: bool = True):
        self.difficulty_bits = difficulty_bits
        self.max_nonces = max_nonces
        self._client = None
        self.label = "host CPU (software nonce search)"
        self.is_hardware = False
        if prefer_asic and host and port:
            try:
                from ai_asic.server.client import HasherClient

                client = HasherClient(host, int(port))
                if client.is_available():
                    info = client.device_info()
                    self._client = client
                    self.is_hardware = True
                    self.label = (f"ASIC via hasher-server @ {host}:{port} "
                                  f"({info.get('chip', '?')})")
            except Exception:
                self._client = None

    @property
    def device(self) -> str:
        return DEVICE_ASIC if self.is_hardware else DEVICE_HOST

    def mine(self, header: bytes) -> MineOutcome:
        if self._client is not None:
            try:
                r = self._client.mine(header, self.difficulty_bits, max_nonces=self.max_nonces)
                return MineOutcome(r.nonce, r.found, r.hash_hex, r.leading_zeros,
                                   r.hashes_tried)
            except Exception:
                # Lost the miner mid-run: fall back to software so the workload still finishes.
                self.is_hardware = False
                self.label = "host CPU (software fallback; miner unreachable)"
                self._client = None
        return _software_mine(header, self.difficulty_bits, self.max_nonces)


# --- the model ---------------------------------------------------------------

class SplitModel:
    """A hash-based classifier whose mining layer offloads to an ASIC.

    Dimensions:
      * ``input_size`` / ``hidden1`` / ``hidden2`` / ``feature_size`` - the host encoder.
      * ``mining_neurons`` - headers mined per inference (the ASIC workload).
      * ``activation_width`` - activation values expanded from each mined nonce.
      * ``output_size`` - number of classes.
    """

    def __init__(self, input_size: int = 64, hidden1: int = 64, hidden2: int = 32,
                 feature_size: int = 16, mining_neurons: int = 4,
                 activation_width: int = 8, output_size: int = 10,
                 difficulty_bits: int = 16, nonce_range: int = 1 << 18,
                 encoder: Optional[HashNetwork] = None,
                 mining_layer: Optional[List[MiningNeuron]] = None,
                 head_seeds: Optional[List[bytes]] = None):
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.feature_size = feature_size
        self.mining_neurons = mining_neurons
        self.activation_width = activation_width
        self.output_size = output_size
        self.difficulty_bits = difficulty_bits
        self.nonce_range = nonce_range

        self.encoder = encoder or HashNetwork(input_size, hidden1, hidden2, feature_size)
        if mining_layer is not None:
            self.mining = mining_layer
        else:
            # Distinct salt per neuron so each mines a different header from the same features.
            self.mining = [
                MiningNeuron(input_dim=feature_size, output_dim=12,
                             salt=0x53480000 + i, nonce_start=0, nonce_end=nonce_range)
                for i in range(mining_neurons)
            ]
        import os
        self.head_seeds = head_seeds or [os.urandom(32) for _ in range(output_size)]
        self.head = [HashNeuron(s) for s in self.head_seeds]

    # -- inference ----------------------------------------------------------
    def infer(self, data: bytes, backend: Optional[MiningBackend] = None) -> WorkloadResult:
        backend = backend or MiningBackend(difficulty_bits=self.difficulty_bits,
                                           max_nonces=self.nonce_range)
        trace = ExecutionTrace(asic_label=backend.label, asic_is_hardware=backend.is_hardware)

        # Stage 1 - HOST: dense hash-layer encoder.
        t0 = time.perf_counter()
        features = self.encoder.forward(data)
        trace.add(Stage("encoder", DEVICE_HOST,
                        ops=self.hidden1 + self.hidden2 + self.feature_size,
                        seconds=time.perf_counter() - t0,
                        detail=f"{self.feature_size}-dim feature vector"))

        # Stage 2 - ASIC (or host fallback): mine one nonce per mining neuron.
        t0 = time.perf_counter()
        nonces: List[int] = []
        total_tried = 0
        for mn in self.mining:
            header = mn.build_header(features)
            outcome = backend.mine(header)
            nonces.append(outcome.nonce)
            total_tried += outcome.hashes_tried
        trace.add(Stage("mining", backend.device, ops=len(self.mining),
                        seconds=time.perf_counter() - t0,
                        detail=f"{len(self.mining)} headers, {total_tried:,} nonces rolled, "
                               f"difficulty {self.difficulty_bits} bits"))

        # Expand nonces into an activation vector (host, negligible).
        activations: List[float] = []
        for nonce in nonces:
            activations.extend(nonce_to_activation(nonce, self.activation_width))

        # Stage 3 - HOST: hash head -> class logits.
        t0 = time.perf_counter()
        head_in = float_slice_to_bytes(activations)
        logits = [n.forward(head_in) for n in self.head]
        trace.add(Stage("head", DEVICE_HOST, ops=self.output_size,
                        seconds=time.perf_counter() - t0,
                        detail=f"{self.output_size} class logits"))

        prediction = max(range(len(logits)), key=lambda i: logits[i])
        return WorkloadResult(prediction=prediction, confidence=logits[prediction],
                              logits=logits, nonces=nonces, trace=trace)

    # -- persistence --------------------------------------------------------
    def serialize(self) -> str:
        return json.dumps({
            "format": "ai-asic-split-model/1",
            "input_size": self.input_size, "hidden1": self.hidden1,
            "hidden2": self.hidden2, "feature_size": self.feature_size,
            "mining_neurons": self.mining_neurons, "activation_width": self.activation_width,
            "output_size": self.output_size, "difficulty_bits": self.difficulty_bits,
            "nonce_range": self.nonce_range,
            "encoder": json.loads(self.encoder.serialize()),
            "mining": [{
                "weights": mn.weights, "bias": mn.bias, "salt": mn.salt,
                "nonce_start": mn.nonce_start, "nonce_end": mn.nonce_end,
            } for mn in self.mining],
            "head_seeds": [s.hex() for s in self.head_seeds],
        }, indent=2)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.serialize())

    @classmethod
    def deserialize(cls, data: str) -> "SplitModel":
        d = json.loads(data)
        if d.get("format") != "ai-asic-split-model/1":
            raise ValueError("not an ai-asic split-model file")
        encoder = HashNetwork.deserialize(json.dumps(d["encoder"]))
        mining = []
        for m in d["mining"]:
            mn = MiningNeuron(input_dim=len(m["weights"][0]) if m["weights"] else 1,
                              output_dim=len(m["weights"]),
                              salt=m["salt"], nonce_start=m["nonce_start"],
                              nonce_end=m["nonce_end"])
            mn.weights = [[float(x) for x in row] for row in m["weights"]]
            mn.bias = [float(x) for x in m["bias"]]
            mining.append(mn)
        return cls(
            input_size=d["input_size"], hidden1=d["hidden1"], hidden2=d["hidden2"],
            feature_size=d["feature_size"], mining_neurons=d["mining_neurons"],
            activation_width=d["activation_width"], output_size=d["output_size"],
            difficulty_bits=d["difficulty_bits"], nonce_range=d["nonce_range"],
            encoder=encoder, mining_layer=mining,
            head_seeds=[bytes.fromhex(s) for s in d["head_seeds"]],
        )

    @classmethod
    def load(cls, path: str) -> "SplitModel":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.deserialize(fh.read())
