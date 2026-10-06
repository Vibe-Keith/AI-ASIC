"""AI workloads that run across the host CPU and an ASIC miner.

The models here are hash-based neural networks: every neuron is a SHA-256 operation, so the
one layer a mining ASIC can natively accelerate - a nonce search over a header - is the
layer this package offloads to the chip. The dense hash layers and the output head stay on
the host CPU. See ``split_model`` for the model and ``MiningBackend`` for device selection.
"""
from ai_asic.workloads.split_model import (
    ExecutionTrace,
    MiningBackend,
    SplitModel,
    Stage,
    WorkloadResult,
)

__all__ = [
    "SplitModel",
    "MiningBackend",
    "WorkloadResult",
    "ExecutionTrace",
    "Stage",
]
