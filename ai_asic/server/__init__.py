"""The on-device hasher-server, ported to pure Python.

A portable stand-in for the original HASHER MIPS gRPC on-device server: it exposes the same
HasherService methods (ComputeHash / ComputeBatch / Mine / GetMetrics / GetDeviceInfo) over
the project's stdlib JSON-over-TCP transport, backed by a pluggable ASIC device - the virtual
BM1387 chip everywhere, or a real ``/dev/bitmain-asic`` chain on a miner's own board.
"""
from ai_asic.server.client import HasherClient, RemoteMineResult
from ai_asic.server.device import (
    AsicDevice,
    ChainAsicDevice,
    MineOutcome,
    VirtualAsicDevice,
    auto_device,
)
from ai_asic.server.hasher_server import DEFAULT_PORT, HasherServer, Metrics

__all__ = [
    "HasherServer",
    "HasherClient",
    "RemoteMineResult",
    "Metrics",
    "AsicDevice",
    "VirtualAsicDevice",
    "ChainAsicDevice",
    "MineOutcome",
    "auto_device",
    "DEFAULT_PORT",
]
