"""Closed-loop host <-> miner transport: a private Stratum v1 pool on the host.

Stock miner firmware takes work only from a pool, so this is how the host reaches real chips
with no outside servers: the miner connects to this host on the LAN and mines its jobs.
"""
from ai_asic.stratum.pool import (
    DEFAULT_SHARE_DIFFICULTY,
    DEFAULT_STRATUM_PORT,
    MinerSession,
    PoolShare,
    StratumPool,
)
from ai_asic.stratum.virtual_miner import VirtualStratumMiner

__all__ = [
    "StratumPool",
    "PoolShare",
    "MinerSession",
    "VirtualStratumMiner",
    "DEFAULT_STRATUM_PORT",
    "DEFAULT_SHARE_DIFFICULTY",
]
