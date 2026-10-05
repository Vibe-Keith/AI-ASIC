"""Miner profile registry and model auto-detection.

Captures the fixed hardware characteristics of supported Bitmain ASIC miners (S1 through
S21) so the inference engine can target the right model and report accurate capabilities.
Values are nominal factory specifications; real chip counts and hash rates vary by batch,
firmware, and tuning.

Pure Python, cross-platform (Windows/macOS/Linux).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Connection(str, Enum):
    USB = "USB"          # direct USB bulk (Antminer S1/S2/S3)
    NETWORK = "Network"  # TCP/IP to the on-board controller (S5 and newer)
    UART = "UART"        # direct serial chain to the chips (direct-drive)


class Protocol(str, Enum):
    BM1382_USB = "bm1382-usb"      # S1/S2/S3 full 80-byte header over USB (0x52 TXTASK)
    BM1387_CHAIN = "bm1387-chain"  # S9 family / T9+ midstate + 12-byte tail chain protocol
    CGMINER_API = "cgminer-api"    # network control-board miners via cgminer/bmminer JSON-RPC


def _th(x: float) -> int:
    return int(x * 1e12)


def _gh(x: float) -> int:
    return int(x * 1e9)


@dataclass(frozen=True)
class MinerProfile:
    model: str
    aliases: Tuple[str, ...]
    chip: str
    process: str
    chip_count: int
    board_count: int
    nominal_hashrate: int  # hashes per second
    default_frequency_mhz: int
    connection: Connection
    protocol: Protocol
    default_device_path: Optional[str] = None
    api_port: int = 4028
    version_rolling: bool = False  # AsicBoost
    notes: str = ""

    def hardware_info(self) -> Dict[str, object]:
        device_path = self.default_device_path or f"cgminer-api:{self.api_port}"
        return {
            "device_path": device_path,
            "chip_count": self.chip_count,
            "version": self.chip,
            "connection_type": self.connection.value,
            "metadata": {
                "model": self.model,
                "chip": self.chip,
                "process": self.process,
                "protocol": self.protocol.value,
                "board_count": str(self.board_count),
                "frequency_mhz": str(self.default_frequency_mhz),
                "version_rolling": str(self.version_rolling).lower(),
            },
        }

    def capabilities(self, available: bool) -> Dict[str, object]:
        max_batch = self.chip_count if self.chip_count > 0 else 256
        return {
            "name": f"ASIC Hardware ({self.model})",
            "is_hardware": available,
            "hash_rate": self.nominal_hashrate if available else 0,
            "production_ready": available,
            "max_batch_size": max_batch,
            "avg_latency_us": 100,
            "hardware_info": self.hardware_info(),
        }


# Ordered most-specific-first so substring alias matching resolves the narrowest model
# (e.g. "S19 Pro" before "S19", "S9j" before "S9").
_REGISTRY: List[MinerProfile] = [
    # --- 5nm / 3nm flagship generation ---------------------------------------
    MinerProfile("Antminer S21", ("antminer s21", "s21"), "BM1370", "5nm", 0, 3,
                 _th(200), 0, Connection.NETWORK, Protocol.CGMINER_API,
                 version_rolling=True, notes="Chip count varies by sub-model; hash rate nominal."),
    MinerProfile("Antminer S19 XP", ("antminer s19 xp", "s19xp", "s19 xp"), "BM1366", "5nm", 0, 3,
                 _th(140), 0, Connection.NETWORK, Protocol.CGMINER_API,
                 version_rolling=True, notes="Chip count varies by sub-model; hash rate nominal."),
    # --- 7nm generation ------------------------------------------------------
    MinerProfile("Antminer S19j Pro", ("antminer s19j pro", "s19j pro", "s19jpro", "s19j"), "BM1398", "7nm", 0, 3,
                 _th(104), 0, Connection.NETWORK, Protocol.CGMINER_API,
                 version_rolling=True, notes="Chip count varies by sub-model; hash rate nominal."),
    MinerProfile("Antminer S19 Pro", ("antminer s19 pro", "s19 pro", "s19pro"), "BM1398", "7nm", 0, 3,
                 _th(110), 0, Connection.NETWORK, Protocol.CGMINER_API,
                 version_rolling=True, notes="Chip count varies by sub-model; hash rate nominal."),
    MinerProfile("Antminer S19", ("antminer s19", "s19"), "BM1398", "7nm", 0, 3,
                 _th(95), 0, Connection.NETWORK, Protocol.CGMINER_API,
                 version_rolling=True, notes="Chip count varies by sub-model; hash rate nominal."),
    MinerProfile("Antminer T17", ("antminer t17", "t17"), "BM1397", "7nm", 0, 3,
                 _th(40), 0, Connection.NETWORK, Protocol.CGMINER_API,
                 version_rolling=True, notes="Chip count varies by sub-model; hash rate nominal."),
    MinerProfile("Antminer S17", ("antminer s17", "s17"), "BM1397", "7nm", 0, 3,
                 _th(56), 0, Connection.NETWORK, Protocol.CGMINER_API,
                 version_rolling=True, notes="Chip count varies by sub-model; hash rate nominal."),
    MinerProfile("Antminer T15", ("antminer t15", "t15"), "BM1391", "7nm", 0, 2,
                 _th(23), 0, Connection.NETWORK, Protocol.CGMINER_API,
                 version_rolling=True, notes="Chip count varies by sub-model; hash rate nominal."),
    MinerProfile("Antminer S15", ("antminer s15", "s15"), "BM1391", "7nm", 0, 2,
                 _th(28), 0, Connection.NETWORK, Protocol.CGMINER_API,
                 version_rolling=True, notes="Chip count varies by sub-model; hash rate nominal."),
    # --- 16nm BM1387 generation (the classic "newer" miners) -----------------
    MinerProfile("Antminer T9+", ("antminer t9+", "t9+", "t9plus"), "BM1387", "16nm", 162, 3,
                 _th(10.5), 600, Connection.NETWORK, Protocol.CGMINER_API,
                 notes="162 chips (3 boards x 54)."),
    MinerProfile("Antminer S9k", ("antminer s9k", "s9k"), "BM1387", "16nm", 0, 3,
                 _th(13.5), 650, Connection.NETWORK, Protocol.CGMINER_API),
    MinerProfile("Antminer S9 SE", ("antminer s9 se", "s9se", "s9 se"), "BM1387", "16nm", 0, 3,
                 _th(16), 650, Connection.NETWORK, Protocol.CGMINER_API),
    MinerProfile("Antminer S9j", ("antminer s9j", "s9j"), "BM1387", "16nm", 189, 3,
                 _th(14.5), 650, Connection.NETWORK, Protocol.CGMINER_API,
                 notes="189 chips (3 boards x 63)."),
    MinerProfile("Antminer S9i", ("antminer s9i", "s9i"), "BM1387", "16nm", 189, 3,
                 _th(14), 650, Connection.NETWORK, Protocol.CGMINER_API,
                 notes="189 chips (3 boards x 63)."),
    MinerProfile("Antminer S9", ("antminer s9", "s9", "bm1387"), "BM1387", "16nm", 189, 3,
                 _th(13.5), 650, Connection.NETWORK, Protocol.CGMINER_API,
                 notes="189 chips (3 boards x 63). Direct-drive uses the BM1387 chain protocol."),
    # --- Earlier control-board generations -----------------------------------
    MinerProfile("Antminer S7", ("antminer s7", "s7", "bm1385"), "BM1385", "28nm", 162, 3,
                 _th(4.73), 600, Connection.NETWORK, Protocol.CGMINER_API,
                 notes="162 chips (3 boards x 54)."),
    MinerProfile("Antminer S5", ("antminer s5", "s5", "bm1384"), "BM1384", "28nm", 60, 1,
                 _gh(1155), 350, Connection.NETWORK, Protocol.CGMINER_API),
    # --- Original USB BM1382 generation (HASHER native target) ---------------
    MinerProfile("Antminer S3", ("antminer s3", "s3", "bm1382"), "BM1382", "28nm", 32, 2,
                 _gh(478), 250, Connection.USB, Protocol.BM1382_USB,
                 default_device_path="/dev/bitmain-asic",
                 notes="HASHER's original direct-USB target. 32 chips (2 boards x 16)."),
    MinerProfile("Antminer S2", ("antminer s2", "s2"), "BM1382", "28nm", 32, 2,
                 _th(1), 250, Connection.USB, Protocol.BM1382_USB,
                 default_device_path="/dev/bitmain-asic",
                 notes="HASHER's original direct-USB target."),
    MinerProfile("Antminer S1", ("antminer s1", "s1", "bm1380"), "BM1380", "55nm", 32, 2,
                 _gh(180), 350, Connection.USB, Protocol.BM1382_USB,
                 default_device_path="/dev/bitmain-asic"),
]

DEFAULT_PROFILE_MODEL = "Antminer S3"


def all_profiles() -> List[MinerProfile]:
    """Return the supported miner registry, most-specific-first."""
    return list(_REGISTRY)


def profile_by_model(name: str) -> Optional[MinerProfile]:
    """Return the profile whose canonical model name matches ``name`` (case-insensitive)."""
    n = (name or "").strip().lower()
    for p in _REGISTRY:
        if p.model.lower() == n:
            return p
    return None


def default_profile() -> MinerProfile:
    """Return the fallback profile (Antminer S3), preserving HASHER's original target."""
    return profile_by_model(DEFAULT_PROFILE_MODEL) or _REGISTRY[-1]


def detect_profile(hint: str) -> Optional[MinerProfile]:
    """Resolve a free-form hint (model name, cgminer "Type" string, chip name, or an
    ASIC_MODEL override) to a known profile via case-insensitive substring matching
    against aliases, in registry order (most-specific-first). Returns None if nothing
    matches."""
    h = (hint or "").strip().lower()
    if not h:
        return None
    for p in _REGISTRY:
        for a in p.aliases:
            if a and a in h:
                return p
    return None


def detect_model_hint(cgminer_host: Optional[str] = None, cgminer_port: int = 4028,
                      timeout: float = 2.0) -> str:
    """Best available model hint for the current host, in priority order:

      1. The ASIC_MODEL environment variable (explicit override).
      2. The miner "Type" reported by a reachable cgminer/bmminer API
         (CGMINER_HOST env or 127.0.0.1:4028).

    Returns an empty string if nothing is discoverable.
    """
    override = os.environ.get("ASIC_MODEL", "").strip()
    if override:
        return override

    host = cgminer_host or os.environ.get("CGMINER_HOST", "").strip() or "127.0.0.1"
    # Imported lazily to avoid a hard dependency when only using the registry.
    from ai_asic.cgminer.client import CGMinerClient

    try:
        client = CGMinerClient(host, cgminer_port, timeout=timeout)
        return client.detect_type() or ""
    except Exception:
        return ""
