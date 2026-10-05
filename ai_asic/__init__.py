"""AI-ASIC: SHA-256 hash neural-network inference on repurposed Bitcoin mining hardware.

A pure-Python, cross-platform (Windows/macOS/Linux) port of the HASHER engine, extended
with support for newer Bitmain ASIC miners (Antminer S9 family and beyond).
"""
from __future__ import annotations

__version__ = "0.1.0"

from ai_asic.hardware import miner_profiles  # noqa: F401
from ai_asic.hardware.device_detector import detect_asic, detection_summary  # noqa: F401
from ai_asic.hashing.network import HashNetwork  # noqa: F401
from ai_asic.hashing.recursive import RecursiveEngine  # noqa: F401
from ai_asic.hashing.methods import SoftwareHashMethod, ASICHashMethod  # noqa: F401

__all__ = [
    "miner_profiles",
    "detect_asic",
    "detection_summary",
    "HashNetwork",
    "RecursiveEngine",
    "SoftwareHashMethod",
    "ASICHashMethod",
    "__version__",
]
