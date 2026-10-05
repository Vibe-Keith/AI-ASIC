"""Model-aware ASIC detection.

Resolves the connected miner from (in priority order) the ASIC_MODEL override, a reachable
cgminer/bmminer API, or the presence of a direct-USB device node, then reports capabilities
from that model's profile. Replaces the original hard-coded Antminer S3 (BM1382) assumption
so newer miners such as the S9 (BM1387) and later are detected correctly.

Cross-platform: on Windows there is no /dev device node, so USB models simply report as
unavailable and network miners are reached over the cgminer API.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from ai_asic.cgminer.client import CGMinerClient
from ai_asic.hardware.miner_profiles import (
    Connection,
    MinerProfile,
    default_profile,
    detect_model_hint,
    detect_profile,
)


def detect_asic(cgminer_host: Optional[str] = None, cgminer_port: int = 4028) -> Dict[str, object]:
    """Detect the attached ASIC and return a capabilities dict.

    The returned dict includes ``available`` (bool), the resolved ``profile``, and the
    capability fields (name, hash_rate, chip info, etc.).
    """
    host = cgminer_host or os.environ.get("CGMINER_HOST", "").strip() or "127.0.0.1"
    hint = detect_model_hint(host, cgminer_port)
    profile = detect_profile(hint)
    detected_by = f"model_hint:{hint!r}" if profile and hint else "profile_default"
    if profile is None:
        profile = default_profile()

    if profile.connection == Connection.USB:
        available, reason = _probe_usb(profile.default_device_path)
    else:
        available = _cgminer_reachable(host, cgminer_port)
        reason = ""
        if not available:
            usb_ok, _ = _probe_usb(profile.default_device_path)
            if usb_ok:
                available = True
            else:
                reason = ("cgminer/bmminer API not reachable "
                          "(is the miner powered and on the network?)")

    caps = profile.capabilities(available)
    if not available:
        caps["hash_rate"] = 0
        caps["reason"] = reason
    info = caps.get("hardware_info")
    if isinstance(info, dict):
        info.setdefault("metadata", {})["detected_by"] = detected_by
    caps["available"] = available
    caps["profile"] = profile
    return caps


def _probe_usb(device_path: Optional[str]):
    path = device_path or "/dev/bitmain-asic"
    if not os.path.exists(path):
        return False, f"Device not found: {path}"
    try:
        with open(path, "rb"):
            pass
    except OSError as exc:
        return False, f"Cannot access device: {exc} (is CGMiner running?)"
    return True, ""


def _cgminer_reachable(host: str, port: int) -> bool:
    try:
        return CGMinerClient(host, port, timeout=2.0).is_available()
    except Exception:
        return False


def detection_summary(cgminer_host: Optional[str] = None) -> str:
    caps = detect_asic(cgminer_host)
    profile: MinerProfile = caps["profile"]  # type: ignore[assignment]
    status = "AVAILABLE" if caps["available"] else "UNAVAILABLE"
    lines = [
        "ASIC Detection Summary",
        "======================",
        f"Model:       {profile.model}",
        f"Chip:        {profile.chip} ({profile.process})",
        f"Protocol:    {profile.protocol.value}",
        f"Connection:  {profile.connection.value}",
        f"Chips:       {profile.chip_count if profile.chip_count else 'varies'}",
        f"Hash rate:   {caps['hash_rate'] / 1e12:.3f} TH/s (nominal)",
        f"Status:      {status}",
    ]
    if not caps["available"] and caps.get("reason"):
        lines.append(f"Reason:      {caps['reason']}")
    return "\n".join(lines)
