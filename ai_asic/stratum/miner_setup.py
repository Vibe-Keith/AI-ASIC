"""Pointing a miner at the host's private pool, over the miner's own cgminer/bmminer API.

Uses only the miner's LAN API (TCP 4028): ``addpool`` then ``switchpool``. Those are privileged
commands; stock S9 firmware allows them from the LAN (``api-allow W:0/0``), some firmwares do
not - then set the pool in the miner's web UI instead (Miner Configuration -> Pool 1).
"""
from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional, Tuple

from ai_asic.cgminer.client import CGMinerClient


def lan_ip_towards(peer: str) -> str:
    """This host's address on the interface that reaches ``peer`` (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer, 4028))
        return s.getsockname()[0]
    finally:
        s.close()


def guess_lan_ip() -> str:
    try:
        return lan_ip_towards("192.168.0.1")  # any private address picks the LAN interface
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def _status(resp: Dict[str, Any]) -> Tuple[bool, str]:
    st = (resp.get("STATUS") or [{}])[0]
    return st.get("STATUS") in ("S", "I"), str(st.get("Msg", ""))


def list_pools(client: CGMinerClient) -> List[Dict[str, Any]]:
    return [p for p in client.send_command("pools").get("POOLS", []) if isinstance(p, dict)]


def point_miner_at(miner_host: str, pool_url: str, worker: str = "ai-asic",
                   password: str = "x", api_port: int = 4028) -> str:
    """Add ``pool_url`` to the miner and switch to it. Returns a status line; raises
    ``RuntimeError`` with guidance when the API refuses."""
    client = CGMinerClient(miner_host, api_port, timeout=10.0)
    pools = list_pools(client)
    match = next((p for p in pools if str(p.get("URL", "")).rstrip("/") == pool_url), None)
    if match is None:
        ok, msg = _status(client.send_command("addpool", f"{pool_url},{worker},{password}"))
        if not ok:
            raise RuntimeError(
                f"miner refused addpool ({msg or 'no message'}). Its API may not allow "
                f"privileged commands from this host; set Pool 1 to {pool_url} in the miner's "
                "web UI instead.")
        pools = list_pools(client)
        match = next((p for p in pools if str(p.get("URL", "")).rstrip("/") == pool_url), None)
        if match is None:
            raise RuntimeError("pool was added but does not appear in the miner's pool list")
    ok, msg = _status(client.send_command("switchpool", str(match.get("POOL", 0))))
    if not ok:
        raise RuntimeError(f"miner refused switchpool ({msg or 'no message'})")
    return f"miner {miner_host} switched to pool {match.get('POOL')} -> {pool_url}"


def other_active_pools(miner_host: str, pool_url: str, api_port: int = 4028) -> Optional[List[str]]:
    """URLs of the miner's other enabled pools (failover targets outside the closed loop), or
    ``None`` if the API is unreachable."""
    try:
        pools = list_pools(CGMinerClient(miner_host, api_port, timeout=5.0))
    except Exception:
        return None
    return [str(p.get("URL")) for p in pools
            if str(p.get("URL", "")).rstrip("/") != pool_url
            and str(p.get("Status", "")).lower() in ("alive", "enabled", "")]
