"""cgminer / bmminer JSON-RPC client.

This is the cross-platform (including Windows) path to real Antminer hardware. The S9 and
every later Bitmain miner run cgminer/bmminer on an on-board controller exposing a
JSON-RPC API on TCP port 4028. The host talks to it over the network, so no Linux kernel
driver or USB access is needed.

Protocol: connect, send a JSON command terminated by a NUL byte, read the NUL-terminated
JSON reply.
"""
from __future__ import annotations

import json
import socket
from typing import Any, Dict, List, Optional


class CGMinerClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 4028, timeout: float = 10.0):
        self.host = host or "127.0.0.1"
        self.port = port or 4028
        self.timeout = timeout

    def send_command(self, command: str, parameter: Optional[Any] = None) -> Dict[str, Any]:
        """Send a single command and return the decoded response."""
        payload: Dict[str, Any] = {"command": command}
        if parameter is not None:
            payload["parameter"] = parameter
        data = json.dumps(payload).encode("utf-8") + b"\x00"

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.settimeout(self.timeout)
            sock.sendall(data)
            chunks: List[bytes] = []
            while True:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                chunks.append(chunk)

        raw = b"".join(chunks).replace(b"\x00", b"")
        if not raw:
            raise ConnectionError(f"empty response from cgminer at {self.host}:{self.port}")
        # Some firmwares emit invalid trailing commas; be lenient on the common case.
        return json.loads(raw.decode("utf-8", errors="replace"))

    def is_available(self) -> bool:
        try:
            self.send_command("version")
            return True
        except Exception:
            return False

    def version(self) -> Dict[str, Any]:
        return self.send_command("version")

    def stats(self) -> Dict[str, Any]:
        return self.send_command("stats")

    def summary(self) -> Dict[str, Any]:
        return self.send_command("summary")

    def devs(self) -> List[Dict[str, Any]]:
        resp = self.send_command("devs")
        return [d for d in resp.get("DEVS", []) if isinstance(d, dict)]

    def detect_type(self) -> Optional[str]:
        """Ask the miner for its hardware type. Tries 'stats' then 'version', scanning the
        common Bitmain response shapes for a 'Type'/'Model'/'Name' field."""
        for cmd in ("stats", "version"):
            try:
                resp = self.send_command(cmd)
            except Exception:
                continue
            t = _extract_type(resp)
            if t:
                return t
        return None


def _extract_type(resp: Dict[str, Any]) -> Optional[str]:
    for key in ("STATS", "VERSION", "DEVS", "SUMMARY"):
        arr = resp.get(key)
        if not isinstance(arr, list):
            continue
        for item in arr:
            if not isinstance(item, dict):
                continue
            for field in ("Type", "Model", "Name"):
                val = item.get(field)
                if isinstance(val, str) and val:
                    return val
    return None
