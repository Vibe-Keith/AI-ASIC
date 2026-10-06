"""A software stratum miner that builds work the way cgminer/bmminer firmware does.

It exists to dry-run the closed loop without hardware, and as an independent check on the
pool's byte orders: it constructs headers the cgminer way (big-endian-word work data, then
``flip80``) rather than reusing the pool's reconstruction, so a convention mismatch shows up as
rejected shares here instead of on a real miner. Pure Python - keep its share difficulty low
(``2**-16`` or so).
"""
from __future__ import annotations

import itertools
import json
import socket
import threading
from typing import Dict, Optional

from ai_asic.stratum import protocol as P


class VirtualStratumMiner:
    def __init__(self, host: str, port: int, worker: str = "virtual.0",
                 version_rolling: bool = True, user_agent: str = "ai-asic-virtual/1.0"):
        self.host, self.port = host, port
        self.worker = worker
        self.version_rolling = version_rolling
        self.user_agent = user_agent
        self.accepted = 0
        self.rejected = 0
        self.last_error = None
        self._sock: Optional[socket.socket] = None
        self._wlock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: Dict[int, str] = {}
        self._job: Optional[list] = None
        self._job_gen = 0
        self._target = P.target_for_difficulty(1.0)
        self._mask = 0
        self._en1 = b""
        self._en2_size = 4
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._threads = []

    def start(self) -> "VirtualStratumMiner":
        self._sock = socket.create_connection((self.host, self.port), timeout=10.0)
        self._sock.settimeout(None)
        self._rfile = self._sock.makefile("rb")
        for target in (self._read_loop, self._mine_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)
        if self.version_rolling:
            self._call("mining.configure", [["version-rolling"],
                                            {"version-rolling.mask": "ffffffff",
                                             "version-rolling.min-bit-count": 2}])
        self._call("mining.subscribe", [self.user_agent])
        self._call("mining.authorize", [self.worker, "x"])
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()

    def __enter__(self) -> "VirtualStratumMiner":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- wire ---------------------------------------------------------------
    def _call(self, method: str, params: list) -> int:
        mid = next(self._ids)
        self._pending[mid] = method
        data = (json.dumps({"id": mid, "method": method, "params": params}) + "\n").encode()
        with self._wlock:
            self._sock.sendall(data)
        return mid

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = self._rfile.readline()
            except (OSError, ValueError):
                break
            if not line:
                break
            msg = json.loads(line)
            method = msg.get("method")
            if method == "mining.set_difficulty":
                with self._cv:
                    self._target = P.target_for_difficulty(float(msg["params"][0]))
            elif method == "mining.notify":
                with self._cv:
                    self._job = msg["params"]
                    self._job_gen += 1
                    self._cv.notify_all()
            elif msg.get("id") in self._pending:
                kind = self._pending.pop(msg["id"])
                result = msg.get("result")
                if kind == "mining.subscribe":
                    self._en1 = bytes.fromhex(result[1])
                    self._en2_size = int(result[2])
                elif kind == "mining.configure" and isinstance(result, dict):
                    if result.get("version-rolling"):
                        self._mask = int(result.get("version-rolling.mask", "0"), 16)
                elif kind == "mining.submit":
                    if result is True:
                        self.accepted += 1
                    else:
                        self.rejected += 1
                        self.last_error = msg.get("error")

    # -- hashing (cgminer's construction) -------------------------------------
    def _mine_loop(self) -> None:
        en2_counter = itertools.count()
        while not self._stop.is_set():
            with self._cv:
                while self._job is None and not self._stop.is_set():
                    self._cv.wait()
                if self._stop.is_set():
                    return
                job, gen, target, mask = self._job, self._job_gen, self._target, self._mask
            job_id, prevhash, coinb1, coinb2, branches, version, nbits, ntime, _ = job
            en2 = next(en2_counter).to_bytes(self._en2_size, "little")
            merkle = P.dsha256(bytes.fromhex(coinb1) + self._en1 + en2 + bytes.fromhex(coinb2))
            for branch in branches:
                merkle = P.dsha256(merkle + bytes.fromhex(branch))
            # gen_stratum_work: hex template, merkle root flipped into BE words.
            data = bytearray.fromhex(version + prevhash + P.swap_words(merkle).hex()
                                     + ntime + nbits + "00000000")
            roll = 0
            if mask:
                roll = (int(version, 16) & ~mask) | ((gen * 0x2000) & mask)  # vary rolled bits
                data[0:4] = (roll & 0xFFFFFFFF).to_bytes(4, "big")
            for nonce in range(1 << 32):
                if nonce & 0xFFF == 0 and (self._job_gen != gen or self._stop.is_set()):
                    break
                data[76:80] = nonce.to_bytes(4, "little")   # htole32(nonce)
                digest = P.dsha256(P.swap_words(bytes(data)))  # flip80, then SHA256d
                if P.hash_value(digest) <= target:
                    params = [self.worker, job_id, en2.hex(), ntime, data[76:80].hex()]
                    if mask:
                        params.append(f"{roll & mask:08x}")
                    try:
                        self._call("mining.submit", params)
                    except OSError:
                        return
