"""Virtual BM1387 ASIC - a software model of an Antminer S9-class mining chip.

The rest of this package can *build* work for a BM1387 (``bm1387.new_work_from_header``) and
*parse* the nonces it returns (``bm1387.parse_nonce_response``), but without real hardware
there is no way to prove that pipeline is correct. This module closes that loop: it is a
cycle-honest model of the chip that

  1. ingests the exact chain frame ``BM1387Work.encode()`` produces (and rejects a bad CRC,
     like the real chip),
  2. mines using *only* the 32-byte midstate and 12-byte tail it was handed - never the full
     header - by finishing the second SHA-256 block from the midstate and then the second
     SHA-256, exactly as the silicon's pipeline does,
  3. emits a golden nonce in the real 6-byte response frame format, and
  4. lets the caller independently confirm, with ``hashlib`` over the full 80-byte header,
     that the nonce the virtual chip found genuinely solves the block.

If all four line up, the host-side encode/decode path in this repo would drive a real S9.

The on-chip SHA-256 is pure Python (``bm1387._sha256_compress``) on purpose: it demonstrates
the midstate shortcut a real ASIC exploits (block 1 is pre-hashed by the host, the chip only
rolls the nonce through block 2). Rolling every nonce through it runs at a few kH/s, so by
default (``SimConfig.fast``) :func:`simulate_header` rolls with ``hashlib`` resumed from the
same midstate (``ai_asic.hashing.nonce_search``, ~1 MH/s per core, process-parallel for long
searches) and re-derives the winning nonce through the pure-Python model, raising on any
disagreement. Results are identical (same first nonce, hash and count); ``fast=False`` (CLI
``--cycle-model``) rolls every nonce through the model. The reported *simulated* time is
scaled to a real chip's nominal hash rate regardless.

Pure Python, standard library only, cross-platform.
"""
from __future__ import annotations

import json
import socketserver
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ai_asic.hardware import bm1387
from ai_asic.hardware.bm1387 import _MASK, _SHA256_INIT, _sha256_compress
from ai_asic.hardware.miner_profiles import MinerProfile, default_profile, detect_profile
from ai_asic.hashing import nonce_search

# Length of an 80-byte Bitcoin header, in bits, as the SHA-256 message-length suffix.
_HEADER_BITLEN = struct.pack(">Q", 80 * 8)
# Length of a 32-byte digest (the inner hash), in bits, for the outer SHA-256.
_DIGEST_BITLEN = struct.pack(">Q", 32 * 8)


def leading_zero_bits(digest: bytes) -> int:
    """Count the leading zero bits of a 32-byte digest read big-endian."""
    value = int.from_bytes(digest, "big")
    return 256 - value.bit_length() if value else 256


def _finish_double_sha_from_midstate(midstate: bytes, data12: bytes, nonce: int) -> bytes:
    """Reproduce double-SHA-256(header) from the pre-hashed state alone.

    ``midstate`` is the SHA-256 state after block 1 (header bytes 0-63). ``data12`` is the
    merkle-tail + ntime + nbits (header bytes 64-75). The chip supplies the 4-byte ``nonce``
    (header bytes 76-79). This rebuilds block 2 with SHA-256 padding, finishes the first
    SHA-256, then runs the second SHA-256 - the two compressions a BM1387 core performs per
    nonce after the host hands it the midstate.
    """
    state = list(struct.unpack(">8I", midstate))
    # Block 2 = header[64:80] (data tail + nonce LE) + SHA-256 padding to one 64-byte block.
    block2 = data12 + struct.pack("<I", nonce & _MASK) + b"\x80" + b"\x00" * 39 + _HEADER_BITLEN
    first = struct.pack(">8I", *_sha256_compress(state, block2))
    # Second SHA-256 over the 32-byte first digest (single padded block).
    block = first + b"\x80" + b"\x00" * 23 + _DIGEST_BITLEN
    return struct.pack(">8I", *_sha256_compress(list(_SHA256_INIT), block))


@dataclass
class SimConfig:
    """How hard to mine and how to model the (pretend) silicon."""

    difficulty_bits: int = 16          # golden nonce needs this many leading zero hash bits
    chip_count: int = 63               # BM1387 chips on one S9 hashboard
    nominal_hashrate: float = 13.5e12  # H/s a real S9 sustains (for the "real chip" estimate)
    max_nonces: int = 1 << 24          # give up after this many rolls (per midstate)
    start_nonce: int = 0
    fast: bool = True                  # hashlib roll + model check of the winner (see module doc)


@dataclass
class SimResult:
    found: bool
    nonce: int = 0
    midstate_index: int = 0
    hash_hex: str = ""                 # the double-SHA-256 the chip computed, big-endian
    leading_zeros: int = 0
    hashes_tried: int = 0
    sim_seconds: float = 0.0           # wall-clock the Python model took
    est_real_seconds: float = 0.0      # how long a real chip at nominal_hashrate would take
    response_frame: bytes = b""        # the 6-byte frame the chip would send back
    chip_id: int = 0                   # which (modeled) chip rolled the winning nonce


class VirtualBM1387:
    """A single simulated BM1387 chip-chain. Hand it an encoded work frame; it mines."""

    def __init__(self, config: Optional[SimConfig] = None):
        self.config = config or SimConfig()
        self.work: Optional[bm1387.BM1387Work] = None

    def load_work(self, frame: bytes) -> bm1387.BM1387Work:
        """Ingest an encoded chain frame, verifying its CRC16 the way the chip does."""
        self.work = bm1387.decode_work(frame, verify_crc=True)
        return self.work

    def mine(self, header: Optional[bytes] = None) -> SimResult:
        """Roll the nonce (and every loaded midstate, AsicBoost-style) until a hash meets
        the difficulty target or the nonce budget is exhausted.

        With ``config.fast`` and the 80-byte ``header`` the work came from (single midstate),
        the roll runs in ``hashlib`` from that midstate and only the winning nonce goes through
        the pure-Python model; the header is checked against the loaded midstate and tail first.
        """
        if self.work is None:
            raise RuntimeError("no work loaded; call load_work() first")
        cfg = self.config
        midstates = self.work.midstates
        data = self.work.data
        target_zeros = cfg.difficulty_bits
        tried = 0
        t0 = time.perf_counter()
        end = cfg.start_nonce + cfg.max_nonces
        if (cfg.fast and header is not None and len(header) == 80 and len(midstates) == 1
                and bytes(header[64:76]) == bytes(data)
                and bm1387.compute_midstate(bytes(header[:64])) == midstates[0]):
            nonce, digest, tried = nonce_search.search(header, target_zeros, cfg.start_nonce,
                                                       cfg.max_nonces)
            if nonce is None:
                return self._result(False, end & _MASK, 0, b"\x00" * 32, tried, t0)
            if _finish_double_sha_from_midstate(midstates[0], data, nonce) != digest:
                raise RuntimeError("midstate model disagrees with the nonce search")
            return self._result(True, nonce, 0, digest, tried, t0)
        for nonce in range(cfg.start_nonce, end):
            for mi, ms in enumerate(midstates):
                digest = _finish_double_sha_from_midstate(ms, data, nonce)
                tried += 1
                if leading_zero_bits(digest) >= target_zeros:
                    return self._result(True, nonce, mi, digest, tried, t0)
        return self._result(False, end & _MASK, 0, b"\x00" * 32, tried, t0)

    def _result(self, found, nonce, mi, digest, tried, t0) -> SimResult:
        cfg = self.config
        return SimResult(
            found=found,
            nonce=nonce,
            midstate_index=mi,
            hash_hex=digest.hex(),
            leading_zeros=leading_zero_bits(digest),
            hashes_tried=tried,
            sim_seconds=time.perf_counter() - t0,
            est_real_seconds=tried / cfg.nominal_hashrate if cfg.nominal_hashrate else 0.0,
            response_frame=bm1387.encode_nonce_response(
                nonce, self.work.work_id if self.work else 0, mi
            ),
            chip_id=nonce % cfg.chip_count if cfg.chip_count else 0,
        )


@dataclass
class VerifiedSim:
    """A full mine + independent verification: everything a 'does it actually work?' run
    needs to show."""

    result: SimResult
    header: bytes
    parsed: bm1387.BM1387NonceResult       # chip's response frame decoded by the host
    header_hash_hex: str                   # hashlib double-SHA-256 over the full header
    midstate_matches_header: bool          # chip's midstate math == hashlib over the header
    response_roundtrips: bool              # parsed nonce == mined nonce

    @property
    def ok(self) -> bool:
        return (
            self.result.found
            and self.midstate_matches_header
            and self.response_roundtrips
        )


def simulate_header(header: bytes, work_id: int = 1,
                    config: Optional[SimConfig] = None) -> VerifiedSim:
    """End-to-end proof on one 80-byte header: encode work -> virtual chip mines from the
    midstate -> chip returns a nonce frame -> host parses it -> independently re-hash the
    full header with ``hashlib`` and confirm the chip's nonce really solves it.
    """
    import hashlib

    if len(header) != 80:
        raise ValueError(f"header must be 80 bytes, got {len(header)}")
    cfg = config or SimConfig()

    # Host side: build and serialize the work frame (what would go down the chain).
    work = bm1387.new_work_from_header(header, work_id)
    frame = work.encode()

    # Chip side: ingest the frame (CRC-checked) and mine from midstate + tail (with cfg.fast the
    # roll uses hashlib from the same midstate and the winner is re-derived by the model).
    chip = VirtualBM1387(cfg)
    chip.load_work(frame)
    res = chip.mine(header)

    # Host side: parse the chip's response frame back into a nonce.
    parsed = bm1387.parse_nonce_response(res.response_frame)

    # Independent check: splice the found nonce into the FULL header and double-SHA it with
    # hashlib (OpenSSL), with no midstate shortcut. This must match the chip's own digest.
    full = bytearray(header)
    struct.pack_into("<I", full, 76, res.nonce & _MASK)
    ref = hashlib.sha256(hashlib.sha256(bytes(full)).digest()).digest()

    return VerifiedSim(
        result=res,
        header=header,
        parsed=parsed,
        header_hash_hex=ref.hex(),
        midstate_matches_header=(res.found and ref.hex() == res.hash_hex),
        response_roundtrips=(parsed.nonce == res.nonce),
    )


# --- virtual cgminer/bmminer API server --------------------------------------
# A real S9-class miner exposes a NUL-terminated JSON-RPC API on TCP 4028 (see
# ai_asic/cgminer/client.py). Serving that same API here makes the virtual chip detectable
# by the ordinary detection path (device_detector.detect_asic) and usable as the ASIC
# backend for inference (ASICHashMethod) - the host code never learns it is simulated.

class _CGMinerAPIHandler(socketserver.BaseRequestHandler):
    """Handle one cgminer-protocol request: read a NUL-terminated JSON command, reply with
    a NUL-terminated JSON response, then close (exactly what bmminer does)."""

    def handle(self) -> None:
        self.request.settimeout(5.0)
        buf = bytearray()
        try:
            while b"\x00" not in buf:
                chunk = self.request.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 8192:  # commands are tiny; guard against junk
                    break
        except OSError:
            return
        raw = bytes(buf).split(b"\x00", 1)[0]
        command, parameter = "version", None
        try:
            req = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
            command = str(req.get("command", "version"))
            parameter = req.get("parameter")
        except (ValueError, AttributeError):
            pass
        resp = self.server.build_response(command, parameter)  # type: ignore[attr-defined]
        try:
            self.request.sendall(json.dumps(resp).encode("utf-8") + b"\x00")
        except OSError:
            pass


class _ThreadingAPIServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class VirtualMinerServer:
    """A background TCP server that impersonates a Bitmain cgminer/bmminer API for one
    modeled miner, so the virtual chip can be *detected* and *driven* like real hardware.

    ::

        srv = VirtualMinerServer(model="Antminer S9")
        srv.start()
        detection_summary("127.0.0.1", srv.port)   # reports AVAILABLE, BM1387
        srv.stop()
    """

    def __init__(self, model: str = "Antminer S9", host: str = "127.0.0.1",
                 port: int = 4028, difficulty_bits: int = 16):
        self.model = model
        self.host = host
        self.port = port
        self.difficulty_bits = difficulty_bits
        self.profile: MinerProfile = detect_profile(model) or default_profile()
        self._server: Optional[_ThreadingAPIServer] = None
        self._thread: Optional[threading.Thread] = None
        self._started_at = 0.0

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "VirtualMinerServer":
        if self._server is not None:
            return self
        server = _ThreadingAPIServer((self.host, self.port), _CGMinerAPIHandler)
        server.build_response = self.build_response  # type: ignore[attr-defined]
        # bound port (handles port=0 "pick a free port")
        self.port = server.server_address[1]
        self._server = server
        self._started_at = time.time()
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def __enter__(self) -> "VirtualMinerServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- response shaping ---------------------------------------------------
    def _elapsed(self) -> int:
        return max(0, int(time.time() - self._started_at))

    def _ghs(self) -> float:
        return self.profile.nominal_hashrate / 1e9  # API reports GH/s

    def _status(self, msg: str) -> List[dict]:
        return [{"STATUS": "S", "When": int(time.time()), "Code": 22,
                 "Msg": msg, "Description": "bmminer 1.0.0 (ai-asic virtual)"}]

    def build_response(self, command: str, parameter=None) -> dict:
        cmd = (command or "version").strip().lower()
        if cmd == "version":
            return {
                "STATUS": self._status("CGMiner versions"),
                "VERSION": [{
                    "BMMiner": "2.0.0", "API": "3.1",
                    "Miner": f"{self.model} (virtual)",
                    "CompileTime": "ai-asic simulator",
                    "Type": self.model,
                }],
                "id": 1,
            }
        if cmd == "stats":
            ghs = self._ghs()
            return {
                "STATUS": self._status("CGMiner stats"),
                "STATS": [{
                    "STATS": 0, "ID": "BC50", "Elapsed": self._elapsed(),
                    "Type": self.model, "Chip": self.profile.chip,
                    "GHS 5s": f"{ghs:.2f}", "GHS av": f"{ghs:.2f}",
                    "frequency": str(self.profile.default_frequency_mhz),
                    "total_rateideal": f"{ghs:.2f}",
                    "chain_acn1": self.profile.chip_count,
                    "chain_acs1": "o" * max(1, self.profile.chip_count),
                    "miner_count": self.profile.board_count,
                    "fan_num": 2, "fan1": 4000, "fan2": 4080,
                    "temp1": 55, "temp2": 57,
                }],
                "id": 1,
            }
        if cmd == "summary":
            ghs = self._ghs()
            accepted = self._elapsed()  # one "share" per second of uptime
            return {
                "STATUS": self._status("Summary"),
                "SUMMARY": [{
                    "Elapsed": self._elapsed(),
                    "GHS 5s": f"{ghs:.2f}", "GHS av": f"{ghs:.2f}",
                    "Found Blocks": 0, "Getworks": 1 + self._elapsed() // 30,
                    "Accepted": accepted, "Rejected": 0, "Hardware Errors": 0,
                    "Utility": round(accepted / max(1, self._elapsed() / 60), 2),
                    "Discarded": 0, "Stale": 0, "Device Hardware%": 0.0,
                    "Device Rejected%": 0.0, "Work Utility": float(accepted),
                }],
                "id": 1,
            }
        if cmd == "devs":
            ghs = self._ghs()
            devs = []
            for i in range(max(1, self.profile.board_count)):
                devs.append({
                    "ASC": i, "Name": "BC5", "ID": i, "Enabled": "Y",
                    "Status": "Alive", "Type": self.model, "Chip": self.profile.chip,
                    "GHS 5s": f"{ghs / max(1, self.profile.board_count):.2f}",
                    "GHS av": f"{ghs / max(1, self.profile.board_count):.2f}",
                    "Temperature": 56.0 + i,
                })
            return {"STATUS": self._status("Devices"), "DEVS": devs, "id": 1}
        if cmd == "pools":
            return {
                "STATUS": self._status("Pools"),
                "POOLS": [{
                    "POOL": 0, "URL": "stratum+tcp://virtual.local:3333",
                    "Status": "Alive", "Accepted": self._elapsed(), "Rejected": 0,
                }],
                "id": 1,
            }
        return {
            "STATUS": [{"STATUS": "E", "When": int(time.time()), "Code": 14,
                        "Msg": f"Invalid command: {command}",
                        "Description": "bmminer 1.0.0 (ai-asic virtual)"}],
            "id": 1,
        }
