import json
import os
import socket
import socketserver
import threading
import time

import pytest

from ai_asic.chat.accelerator import HashAccelerator
from ai_asic.chat.engine import ChatEngine, verify_transcripts
from ai_asic.chat.models_dir import ensure_layout
from ai_asic.stratum import StratumPool, VirtualStratumMiner
from ai_asic.stratum import protocol as P
from ai_asic.workloads.split_model import DEVICE_ASIC, DEVICE_HOST

EASY = 2 ** -16  # 16-bit shares: quick for the pure-Python miner


@pytest.fixture
def pool():
    p = StratumPool("127.0.0.1", 0, share_difficulty=EASY).start()
    try:
        yield p
    finally:
        p.stop()


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ASIC_MODELS", str(tmp_path / "ai_models"))
    return ensure_layout(tmp_path / "ai_models")


class _Raw:
    """A bare stratum client for protocol-level checks."""

    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.f = self.sock.makefile("rb")
        self.n = 0

    def call(self, method, params):
        self.n += 1
        self.sock.sendall((json.dumps({"id": self.n, "method": method, "params": params})
                           + "\n").encode())
        while True:
            msg = json.loads(self.f.readline())
            if msg.get("id") == self.n:
                return msg

    def close(self):
        self.sock.close()


# --- protocol -------------------------------------------------------------------

def test_difficulty_targets():
    assert P.target_for_difficulty(1) == P.DIFF1_TARGET
    assert P.bits_for_difficulty(1) == 32
    assert P.bits_for_difficulty(256) == 40
    assert P.bits_for_difficulty(2 ** -16) == 16


def test_coinbase_commits_to_payload():
    payload = os.urandom(32)
    c1, c2 = P.build_coinbase(payload, height=7, extranonce_size=8)
    coinbase = c1 + b"\x11" * 8 + c2
    assert P.coinbase_commits_to(coinbase, payload)
    assert not P.coinbase_commits_to(coinbase, os.urandom(32))
    assert c1[41] == len(c1) - 42 + 8  # script length covers the extranonce


# --- pool + virtual miner --------------------------------------------------------

@pytest.mark.parametrize("rolling", [False, True])
def test_virtual_miner_seals_verify(pool, rolling):
    with VirtualStratumMiner("127.0.0.1", pool.port, version_rolling=rolling) as miner:
        assert pool.wait_for_miner(5)
        prev = b"\x00" * 32
        for _ in range(2):
            digest = os.urandom(32)
            share = pool.mine(prev, digest, int(time.time()), min_bits=12, timeout=30)
            assert share is not None
            assert P.verify_seal(prev, digest, share.coinbase, share.header, 12) is None
            assert share.leading_zeros >= 16
            prev = digest
        assert miner.rejected == 0
        assert pool.miners()[0].version_mask == (P.DEFAULT_VERSION_MASK if rolling else 0)


def test_tampered_proof_is_caught(pool):
    with VirtualStratumMiner("127.0.0.1", pool.port):
        assert pool.wait_for_miner(5)
        prev, digest = os.urandom(32), os.urandom(32)
        share = pool.mine(prev, digest, 0, min_bits=12, timeout=30)
    assert P.verify_seal(prev, os.urandom(32), share.coinbase, share.header, 12)
    assert P.verify_seal(os.urandom(32), digest, share.coinbase, share.header, 12)
    bad = bytearray(share.header)
    bad[76] ^= 1
    assert P.verify_seal(prev, digest, share.coinbase, bytes(bad), 12)


def test_pool_rejects_bad_shares(pool):
    c = _Raw(pool.port)
    try:
        sub = c.call("mining.subscribe", ["raw/1"])["result"]
        assert len(bytes.fromhex(sub[1])) == 4 and sub[2] == 4
        assert c.call("mining.authorize", ["w", "x"])["result"] is True
        job_id = pool._current.job_id
        bad = c.call("mining.submit", ["w", job_id, "00000000", "00000000", "00000000"])
        assert bad["result"] is False and bad["error"][0] == 23   # low difficulty
        stale = c.call("mining.submit", ["w", "nope", "00000000", "00000000", "00000000"])
        assert stale["error"][0] == 21
        junk = c.call("mining.submit", ["w", job_id, "zz", "00000000", "00000000"])
        assert junk["error"][0] == 20
        assert c.call("mining.bogus", [])["error"][0] == 20
    finally:
        c.close()


def test_mine_without_miner_returns_none(pool):
    assert pool.mine(b"\x00" * 32, b"\x00" * 32, 0, timeout=0.1) is None


def test_allow_list_refuses_other_hosts():
    p = StratumPool("127.0.0.1", 0, share_difficulty=EASY, allow=["10.9.9.9"]).start()
    try:
        s = socket.create_connection(("127.0.0.1", p.port), timeout=5)
        s.sendall(b'{"id":1,"method":"mining.subscribe","params":[]}\n')
        assert s.recv(100) == b""  # closed without a reply
        s.close()
        assert not p.has_miner
    finally:
        p.stop()


def test_hashrate_estimate(pool):
    with VirtualStratumMiner("127.0.0.1", pool.port):
        assert pool.wait_for_miner(5)
        for _ in range(3):
            pool.mine(os.urandom(32), os.urandom(32), 0, timeout=30)
    assert pool.hashrate() > 0


# --- chat engine through the pool ---------------------------------------------------

class _FakeLlama:
    def create_chat_completion(self, messages, stream, max_tokens, temperature):
        yield {"choices": [{"delta": {"content": "sealed on the miner"}}]}

    def tokenize(self, data, add_bos=False):
        return data.split()


def test_engine_seals_on_miner_and_verifies(root, pool):
    with VirtualStratumMiner("127.0.0.1", pool.port):
        assert pool.wait_for_miner(5)
        acc = HashAccelerator(pool=pool)
        assert acc.is_hardware and acc.device == DEVICE_HOST and acc.mine_device == DEVICE_ASIC
        eng = ChatEngine(root=root, llm=_FakeLlama(), accelerator=acc, seal_difficulty=12,
                         use_cache=False)
        r1 = eng.reply("one")
        eng.reply("two")
        eng.flush()  # seals run in the background; finish them while the miner is attached
    assert r1.seal["kind"] == "stratum"
    stages = {s.name: s for s in r1.trace.stages}
    assert stages["seal"].device == DEVICE_ASIC
    assert stages["fingerprint"].device == DEVICE_HOST  # miners only hash block headers
    assert r1.seal["kind"] == "stratum" and r1.seal["found"]

    log = root / "cache" / "transcripts.jsonl"
    ok, n, problems = verify_transcripts(log)
    assert ok and n == 2, problems

    lines = log.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["header"] = rec["header"][:-2] + ("00" if rec["header"][-2:] != "00" else "01")
    log.write_text("\n".join([lines[0], json.dumps(rec)]) + "\n", encoding="utf-8")
    ok, _, problems = verify_transcripts(log)
    assert not ok and any("miner seal invalid" in p for p in problems)


def test_engine_falls_back_to_host_seal_without_miner(root, pool):
    acc = HashAccelerator(pool=pool)
    assert not acc.is_hardware and "no miner" in acc.label
    eng = ChatEngine(root=root, llm=_FakeLlama(), accelerator=acc, seal_difficulty=8,
                     use_cache=False)
    r = eng.reply("hi")
    assert "kind" not in r.seal and r.seal["found"]
    assert {s.name: s for s in r.trace.stages}["seal"].device == DEVICE_HOST
    assert verify_transcripts(root / "cache" / "transcripts.jsonl")[0]


# --- miner setup over the cgminer API -------------------------------------------------

class _FakeMinerAPI:
    """Answers cgminer API commands like stock bmminer: pools / addpool / switchpool."""

    def __init__(self, allow_write=True):
        self.pools = [{"POOL": 0, "URL": "stratum+tcp://public.example:3333",
                       "Status": "Alive"}]
        self.active = 0
        self.allow_write = allow_write
        api = self

        class H(socketserver.BaseRequestHandler):
            def handle(self):
                req = json.loads(self.request.recv(4096).rstrip(b"\x00"))
                self.request.sendall(json.dumps(api.answer(req)).encode() + b"\x00")

        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def answer(self, req):
        cmd, param = req["command"], req.get("parameter", "")
        if cmd == "pools":
            return {"STATUS": [{"STATUS": "S"}], "POOLS": self.pools}
        if not self.allow_write:
            return {"STATUS": [{"STATUS": "E", "Msg": "Access denied to 'addpool' command"}]}
        if cmd == "addpool":
            url = param.split(",")[0]
            self.pools.append({"POOL": len(self.pools), "URL": url, "Status": "Alive"})
            return {"STATUS": [{"STATUS": "S", "Msg": "Added pool"}]}
        if cmd == "switchpool":
            self.active = int(param)
            return {"STATUS": [{"STATUS": "S", "Msg": "Switching pool"}]}
        return {"STATUS": [{"STATUS": "E", "Msg": "Invalid command"}]}

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def test_point_miner_at_pool():
    from ai_asic.stratum.miner_setup import other_active_pools, point_miner_at

    api = _FakeMinerAPI()
    try:
        url = "stratum+tcp://192.168.1.10:3333"
        assert other_active_pools("127.0.0.1", url, api.port) == [
            "stratum+tcp://public.example:3333"]
        point_miner_at("127.0.0.1", url, api_port=api.port)
        assert api.active == 1 and api.pools[1]["URL"] == url
        point_miner_at("127.0.0.1", url, api_port=api.port)  # idempotent: no second add
        assert len(api.pools) == 2
    finally:
        api.close()


def test_point_miner_at_explains_refusal():
    from ai_asic.stratum.miner_setup import point_miner_at

    api = _FakeMinerAPI(allow_write=False)
    try:
        with pytest.raises(RuntimeError, match="web UI"):
            point_miner_at("127.0.0.1", "stratum+tcp://10.0.0.2:3333", api_port=api.port)
    finally:
        api.close()
