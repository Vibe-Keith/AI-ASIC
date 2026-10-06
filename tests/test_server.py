import hashlib
import json
import struct

import pytest

from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.server import (
    ChainAsicDevice,
    HasherClient,
    HasherServer,
    VirtualAsicDevice,
    auto_device,
)
from ai_asic.server.device import MAX_BATCH_SIZE


@pytest.fixture
def server():
    srv = HasherServer(device=VirtualAsicDevice("Antminer S9"), port=0,
                       difficulty_bits=12).start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture
def client(server):
    return HasherClient("127.0.0.1", server.port)


# --- device backends ---------------------------------------------------------

def test_virtual_device_always_available():
    dev = VirtualAsicDevice("Antminer S9")
    assert dev.available
    assert dev.compute_hash(b"x") == hashlib.sha256(b"x").digest()
    assert dev.profile.chip == "BM1387"


def test_chain_device_unavailable_off_board():
    dev = ChainAsicDevice("Antminer S9", device_path="/nonexistent/bitmain-asic")
    assert dev.available is False
    with pytest.raises(RuntimeError):
        dev.mine(prepare_asic_job([0] * 12, 0, timestamp=0), 8)


def test_auto_device_falls_back_to_virtual():
    dev = auto_device("Antminer S9", device_path="/nonexistent/bitmain-asic")
    assert isinstance(dev, VirtualAsicDevice)


# --- service methods ---------------------------------------------------------

def test_health_and_info(client):
    assert client.is_available()
    info = client.device_info()
    assert info["chip"] == "BM1387"
    assert info["model"] == "Antminer S9"


def test_compute_hash(client):
    assert client.compute_hash(b"hello") == hashlib.sha256(b"hello").digest()


def test_compute_batch(client):
    data = [b"a", b"b", b"c"]
    out = client.compute_batch(data)
    assert out == [hashlib.sha256(d).digest() for d in data]


def test_compute_batch_limit(server):
    with pytest.raises(RuntimeError):  # device raises; server reports ok=False -> client raises
        HasherClient("127.0.0.1", server.port).compute_batch([b"x"] * (MAX_BATCH_SIZE + 1))


def test_mine_verifies(client):
    header = prepare_asic_job([7] * 12, 0, timestamp=0)
    r = client.mine(header, difficulty_bits=12)
    assert r.found
    full = bytearray(header)
    struct.pack_into("<I", full, 76, r.nonce)
    ref = hashlib.sha256(hashlib.sha256(bytes(full)).digest()).digest()
    assert ref.hex() == r.hash_hex  # the nonce really solves the header
    assert r.leading_zeros >= 12


def test_metrics_accumulate(client):
    client.compute_hash(b"one")
    client.compute_batch([b"two", b"three"])
    m = client.metrics()
    assert m["total_requests"] >= 2
    assert m["total_hashes"] >= 3
    assert m["error_count"] == 0


def test_unknown_method_errors(server):
    c = HasherClient("127.0.0.1", server.port)
    with pytest.raises(RuntimeError):
        c._rpc("NoSuchMethod")


def test_client_reuses_one_connection(server, client):
    client.compute_hash(b"a")
    sock = client._sock
    assert sock is not None
    for i in range(20):
        client.compute_batch([bytes([i])] * 4)
    assert client._sock is sock  # no reconnects across calls
    assert client.metrics()["total_requests"] >= 21


def test_client_reconnects_after_dropped_connection(server, client):
    client.compute_hash(b"a")
    server._server.close_connections()  # e.g. server idle timeout or restart
    assert client.compute_hash(b"b") == hashlib.sha256(b"b").digest()


def test_client_works_with_close_per_reply_server(server, client, monkeypatch):
    # An older server answers one request per connection; the client must keep working.
    from ai_asic.server import hasher_server

    def one_shot(self):
        self.request.settimeout(5.0)
        buf = bytearray()
        while b"\x00" not in buf:
            chunk = self.request.recv(65536)
            if not chunk:
                return
            buf += chunk
        req = json.loads(bytes(buf).partition(b"\x00")[0])
        self.request.sendall(json.dumps(self.server.dispatch(req)).encode() + b"\x00")

    monkeypatch.setattr(hasher_server._Handler, "handle", one_shot)
    for d in (b"x", b"y", b"z"):
        assert client.compute_hash(d) == hashlib.sha256(d).digest()


def test_client_raises_when_server_gone():
    srv = HasherServer(device=VirtualAsicDevice(), port=0).start()
    c = HasherClient("127.0.0.1", srv.port, timeout=5.0)
    assert c.compute_hash(b"a")
    srv.stop()
    with pytest.raises(OSError):
        c.compute_hash(b"b")


def test_stop_is_idempotent():
    srv = HasherServer(device=VirtualAsicDevice(), port=0).start()
    assert srv.running
    srv.stop()
    assert not srv.running
    srv.stop()
