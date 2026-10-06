import hashlib
import struct

import pytest

from ai_asic.cgminer.client import CGMinerClient
from ai_asic.hardware import bm1387
from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.hardware.device_detector import detect_asic, detection_summary
from ai_asic.hardware.simulator import (
    SimConfig,
    VirtualBM1387,
    VirtualMinerServer,
    _finish_double_sha_from_midstate,
    leading_zero_bits,
    simulate_header,
)


def _double_sha(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


# --- protocol round-trips ----------------------------------------------------

def test_decode_work_roundtrip():
    header = bytes(range(80))
    w = bm1387.new_work_from_header(header, 0xDEADBEEF)
    back = bm1387.decode_work(w.encode())
    assert back.work_id == 0xDEADBEEF
    assert back.midstates == w.midstates
    assert back.data == w.data


def test_decode_work_asicboost_roundtrip():
    header = bytes(80)
    w = bm1387.new_work_asicboost(header, 7, [1, 2, 3, 4])
    back = bm1387.decode_work(w.encode())
    assert len(back.midstates) == 4
    assert back.midstates == w.midstates


def test_decode_work_bad_crc_rejected():
    frame = bytearray(bm1387.new_work_from_header(bytes(80), 1).encode())
    frame[-1] ^= 0xFF  # corrupt the CRC
    with pytest.raises(ValueError):
        bm1387.decode_work(bytes(frame))


def test_nonce_response_roundtrip():
    frame = bm1387.encode_nonce_response(0xABCD1234, work_id=9, midstate_index=2)
    res = bm1387.parse_nonce_response(frame)
    assert res.nonce == 0xABCD1234
    assert res.work_id == 9
    assert res.midstate_index == 2


# --- the core claim: midstate mining == hashing the full header --------------

def test_midstate_finish_matches_hashlib():
    header = bytearray(prepare_asic_job([3] * 12, 0, timestamp=0))
    for nonce in (0, 1, 42, 0xFFFF, 0x12345678):
        struct.pack_into("<I", header, 76, nonce)
        ref = _double_sha(bytes(header))
        got = _finish_double_sha_from_midstate(
            bm1387.compute_midstate(bytes(header[0:64])), bytes(header[64:76]), nonce
        )
        assert got == ref, f"midstate mismatch at nonce {nonce}"


def test_leading_zero_bits():
    assert leading_zero_bits(b"\x00" * 32) == 256
    assert leading_zero_bits(b"\x80" + b"\x00" * 31) == 0
    assert leading_zero_bits(b"\x00\x01" + b"\x00" * 30) == 15


# --- end to end --------------------------------------------------------------

def test_virtual_chip_finds_and_verifies():
    header = prepare_asic_job([1] * 12, 0, timestamp=0)
    cfg = SimConfig(difficulty_bits=8, max_nonces=1 << 20)  # easy target, quick
    sim = simulate_header(header, work_id=1, config=cfg)

    assert sim.result.found
    assert sim.result.leading_zeros >= 8
    # The chip mined from midstate only; an independent hashlib pass over the full header
    # with the found nonce must produce the identical digest.
    assert sim.midstate_matches_header
    assert sim.header_hash_hex == sim.result.hash_hex
    assert sim.response_roundtrips
    assert sim.ok

    # And the nonce genuinely solves the full header.
    full = bytearray(header)
    struct.pack_into("<I", full, 76, sim.result.nonce)
    assert leading_zero_bits(_double_sha(bytes(full))) >= 8


def test_virtual_chip_reports_failure_when_budget_exhausted():
    header = prepare_asic_job([0] * 12, 0, timestamp=0)
    cfg = SimConfig(difficulty_bits=240, max_nonces=50)  # unreachable in 50 rolls
    chip = VirtualBM1387(cfg)
    chip.load_work(bm1387.new_work_from_header(header, 1).encode())
    res = chip.mine()
    assert not res.found
    assert res.hashes_tried == 50


def test_asicboost_sets_midstate_index():
    # Many midstates per nonce; the winner should report which version hit.
    header = prepare_asic_job([5] * 12, 0, timestamp=0)
    work = bm1387.new_work_asicboost(header, 3, [0x20000000, 0x20000004, 0x20000008, 0x2000000C])
    chip = VirtualBM1387(SimConfig(difficulty_bits=8, max_nonces=1 << 20))
    chip.load_work(work.encode())
    res = chip.mine()
    assert res.found
    assert 0 <= res.midstate_index < 4


# --- virtual cgminer API server + detection ----------------------------------

@pytest.fixture
def running_server():
    srv = VirtualMinerServer(model="Antminer S9", port=0).start()
    try:
        yield srv
    finally:
        srv.stop()


def test_server_answers_cgminer_commands(running_server):
    c = CGMinerClient("127.0.0.1", running_server.port)
    assert c.is_available()
    assert c.detect_type() == "Antminer S9"
    assert c.version()["VERSION"][0]["Type"] == "Antminer S9"
    assert "SUMMARY" in c.summary()
    assert c.devs()  # non-empty device list


def test_simulated_miner_is_detected(running_server):
    caps = detect_asic("127.0.0.1", running_server.port)
    assert caps["available"] is True
    profile = caps["profile"]
    assert profile.chip == "BM1387"
    summary = detection_summary("127.0.0.1", running_server.port)
    assert "AVAILABLE" in summary
    assert "BM1387" in summary


def test_server_models_requested_profile():
    with VirtualMinerServer(model="Antminer S19 Pro", port=0) as srv:
        assert srv.profile.chip == "BM1398"
        caps = detect_asic("127.0.0.1", srv.port)
        assert caps["available"] is True
        assert caps["profile"].model == "Antminer S19 Pro"


def test_server_stop_is_idempotent():
    srv = VirtualMinerServer(port=0).start()
    assert srv.running
    srv.stop()
    assert not srv.running
    srv.stop()  # must not raise
