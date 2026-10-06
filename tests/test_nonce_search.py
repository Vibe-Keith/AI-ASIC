"""Fast host nonce search: same answers as the straightforward loop, serial or parallel."""
import hashlib
import json
import struct

from ai_asic.chat.engine import ResponseCache, llama_kwargs
from ai_asic.chat.models_dir import DEFAULT_CONFIG
from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.hardware.simulator import SimConfig, simulate_header
from ai_asic.hashing import nonce_search
from ai_asic.hashing.methods import SoftwareHashMethod

HEADER = hashlib.sha256(b"nonce search").digest() * 2 + bytes(16)


def _reference(header, bits, start, count):
    work = bytearray(header)
    for nonce in range(start, start + count):
        struct.pack_into("<I", work, 76, nonce & 0xFFFFFFFF)
        d = hashlib.sha256(hashlib.sha256(bytes(work)).digest()).digest()
        v = int.from_bytes(d, "big")
        if (256 - v.bit_length() if v else 256) >= bits:
            return nonce, d, nonce - start + 1
    return None, b"", count


def test_target_matches_leading_zero_bits():
    for bits in (0, 1, 7, 8, 9, 28, 255, 256, 300):
        t = nonce_search.target_for_bits(bits)
        for d in (bytes(32), b"\x00" * 31 + b"\x01", b"\x00\x00\x00\x0f" + b"\xff" * 28,
                  b"\x00\x00\x00\x10" + bytes(28), b"\x7f" + b"\xff" * 31, b"\xff" * 32):
            assert (d < t) == (nonce_search.leading_zero_bits(d) >= bits), (bits, d.hex())


def test_serial_matches_reference():
    for bits, start in ((4, 0), (10, 0), (10, 1000), (30, 0)):
        assert nonce_search.search_serial(HEADER, bits, start, 5000) == \
            _reference(HEADER, bits, start, 5000)


def test_parallel_matches_serial(monkeypatch):
    monkeypatch.setattr(nonce_search, "PARALLEL_MIN_WORK", 1 << 10)
    monkeypatch.setattr(nonce_search, "CHUNK", 1 << 10)
    monkeypatch.setenv("AI_ASIC_WORKERS", "2")
    for bits in (12, 14):
        assert nonce_search.search(HEADER, bits, 0, 1 << 16) == \
            nonce_search.search_serial(HEADER, bits, 0, 1 << 16)
    assert nonce_search.search(HEADER, 60, 0, 1 << 13) == (None, b"", 1 << 13)


def test_simulator_fast_path_matches_cycle_model():
    header = prepare_asic_job([3] * 12, 0, timestamp=0)
    fast = simulate_header(header, config=SimConfig(difficulty_bits=9))
    slow = simulate_header(header, config=SimConfig(difficulty_bits=9, fast=False))
    assert fast.ok and slow.ok
    for f in ("nonce", "hash_hex", "hashes_tried", "midstate_index", "response_frame"):
        assert getattr(fast.result, f) == getattr(slow.result, f), f


def test_software_mine_header_difficulty1_semantics():
    # Not found in a short range: returns nonce_end, as before.
    assert SoftwareHashMethod().mine_header(HEADER, 0, 255) == 255


def test_compute_batch_large():
    data = [bytes([i % 256]) * 40 for i in range(1000)]
    assert SoftwareHashMethod().compute_batch(data) == [hashlib.sha256(d).digest() for d in data]


def test_response_cache_jsonl_and_legacy_import(tmp_path):
    legacy = tmp_path / "responses.json"
    legacy.write_text(json.dumps({"k1": {"text": "old", "model": "m", "time": 1}}))
    c = ResponseCache(tmp_path / "responses.jsonl")
    assert c.get("k1") == "old"
    c.put("k2", "new", "m")
    c.put("k1", "newer", "m")
    again = ResponseCache(tmp_path / "responses.jsonl")
    assert again.get("k1") == "newer" and again.get("k2") == "new" and len(again) == 2


def test_llama_kwargs():
    kw = llama_kwargs(DEFAULT_CONFIG)
    assert kw["n_ctx"] == 2048 and "flash_attn" not in kw and "n_threads" not in kw
    assert "type_k" not in kw
    kw = llama_kwargs({**DEFAULT_CONFIG, "kv_cache_type": "q8_0", "n_threads": 6})
    assert kw["type_k"] == kw["type_v"] == 8 and kw["n_threads"] == 6 and kw["flash_attn"]

    class Old:
        def __init__(self, model_path, n_ctx=512, n_threads=None):
            pass

    assert set(llama_kwargs({**DEFAULT_CONFIG, "n_threads": 4}, Old)) == {"n_ctx", "n_threads"}
