import hashlib

import pytest

from ai_asic.hardware import bm1387


def test_sha256_compress_empty_string():
    # SHA-256("") = e3b0...b855. Padded single block: 0x80 then zeros, length 0.
    block = bytearray(64)
    block[0] = 0x80
    ms = bm1387.compute_midstate(bytes(block))
    assert ms.hex() == hashlib.sha256(b"").hexdigest()


def test_compute_midstate_deterministic():
    block = bytes(range(64))
    assert bm1387.compute_midstate(block) == bm1387.compute_midstate(block)


def test_compute_midstate_bad_length():
    with pytest.raises(ValueError):
        bm1387.compute_midstate(bytes(63))


def test_swap_midstate_word_endianness():
    ms = bytes(range(32))
    swapped = bm1387.swap_midstate_word_endianness(ms)
    assert swapped[0:4] == bytes([3, 2, 1, 0])
    assert bm1387.swap_midstate_word_endianness(swapped) == ms


def test_new_work_from_header():
    header = bytes(range(80))
    w = bm1387.new_work_from_header(header, 0xDEADBEEF)
    assert len(w.midstates) == 1
    assert w.data == header[64:76]
    assert w.midstates[0] == bm1387.compute_midstate(header[0:64])


def test_new_work_from_header_bad_length():
    with pytest.raises(ValueError):
        bm1387.new_work_from_header(bytes(79), 0)


def test_asicboost_distinct_midstates():
    header = bytes(80)
    w = bm1387.new_work_asicboost(header, 1, [0x20000000, 0x20000004, 0x20000008, 0x2000000C])
    assert len(w.midstates) == 4
    assert w.midstates[0] != w.midstates[1]
    with pytest.raises(ValueError):
        bm1387.new_work_asicboost(header, 1, [])
    with pytest.raises(ValueError):
        bm1387.new_work_asicboost(header, 1, [0, 1, 2, 3, 4])


def test_encode_layout():
    header = bytes(80)
    w = bm1387.new_work_from_header(header, 0x01020304)
    frame = w.encode()
    assert len(frame) == 50  # 4 + 32 + 12 + 2
    assert frame[0:4] == bytes([0x04, 0x03, 0x02, 0x01])  # work id little-endian
    want = bm1387.crc16(frame[:-2])
    got = (frame[-2] << 8) | frame[-1]
    assert got == want


def test_encode_asicboost_length():
    header = bytes(80)
    w = bm1387.new_work_asicboost(header, 1, [1, 2, 3, 4])
    assert len(w.encode()) == 146  # 4 + 4*32 + 12 + 2


def test_crc16_known_vector():
    # CRC-16/CCITT-FALSE("123456789") = 0x29B1.
    assert bm1387.crc16(b"123456789") == 0x29B1


def test_crc5_deterministic_and_width():
    data = bytes([0x05, 0, 0, 0, 0])
    a = bm1387.crc5(data, len(data) * 8 - 5)
    assert a == bm1387.crc5(data, len(data) * 8 - 5)
    assert a <= 0x1F


def test_bm1387_command():
    frame = bm1387.bm1387_command(bm1387.CMD_CHAIN_INACTIVE, 0, 0, broadcast=True)
    assert len(frame) == 5
    assert frame[0] & 0x80  # broadcast flag
    assert frame[4] <= 0x1F


def test_parse_nonce_response():
    frame = bytes([0x11, 0x22, 0x33, 0x44, 0x07, 0x02, 0x00])
    res = bm1387.parse_nonce_response(frame)
    assert res.nonce == 0x11223344
    assert res.work_id == 0x07
    assert res.midstate_index == 2
    with pytest.raises(ValueError):
        bm1387.parse_nonce_response(bytes([0x00]))
