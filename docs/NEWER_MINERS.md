# Newer ASIC Miner Support (S9 and beyond)

The original engine targeted the Antminer S2/S3 (BM1382) over direct USB. This project adds
a **miner profile registry** and a **BM1387 work encoder** so the same inference engine can
target newer Bitmain miners - the S9 family (BM1387) through the S17/S19/S21 generations -
and auto-detect which model is attached. All of it is pure Python and runs on Windows.

## Components

| Component | File | Purpose |
|-----------|------|---------|
| Miner profile registry | `ai_asic/hardware/miner_profiles.py` | Per-model specs + model auto-detection |
| BM1387 work encoder | `ai_asic/hardware/bm1387.py` | Midstate + CRC work protocol for the S9 chain |
| Model-aware detection | `ai_asic/hardware/device_detector.py` | Reports capabilities from the detected model |
| cgminer/bmminer client | `ai_asic/cgminer/client.py` | Network path to real S9+ hardware |

## Why the S9 needs a different path

- **BM1382 (S2/S3)** - the host pushes a full **80-byte Bitcoin header** to the chip over USB
  bulk transfers (Bitmain's `0x52` TXTASK token).
- **BM1387 (S9/S9i/S9j/T9+)** - the host pushes **pre-hashed work**: the SHA-256 *midstate*
  of the first 64 bytes of the header, plus the remaining 12 bytes (merkle-root tail + ntime
  + nbits). The chip rolls the 4-byte nonce internally. This is what `new_work_from_header`
  produces.

Most S9-class rigs run `cgminer`/`bmminer` on an on-board controller and expose a JSON-RPC
API on port 4028, so the host can drive them model-agnostically over the network
(`Protocol.CGMINER_API`) - the Windows-friendly path. The BM1387 encoder is for direct-drive
/ repurposing and for correct work generation + nonce interpretation.

## Supported models

`all_profiles()` covers:

- **BM1380 / BM1382 (USB):** S1, S2, S3 (original target)
- **BM1384 / BM1385:** S5, S7
- **BM1387 (16nm):** S9, S9i, S9j, S9k, S9 SE, T9+
- **BM1391 (7nm):** S15, T15
- **BM1397 (7nm):** S17, T17
- **BM1398 (7nm):** S19, S19 Pro, S19j Pro
- **BM1366 / BM1370 (5nm):** S19 XP, S21

Chip counts and hash rates are **nominal factory specifications** and vary by batch,
firmware, and tuning.

## Detection

`detect_model_hint()` resolves the attached miner in priority order:

1. The `ASIC_MODEL` environment variable, e.g. `ASIC_MODEL="Antminer S9"`.
2. The `Type` field from a reachable `cgminer`/`bmminer` API (`CGMINER_HOST`, default
   `127.0.0.1:4028`).
3. Fallback to the Antminer S3 default when nothing is discoverable.

```python
from ai_asic.hardware.miner_profiles import detect_profile, detect_model_hint

profile = detect_profile("Antminer S9")
caps = profile.capabilities(available=True)
print(caps["name"], caps["hash_rate"], profile.chip)
# ASIC Hardware (Antminer S9) 13500000000000 BM1387

hint = detect_model_hint()          # queries ASIC_MODEL / cgminer
profile = detect_profile(hint)
```

## BM1387 work encoding

```python
from ai_asic.hardware import bm1387

# Single midstate
work = bm1387.new_work_from_header(header_80, work_id)
frame = work.encode()               # [work_id:4][midstate:32][data:12][crc16:2]

# AsicBoost version rolling (up to 4 midstates)
work = bm1387.new_work_asicboost(header_80, work_id, [v0, v1, v2, v3])

# Parse a returned nonce
res = bm1387.parse_nonce_response(frame)   # .nonce, .work_id, .midstate_index
```

## Adding another model

Insert a `MinerProfile` into `_REGISTRY` in `miner_profiles.py`, keeping the list
**most-specific-first** (so `S19 Pro` resolves before `S19`, `S9j` before `S9`). Give it
aliases that appear in the miner's reported `Type` string, then add a case to
`tests/test_miner_profiles.py`.

## Limitations

- The outer FPGA framing on the S9 control board is firmware-specific; this encoder produces
  the canonical midstate + data + CRC payload that framing wraps. The nonce-response parser
  extracts nonce / work-id / midstate-index; exact high bits of the trailing bytes vary by
  firmware.
- CRC16 is CRC-16/CCITT-FALSE and CRC5 is the bmminer variant. If a specific firmware expects
  a different CRC convention, adjust `crc16` / `crc5`.
- Chip counts / hash rates are nominal; query the live cgminer API for exact per-rig values.
