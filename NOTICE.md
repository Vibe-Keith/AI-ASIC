# NOTICE

This project (AI-ASIC) is a derivative work of **HASHER** by guiperry:

- Upstream: https://github.com/guiperry/HASHER
- License: GNU General Public License v3.0 (GPL-3.0)

The original HASHER source is incorporated here under the terms of the GPL-3.0, which is
retained in [`LICENSE`](LICENSE). As a derivative work, this repository is also
distributed under the GPL-3.0.

## Modifications in this fork

Support for newer Bitmain ASIC miners beyond the original Antminer S2/S3 (BM1382) target:

- `pkg/hashing/hardware/miner_profiles.go` - miner profile registry (S1 through S21) and
  model auto-detection via `ASIC_MODEL` / the cgminer-bmminer JSON-RPC API.
- `pkg/hashing/hardware/bm1387_header.go` - BM1387 (Antminer S9 family / T9+) work
  encoder: SHA-256 midstate computation, 12-byte work tail, CRC5/CRC16, chain command
  framing, AsicBoost version rolling, and nonce-response parsing.
- `pkg/hashing/hardware/device_detector.go` - model-aware ASIC detection that reports
  capabilities from the detected miner profile rather than a hard-coded Antminer S3.
- Tests: `pkg/hashing/hardware/miner_profiles_test.go`,
  `pkg/hashing/hardware/bm1387_header_test.go`.
- Documentation: `docs/NEWER_MINERS.md`.

The Go module path remains `hasher` for upstream source compatibility.
