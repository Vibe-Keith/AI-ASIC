# NOTICE

This project (AI-ASIC) is a derivative work of **HASHER** by guiperry:

- Upstream: https://github.com/guiperry/HASHER
- License: GNU General Public License v3.0 (GPL-3.0)

The original HASHER source was written in Go. This repository is a **Python port** of its
portable core, distributed under the same GPL-3.0 license (retained in [`LICENSE`](LICENSE)).

## Changes in this project

1. **Python port (cross-platform, Windows-compatible).** The Go sources were replaced with a
   pure-Python package (`ai_asic/`) that depends only on the Python standard library at
   runtime. See the "What was ported" table in [`README.md`](README.md) for the per-component
   mapping and for the Linux/hardware-only subsystems (eBPF, CUDA, MIPS on-device server, USB
   kernel driver, gRPC server, TUI, spaCy pipeline) that were intentionally not ported
   because they cannot run on Windows.

2. **Newer ASIC miner support.** Added a miner profile registry (Antminer S1 through S21),
   model auto-detection via `ASIC_MODEL` or the cgminer/bmminer JSON-RPC API, and a BM1387
   (Antminer S9 family / T9+) work encoder (SHA-256 midstate, 12-byte tail, CRC5/CRC16,
   chain command framing, AsicBoost version rolling, nonce-response parsing).

The original Go implementation remains available in this repository's git history and from
the upstream project.
