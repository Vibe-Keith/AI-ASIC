<p align="center">
<img src="assets/logo-clear.png" width="350" alt="logo">
</p>

# AI-ASIC

**SHA-256 hash neural-network inference on repurposed Bitcoin mining hardware.**

AI-ASIC is a pure-Python, cross-platform (Windows / macOS / Linux) implementation of the
[HASHER](https://github.com/guiperry/HASHER) engine (GPL-3.0), extended with support for
**newer Bitmain ASIC miners** - the Antminer S9 family (BM1387) through the S17 / S19 / S21
generations - with automatic model detection.

It treats a mining ASIC as a deterministic SHA-256 engine: by mining at a "Difficulty 1"
target, the first valid nonce is used as a locality-sensitive hash signature, which drives
a hash-based neural network and a recursive temporal-ensemble inference loop.

> Ported from Go to Python. The engine core uses **only the Python standard library** (no
> third-party runtime dependencies), so it runs on a clean Python 3.8+ install on Windows.

## Why Python / what runs on Windows

The original project had large Linux- and hardware-only subsystems (an eBPF tracer, CUDA
kernels, a MIPS cross-compiled on-device gRPC server, a `/dev/bitmain-asic` kernel driver,
and a spaCy C++ data pipeline). This port keeps the portable, useful core and reaches real
miners the way they are actually driven in the field - over the **cgminer / bmminer JSON-RPC
network API** (TCP port 4028), which works from Windows with no drivers. The on-device gRPC
server - the piece the AI pipeline actually depended on - is now ported to pure Python
(`ai_asic/server/`), so the host/ASIC split runs end-to-end without the MIPS toolchain; see
[The on-device hasher-server](#the-on-device-hasher-server). See also
[What was ported](#what-was-ported).

## Install

```bash
# From the repo root (Python 3.8+)
pip install -e .
```

On Windows (PowerShell or cmd) with the official python.org build:

```powershell
py -m pip install -e .
```

No install is required to run it - you can use the launchers or `python -m ai_asic.cli`.

## Quick start

```bash
# List every supported miner model
python -m ai_asic.cli profiles

# Detect the attached miner (uses ASIC_MODEL override or the cgminer API)
python -m ai_asic.cli detect
ASIC_MODEL="Antminer S9" python -m ai_asic.cli detect

# Run recursive inference on text (software backend)
python -m ai_asic.cli infer "optical alignment with AI" --passes 21 --classes 10

# Drive a real S9 (or newer) over the network via cgminer
python -m ai_asic.cli infer "hello" --asic --host 192.168.1.50 --port 4028

# Build a BM1387 (S9) work frame from an 80-byte header
python -m ai_asic.cli encode-work --work-id 42

# Simulate a BM1387 chip end-to-end and verify it actually mines (no hardware needed)
python -m ai_asic.cli simulate --difficulty 16 --model "Antminer S9"

# Run an AI workload split across the host CPU and the ASIC miner
python -m ai_asic.cli workload "optical alignment with AI" --difficulty 12

# Chat with a local LLM: CPU runs the model, the ASIC does native nonce search (see "Chatbot")
python -m ai_asic.cli models download qwen2.5-0.5b
python -m ai_asic.cli chat
```

On Windows you can also use the launchers: `run.bat detect` or `.\run.ps1 detect`.

## Graphical interface

A Tkinter GUI ties every capability above (detect, simulate, workload, chat, profiles,
infer, encode-work) into one window. Tkinter ships with the standard library, so there is
nothing extra to install on a normal Windows/macOS Python build.

The **Simulate** tab starts a virtual BM1387 miner that serves the real cgminer/bmminer
API in-process, so the **Detect** tab finds it exactly like real hardware (and the Infer
tab's ASIC backend can drive it). It also runs the midstate mining self-test with
pass/fail verification. No miner required.

```bash
python -m ai_asic.gui
```

On Windows use `run-gui.bat` or `.\run-gui.ps1`. Long-running detection and
inference run on background threads, so the window stays responsive; switch the
Infer tab's "Use cgminer ASIC backend" checkbox on to target a real miner.

(On a minimal Linux install Tkinter may be a separate package, e.g.
`sudo apt install python3-tk`.)

## Library usage

```python
from ai_asic import HashNetwork, RecursiveEngine, SoftwareHashMethod, detect_asic
from ai_asic.hardware.miner_profiles import detect_profile
from ai_asic.hardware import bm1387

# Detect / describe a miner
print(detect_asic())
profile = detect_profile("Antminer S9")
print(profile.chip, profile.nominal_hashrate)   # BM1387 13500000000000

# Recursive inference
net = HashNetwork(input_size=64, hidden1=128, hidden2=64, output_size=10)
engine = RecursiveEngine(net, passes=21, jitter=0.01, hash_method=SoftwareHashMethod())
result = engine.infer(b"input bytes")
print(result.consensus.prediction, result.consensus.confidence)

# BM1387 (S9) work encoding: midstate + 12-byte tail + CRC16
work = bm1387.new_work_from_header(header_80_bytes, work_id=1)
frame = work.encode()
```

## Supported miners

| Generation | Chip | Models |
|-----------|------|--------|
| USB (original HASHER target) | BM1380 / BM1382 | S1, S2, S3 |
| 28nm control-board | BM1384 / BM1385 | S5, S7 |
| 16nm | BM1387 | **S9, S9i, S9j, S9k, S9 SE, T9+** |
| 7nm | BM1391 / BM1397 / BM1398 | S15, T15, S17, T17, S19, S19 Pro, S19j Pro |
| 5nm | BM1366 / BM1370 | S19 XP, S21 |

Chip counts and hash rates are **nominal factory specs** and vary by batch/firmware.
Details and how to add a model: [`docs/NEWER_MINERS.md`](docs/NEWER_MINERS.md).

## How the S9 differs (BM1387)

- **BM1382 (S2/S3)** - the host pushes a full **80-byte header** to the chip over USB.
- **BM1387 (S9 family / T9+)** - the host pushes a SHA-256 **midstate** of the first 64
  header bytes plus the 12-byte tail (merkle-tail + ntime + nbits); the chip rolls the
  nonce. `ai_asic/hardware/bm1387.py` implements the midstate, 4-way AsicBoost version
  rolling, CRC5 command framing, CRC16 work framing, and nonce-response parsing.

Most S9-class rigs are driven over the cgminer/bmminer API (`ai_asic/cgminer/client.py`),
which is model-agnostic and the Windows-friendly path to real hardware.

## Simulating a chip without hardware

`ai_asic/hardware/simulator.py` is a software model of a BM1387 chip, so you can prove the
work/mine/nonce pipeline is correct with no miner attached:

```bash
python -m ai_asic.cli simulate --difficulty 16 --model "Antminer S9"
```

The virtual chip ingests the same frame `BM1387Work.encode()` produces (rejecting a bad
CRC16, like real silicon), mines using **only** the 32-byte midstate and 12-byte tail - it
finishes block 2 from the midstate and runs the second SHA-256, exactly the two compressions
a real core does per nonce - then returns a golden nonce in the real 6-byte response frame.
The run then independently re-hashes the **full 80-byte header** with `hashlib` and confirms
the chip's nonce genuinely solves it. If `midstate math == hashlib over full header`, the
host-side encoder in this repo would drive a real S9.

`--difficulty` sets the required leading zero bits (16 ≈ 65k rolls); the reported real-chip
time is scaled to the model's nominal hash rate. By default the virtual chip rolls nonces with
`hashlib` resumed from the same midstate and re-derives the winning nonce through the
pure-Python chip model (identical result, about 13 ms instead of 7 s at difficulty 16);
`--cycle-model` rolls every nonce through the pure-Python model. Library use:

```python
from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.hardware.simulator import SimConfig, simulate_header

header = prepare_asic_job([1] * 12, 0, timestamp=0)
sim = simulate_header(header, config=SimConfig(difficulty_bits=16))
print(sim.ok, hex(sim.result.nonce), sim.result.hash_hex)
```

## AI workloads split across host and ASIC

`ai_asic/workloads/split_model.py` is the project's thesis made runnable: a hash-based
classifier whose layers are partitioned across the two devices by what each is good at.

| Stage | Device | Work |
|-------|--------|------|
| encoder | **host CPU** | dense SHA-256 hash layers → feature vector |
| mining layer | **ASIC miner** | project features into block headers, mine a golden nonce for each |
| head | **host CPU** | SHA-256 hash layer → class logits → prediction |

The mining layer is the one operation a SHA-256 ASIC natively accelerates (a nonce search),
so that is the layer offloaded to the chip; the rest stays on the host. Run it:

```bash
# Host only (software nonce search)
python -m ai_asic.cli workload "optical alignment with AI" --no-asic

# Split: run the on-device hasher-server first, so the mining layer runs on the ASIC
python -m ai_asic.cli serve --port 8888 &
python -m ai_asic.cli workload "optical alignment with AI" --difficulty 12
```

The output is a per-stage execution trace showing exactly what ran where:

```
Stage     Device       Ops        Time  Detail
encoder   host-cpu     112    155.9 us  16-dim feature vector
mining    asic           4      2.38 s  4 headers, difficulty 12 bits
head      host-cpu      10     41.3 us  10 class logits
Ran on hardware:   True
```

The host and ASIC paths produce the **same prediction and the same nonces** - only the
device that performed the nonce search changes, which is the entire point of the split.
In the GUI, the **Workload** tab does this interactively (and loads/saves models):

```python
from ai_asic.workloads.split_model import SplitModel, MiningBackend

model = SplitModel(mining_neurons=4, output_size=10, difficulty_bits=12)
model.save("mymodel.json")                       # loading AI models
backend = MiningBackend("127.0.0.1", 8888)       # target the on-device hasher-server
result = model.infer(b"optical alignment with AI", backend)
print(result.prediction, result.trace.asic_is_hardware)
```

## The on-device hasher-server

A stock Antminer over the cgminer API mines pool work; it cannot be told to mine an arbitrary
header and return the nonce. The original HASHER project solved this with a **custom on-device
server** - a Go gRPC service (`hasher.v1.HasherService`) cross-compiled to a MIPS binary,
deployed onto the miner, talking to `/dev/bitmain-asic`. That was the one Linux-only piece the
whole AI pipeline depended on, and it is now ported in `ai_asic/server/`:

| Original | Port |
|----------|------|
| gRPC + protobuf over HTTP/2 | stdlib JSON-over-TCP (same framing as the cgminer client, zero deps) |
| `ComputeHash` / `ComputeBatch` / `StreamCompute` | same methods (`StreamCompute` → batch on this transport) |
| `GetMetrics` (eBPF trace points) | in-process counters: requests, bytes, hashes, avg/peak latency, errors |
| `GetDeviceInfo` | device/profile capabilities |
| `/dev/bitmain-asic` driver | `ChainAsicDevice` (live only on a miner's board) + `VirtualAsicDevice` (everywhere) |
| — | `Mine` (native nonce search) added, so the split-model offloads real work |

Run it standalone and point the workload at it:

```bash
python -m ai_asic.cli serve --model "Antminer S9" --port 8888   # auto-picks the real chain if present
python -m ai_asic.cli workload "optical alignment with AI"       # mines on the hasher-server
```

```python
from ai_asic.server import HasherServer, HasherClient, VirtualAsicDevice

srv = HasherServer(device=VirtualAsicDevice("Antminer S9"), port=8888).start()
cli = HasherClient("127.0.0.1", 8888)
cli.compute_hash(b"hello")          # SHA-256 on the chip's hash core
cli.mine(header_80_bytes, difficulty_bits=16)   # native nonce search
cli.metrics()                       # the eBPF-style counters
```

`serve` with `--device auto` uses `ChainAsicDevice` when `/dev/bitmain-asic` exists (on the
miner's own control board) and the virtual chip otherwise, so the exact same server binary -
and the exact same split-model - runs on real hardware or on a laptop with no code change. In
the GUI, the Simulate tab starts this hasher-server automatically alongside the cgminer API.

## Chatbot: local LLM, CPU decoding, ASIC nonce search

`ai_asic/chat/` adds a real chatbot. A SHA-256 mining ASIC **cannot** run a language model's
neural network (matrix multiplies, attention, sampling), so llama.cpp runs it on the CPU. What
this project optimizes is how much work the CPU has to do per reply, and it uses the ASIC for
the one operation a BM1387 natively performs: **nonce search** over an 80-byte header.

| Stage | Device | What it does |
|-------|--------|--------------|
| fingerprint | host | SHA-256 content address of (model, conversation, sampling) → response cache; an exact repeat is answered without running the LLM |
| route.buckets | **ASIC** | LSH bucket IDs for the new message (and any history exchange not yet bucketed) by native nonce search: one batched request, mostly cache hits |
| route | host | **KV-block routing**: once the prompt outgrows a budget, keep the newest exchanges plus the old ones whose buckets match the new message, so the CPU attends over a shorter context |
| retrieve | host | past replies to similar prompts (found by bucket) become extra draft candidates |
| llm | host | llama.cpp prefill, verification passes, sampling - with exact counts of verification passes and attention work |
| draft | host | speculative drafts: n-gram index (token-tuple keys) with consensus over every earlier occurrence and length capped by match strength |
| seal | **ASIC** | proof-of-work nonce over each turn's chained transcript digest, mined in the background → tamper-evident log |

**Only nonce searches are counted as ASIC work.** Hashing arbitrary data (fingerprints, draft
keys, digests) is not something a mining chip can do, so it runs on the host and the trace says
so; it also costs no network round trip. Nonce searches are asynchronous and batched (one
`MineBatch` request per group, never one per token), and the accelerator keeps the device path
warm with a tiny search while idle. Every reply prints the device that ran each stage and the two
numbers that predict real speed on a CPU: **accepted tokens per verification pass** and
**attention pairs** (query-key work per layer and head, including draft tokens evaluated and not
kept):

```
Accelerator: ASIC via hasher-server @ 127.0.0.1:45699 (BM1387); nonce search on the ASIC, other hashing on host CPU
Verification passes: 3   accepted tokens/pass: 9.33   attention pairs: 3,741   unused draft tokens: 10
Stage         Device          Ops  Time (ms)  Detail
fingerprint   host-cpu          1        0.1  b1dbffd16f5c8fb8... cache miss
route.buckets asic            684        9.5  10 nonce searches in 1 round trip(s), 0 cached (LSH bucket IDs)
route         host-cpu          0        1.0  fits budget; 55 prompt tokens
retrieve      host-cpu          0        0.0  no similar past reply (0 stored)
llm           host-cpu         28        0.6  28 tokens, ...; 3 verification passes, 86 tokens evaluated (0 KV reused), 3,741 attention pairs
draft         host-cpu        274        0.1  3 lookups, 35 proposed from 3 candidates, 19/19 accepted (100%), 9.33 tok/pass, 10 unused draft tokens
seal          asic            910        4.6  nonce 0x0000038d at 10 bits (background)
```

(That trace is from the weightless stand-in model, so its speeds are meaningless; the counts are
real.) Design, measurements and limits: [`docs/ASIC_DECODING.md`](docs/ASIC_DECODING.md).

### Measure it on your machine

```bash
# Speculative drafts: off vs the old single lookup (3x10l) vs the new default (4x16)
python -m ai_asic.cli bench --model qwen2.5-0.5b-instruct-q4_k_m.gguf --runs 2

# KV-block routing in a long live chat: full history vs routed at two budgets
python -m ai_asic.cli bench --suite route --budgets off,768,512

# Either suite without a model or llama-cpp-python (stand-in; counts only)
python -m ai_asic.cli bench --sim
```

### Setup

```bash
# 1. Runtime (optional extra; prebuilt CPU wheel, no compiler needed)
pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 2. A model: download the default (469 MB, SHA-256 verified) or drop any .gguf in ai_models/llm/
python -m ai_asic.cli models download qwen2.5-0.5b
python -m ai_asic.cli models            # show what's in ai_models/
```

### Use

```bash
python -m ai_asic.cli serve --port 8888 &            # the ASIC (virtual chip, or a real board)
python -m ai_asic.cli chat --trace                   # interactive; /new, /trace, /quit
python -m ai_asic.cli chat --once "Explain SHA-256"  # one turn
python -m ai_asic.cli chat --verify                  # re-check every transcript seal
```

In the GUI, open the **Chat** tab. Start the virtual miner on the Simulate tab first if you want
the nonce searches (LSH buckets, seals) on the ASIC; otherwise they run on the host. The **KV
routing** checkbox turns context routing on or off. Useful chat flags: `--no-route`,
`--route-budget N`, `--draft-mode lookup`, `--no-retrieval`.

### Real hardware (closed loop, no outside servers)

A stock miner (S9 bmminer and later firmware) takes work only from a mining pool and only
hashes block headers, so the host runs its own **private Stratum pool** and the miner connects
to it over the LAN. Nothing leaves your network: the host sends jobs, the miner returns shares.
Each chat turn's seal is a job whose coinbase commits to the turn's digest; the share the miner
returns is proof-of-work over it, verifiable later with `hashlib` alone (`chat --verify`).
Fingerprint and draft hashing stay on the host - a mining ASIC cannot hash arbitrary data.

```bash
# 0. Dry run of the whole loop with a built-in software miner (no hardware)
python -m ai_asic.cli hwtest --virtual

# 1. Prove the loop on the miner (prints the pool URL to give the miner)
python -m ai_asic.cli hwtest --miner 192.168.1.50                    # set Pool 1 in the web UI
python -m ai_asic.cli hwtest --miner 192.168.1.50 --configure-miner  # or via the miner's API

# 2. Chat with seals mined on the miner
python -m ai_asic.cli chat --stratum --miner 192.168.1.50 --trace
```

`hwtest` reports the miner's firmware, version-rolling (AsicBoost) mode, accepted/rejected
shares, observed hashrate, and per-seal latency, and verifies every seal. Notes: allow Python
through the Windows firewall on Private networks (inbound TCP 3333); remove other pools from
the miner if it must never fail over to the internet when the host stops; the miner hashes at
full power on an idle job between seals. `--share-difficulty` (default 256, ~12 shares/s on an
S9) trades seal latency against share traffic.

### The `ai_models/` folder

```
ai_models/
  config.json   chat defaults: model, system prompt, n_ctx, sampling, draft/routing/cache/seal options
  llm/          GGUF language models (*.gguf)
  hasher/       trained HASHER split-models (*.json) - the Workload tab saves here
  cache/        responses.jsonl (response cache), transcripts.jsonl (sealed chat log),
                draft_store.jsonl (past replies used as draft candidates)
```

Weights and caches are git-ignored. Set `AI_ASIC_MODELS` to keep the folder elsewhere (for
example off a synced drive; model files are hundreds of MB).

### What to expect (measured on this machine, Qwen2.5-0.5B Q4_K_M)

These numbers were measured before the draft and routing changes described in
[`docs/ASIC_DECODING.md`](docs/ASIC_DECODING.md); re-run `bench` to see the current ones.

- **Drafts help when a reply repeats its context**: 122 tok/s with drafts against about 91 without
  on a quoting prompt. They cost speed when there is nothing to look up (41 vs 59 tok/s on a
  first turn). Turn them off with `--no-draft` or the checkbox.
- **Memory**: about 550 MB peak working set without drafts, about 730 MB with them (llama-cpp-python
  keeps per-position logits when a draft model is attached, which is why `n_ctx` defaults to 2048).
- **Cache hits** answer in about 1 ms instead of a full generation.
- **Seals** now run in the background and no longer delay the reply. The virtual chip searches
  with `hashlib` from the midstate and confirms each winning nonce with the cycle-honest model,
  so a seal takes a few ms. `--seal-difficulty 0` disables them.
- **Quality** is that of a 0.5B model: fluent, but often wrong on facts. Drop a larger instruct
  GGUF (1-3B) into `ai_models/llm/` and pick it in the Chat tab or with `--model` for better answers.

## What was ported

| Area | Status | Where |
|------|--------|-------|
| Miner profiles + model detection | ✅ ported | `ai_asic/hardware/miner_profiles.py` |
| BM1387 (S9) work encoder | ✅ ported | `ai_asic/hardware/bm1387.py` |
| Virtual BM1387 chip simulator | ✅ new | `ai_asic/hardware/simulator.py` |
| On-device hasher-server (was MIPS/gRPC) | ✅ ported | `ai_asic/server/` |
| Host/ASIC split workload (model) | ✅ new | `ai_asic/workloads/split_model.py` |
| Chatbot (local LLM, speculative drafts, KV routing, ASIC nonce search) | ✅ new | `ai_asic/chat/`, `ai_models/` |
| Bitcoin header construction | ✅ ported | `ai_asic/hardware/bitcoin_header.py` |
| Device detection | ✅ ported | `ai_asic/hardware/device_detector.py` |
| cgminer/bmminer client | ✅ ported | `ai_asic/cgminer/client.py` |
| Private Stratum pool (closed-loop host -> real miner) | ✅ new | `ai_asic/stratum/` |
| Hash neuron / mining neuron | ✅ ported | `ai_asic/hashing/neuron.py` |
| Hash network | ✅ ported | `ai_asic/hashing/network.py` |
| Recursive inference engine | ✅ ported | `ai_asic/hashing/recursive.py` |
| Hashing backends (software + ASIC) | ✅ ported | `ai_asic/hashing/methods.py` |
| CLI | ✅ rewritten | `ai_asic/cli.py` |
| eBPF tracer, CUDA kernels, USB kernel driver, TUI, spaCy pipeline | ❌ not ported | Linux/hardware-only; do not run on Windows. Reach real miners via the cgminer API instead. |

## Performance

SHA-256 itself runs through `hashlib` (OpenSSL C). `hashlib` only releases the GIL for inputs
of 2 KiB or more, so batches of small inputs run in a plain loop (threads made them 48x slower).

Every software nonce search (split-model mining layer, host fallback, virtual chip, seals) goes
through `ai_asic/hashing/nonce_search.py`: the header's first 76 bytes are hashed once and the
`hashlib` state copied per nonce, and the target is a byte comparison. Searches expected to take
at least 2^20 hashes are split across a process pool in nonce order, so the first valid nonce is
the same as a serial search. Measured on a 4-core VM:

| Search | Before | Now |
|--------|--------|-----|
| serial software nonce search | 827 kH/s | ~1.2 MH/s |
| 22-bit search (19M hashes) | ~23 s (at 827 kH/s) | 3.7 s (4 processes) |
| `simulate --difficulty 16` | 7.3 s | 13 ms |

Set `AI_ASIC_WORKERS=1` to keep searches in one process. The highest throughput still comes from
the ASIC, not the host CPU.

For chat, llama.cpp does nearly all the work. Its CPU settings live in `ai_models/config.json`
(`n_threads`, `n_threads_batch`, `n_batch`, `n_ubatch`, `flash_attn`, `kv_cache_type`) and on
the command line (`chat --threads N --flash-attn --kv-cache q8_0`). Measured on a 4-vCPU VM with
a random-weight model shaped like Qwen2.5-0.5B (Q4_K_M, 700-token context, best of 3):

| llama.cpp setting | Prompt tok/s | Decode tok/s |
|-------------------|-------------:|-------------:|
| defaults (2 decode threads) | 268 | 21.8 |
| `n_threads` 4 | 269 | 35.4 |
| `flash_attn` | 303 | 17.4 |
| `flash_attn` + `n_threads` 4 | 325 | 29.6 |
| `flash_attn` + `n_threads` 4 + q8_0 KV | 240 | 37.2 |

The thread count matters most. On a CPU with SMT/hyperthreading the physical core count is
usually best, so try `--threads` at both the physical and the logical core count. Flash
attention is off by default because it slowed decoding here.

## Tests

```bash
pip install pytest
pytest -q
```

## License

GPL-3.0 (inherited from HASHER). See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
