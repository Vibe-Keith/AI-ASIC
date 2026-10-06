# ASIC-assisted decoding: what changed, what it measured, what it cannot do

This document covers the chat pipeline's decoding optimizations: speculative drafts, retrieval
drafts, KV-block routing, asynchronous batched ASIC requests, and honest device accounting.
The goal is a real tokens/sec gain on a CPU, so the two numbers that matter are:

* **accepted tokens per verification pass** - every pass is one full forward pass of the model,
  so more tokens per pass means fewer passes per reply;
* **attention pairs** - query-key dot products per layer and head, prefill plus decode, counted
  exactly from what llama.cpp evaluates (including draft tokens it evaluates and then discards).

Both are measured by `ai_asic/chat/metering.py`, which wraps the model's `eval` and `sample`
methods. They are reported on every turn and by `ai-asic bench`.

## Results in one table

All numbers below come from the **weightless stand-in model** (`ai_asic/chat/simllm.py`). It
runs llama-cpp-python's own generate loop (KV prefix reuse, `eval`, `sample`, draft
verification and rewind) around a scripted reply. The counts follow real mechanics; real
acceptance on real text and real tokens/sec need a GGUF model, see
[Measuring on your machine](#measuring-on-your-machine).

| Change | Tokens per pass | Attention pairs |
|--------|-----------------|-----------------|
| Drafts: new default `4x16` vs the previous `3x10` lookup (echo / code / free) | 10.25 vs 8.20 / 2.33 vs 2.62 / 1.00 vs 1.00 | 0.91x / 0.72x / 0.56x of previous |
| Drafts: unused draft tokens per reply (echo / code / free) | - | 6 vs 14 / 22 vs 44 / 46 vs 118 |
| Retrieval drafts on repeated questions (`4x16r` vs `4x16`, 2 runs) | 2.90 vs 1.70 | 1.26x vs 1.46x of no-draft |
| Sticky KV routing, 11-turn chat, budget 384 | - | 0.94x total, 0.76x decode |
| Sticky KV routing, 40-turn chat, budget 768 | - | **0.08x total**, 0.43x decode |

The 40-turn result compares against the previous behaviour. Once a chat outgrew `n_ctx`, it
dropped the oldest exchange every turn, which changes the start of the prompt and forces
llama.cpp to re-prefill everything. Routing avoids that, and it also attends over a shorter
context at every decode step.

## What the ASIC does, and what it cannot do

A BM1387 does exactly one thing: roll a 32-bit nonce over an 80-byte block header and report
nonces whose double SHA-256 meets a target. It cannot multiply matrices, compute attention, run
a transformer layer, or even hash arbitrary bytes. Everything in an LLM forward pass therefore
runs on the CPU, and any speedup has to come from the CPU doing less work.

That is why accounting changed. **Only nonce searches count as ASIC work** (`DEVICE_ASIC` in
the trace, `native_hashes` in the hasher-server's `GetMetrics`):

* `HashAccelerator.hash` / `hash_batch` (fingerprints, seal digests) now always run
  on the host with `hashlib` and are reported as host work. Before, they were sent to the
  hasher-server and labeled ASIC. On real hardware that server runs them on the control
  board's CPU, so the label was wrong, and it cost a network round trip per draft step.
* `mine`, `submit_mine`, `mine_batch` and pool seals are the ASIC work. They run on the
  hasher-server's chip (virtual or real) or on the private pool's miner, and fall back to the
  host when neither is attached. The trace always names the device that actually ran each job.

Where the ASIC contributes:

* **LSH bucket IDs** (`ai_asic/chat/lsh.py`). Each band of a text's 64-bit SimHash is written into
  a header, and the first nonce meeting a low target defines the bucket (HASHER's "first valid
  nonce" signature). This is genuine native work, but it is worth being precise about its
  value. The bucket is a keyed hash of the band and adds no similarity information. A host
  SHA-256 would serve equally well. There are only 640 possible band headers, so after warm-up
  nearly every lookup is a cache hit.
* **Transcript seals.** These are proof-of-work over each turn's chained digest.

Neither makes the model faster by itself. The gains in the table come from the algorithms that
use those buckets (routing, retrieval) and from better drafts.

## Speculative drafts (`ai_asic/chat/draft.py`)

Prompt-lookup decoding: if the last few tokens appeared earlier, what followed them is a cheap
guess for what comes next. llama.cpp verifies the whole guess in one batched pass and keeps the
prefix it agrees with.

What changed:

* **Candidate batches with consensus** (`draft_mode: "consensus"`). Every earlier occurrence of
  the matched suffix contributes a continuation. This includes retrieved past replies (below).
  Longer matches weigh more. The draft follows the weighted majority and stops where
  candidates disagree.
* **Length by match strength** (`draft_len_by_match: [3, 8]`). Drafts backed only by a 1-token
  match are capped at 3 tokens, 2-token matches at 8. Weak matches ("the", "and") caused most of
  the wasted verification work. On free text the previous drafter evaluated 118 draft tokens
  that were never kept, and doubled attention work for no tokens-per-pass gain.
* **Longer drafts where matches are strong.** The default shape is now `4x16`: 4-gram matching,
  up to 16 tokens.
* **Adaptive length** (`draft_adaptive`). It is implemented but off, because it lowered tokens
  per pass in every scenario tested.

Trade-off: code rewriting with renames dropped from 2.62 to 2.33 tokens per pass. Renames
break the long matches, and the old drafter's long drafts off single-token matches occasionally
paid off there. Use `--draft-mode lookup` plus `"draft_len_by_match": []` in `config.json` to get
the previous behaviour back.

## Retrieval drafts (`ai_asic/chat/retrieval.py`)

Prompt lookup can only copy from the current context. Each finished reply is now stored
(`ai_models/cache/draft_store.jsonl`, last 500) under the LSH buckets of its prompt. At the start
of a turn, the prompt's buckets come from one batched nonce search, usually cache hits. They
pull up to two past replies whose prompts are similar (signature similarity at least 0.65), and
those replies become extra draft candidates. This helps when you ask similar things across
chats or sessions: tokens per pass went from 1.70 to 2.90 on repeated benchmark prompts. It
does nothing for novel questions. `--no-retrieval` turns it off.

## KV-block routing (`ai_asic/chat/routing.py`)

Every past exchange is KV-cache entries that every forward pass attends over. Routing decides
which exchanges enter the prompt and so the cache: the system prompt, the newest
`route_keep_recent` exchanges, then older exchanges whose ASIC LSH buckets collide with the new
message's (best similarity first), then the most recent remaining ones, within the budget.

**The first design failed, and the measurement is why routing is sticky.** Re-selecting the
blocks every turn changes the prompt prefix, and llama.cpp only reuses the KV cache for an
unchanged prefix. In a live 11-turn chat, per-turn routing at budget 384 **doubled** total
attention (2.1-2.3x per turn), because each turn re-prefilled about 300 tokens instead of 43.
It did attend over less during decode (0.75x).

Sticky routing:

* While the prompt fits `route_budget_tokens` (768), nothing is dropped.
* When it exceeds the budget, a **compaction** rebuilds the history down to `route_low_water`
  (0.6) times the budget.
* Later turns keep that selection and append new exchanges, so the KV prefix is reused until
  the next compaction.
* **Recall:** if the new message strongly matches a dropped exchange (bucket collision and
  similarity at least `route_recall_sim`, 0.8), it compacts early so that exchange comes back.

Measured, attention summed over every turn of a live chat versus full history:

| Chat | Budget | Compactions | Total attention | Decode attention |
|------|--------|-------------|-----------------|------------------|
| 11 turns (~600 tokens) | 512 | 1 | 1.04x | 0.91x |
| 11 turns | 384 | 2 | 0.94x | 0.76x |
| 40 turns (~2,500 tokens, beyond `n_ctx`) | 1536 | 2 | 0.14x | 0.78x |
| 40 turns | 1024 | 4 | 0.10x | 0.55x |
| 40 turns | 768 | 6 | 0.08x | 0.43x |

In short chats routing is roughly neutral; each compaction costs one re-prefill. In long chats
it is a large saving. Low-water 0.5 was slightly cheaper (0.07x at 768) but drops more context
per compaction; 0.6 is the default.

**Quality risk.** A dropped exchange is information the model no longer sees. The stand-in's
replies are scripted, so they cannot show that. The routing benchmark therefore reports whether
the final replies with a real model match the full-history ones. Selection is at exchange
granularity, because that is what a chat template can express. Per-token, per-step KV page
selection (Quest-style) would need changes inside llama.cpp's attention kernels.

## Asynchronous, batched, warm ASIC requests (`ai_asic/chat/accelerator.py`)

* `submit_mine` / `submit_mine_batch` return futures immediately. One worker thread drains the
  queue and sends everything waiting as a single `MineBatch` request, up to 256 searches per
  round trip. `MineBatch` is new on the hasher-server; older servers fall back to one `Mine` per
  job.
* There is no per-token RPC. Draft-key hashing is local, and bucket searches happen once per
  turn (query) or in the background (each finished exchange is bucketed right after the turn,
  so routing the next turn finds it cached).
* **Keep-warm:** while idle the worker runs a tiny search every 20 s (`warm_interval`), so the
  connection and device path are never cold. Warm-up searches are metered separately
  (`HashAccelerator.warm`) and never attributed to a turn. In pool mode, the miner already
  hashes an idle job between seals.
* **Seals run in the background.** The reply returns without waiting. The chain digest is fixed
  synchronously, and one writer thread per log file keeps records in order. `TurnResult.seal`
  and `ChatEngine.flush()` wait when you need the record. The CLI flushes before exiting.
* The virtual chip now searches with `hashlib` resumed from the midstate. It still ingests the
  CRC-checked BM1387 frame and confirms each winning nonce through the cycle-honest
  midstate-only model, so results are identical to the pure-Python chip at a few hundred times
  the speed.

## Not done: ASIC-derived residual adapters inside the transformer

The request was to add a small ASIC-derived residual adapter every few transformer layers. It
is not implemented, for three concrete reasons:

1. **Nothing to plug into.** llama-cpp-python cannot run Python code between layers. The only
   per-layer hook llama.cpp exposes is a *control vector*: one fixed vector added to the
   residual stream, the same for every token in a call. That is a per-turn bias, not
   input-dependent nonlinear features.
2. **Untrained features are noise.** A residual derived from nonce-search outputs is a
   pseudo-random function of its input. Adding it to the residual stream of a trained model
   perturbs every layer after it and degrades output, unless the adapter is trained. Training
   needs gradients through the model, which llama.cpp inference does not provide.
3. **It doesn't move either target metric.** An adapter adds work to every forward pass and
   changes no pass count or attention size. Its only possible value is quality, and only after
   training.

Workable options:

* **Logit-level hash n-gram bias.** Through llama-cpp-python's `logits_processor`, mix in a
  hashed n-gram "cache LM" keyed by the same ASIC buckets (kNN-LM style). It is implementable
  and measurable, but it changes the model's outputs, and it would inflate draft acceptance by
  making the model copy more. Every gain would need checking against quality.
* **A trained hashed-embedding adapter (Engram-style).** Train a table of vectors indexed by
  ASIC bucket IDs, added at chosen layers, in PyTorch against the original model, then export it
  to llama.cpp. This is a real research project: a training pipeline, a GPU, evaluation sets.

## Measuring on your machine

```bash
pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m ai_asic.cli models download qwen2.5-0.5b

# Drafts: off, previous drafter (3x10l), new default (4x16), new default + retrieval (4x16r)
python -m ai_asic.cli bench --configs off,3x10l,4x16,4x16r --runs 2

# KV routing in a live multi-turn chat; check "Same final replies as full"
python -m ai_asic.cli bench --suite route --budgets off,768,512
```

Read the summary lines. `Tok/pass` and `Attn vs off` are the targets. `Decode tok/s` is the
truth. The earlier note in the README still applies: drafting is not free in llama-cpp-python,
because it keeps logits for every position and checks drafts in Python. So a higher
tokens-per-pass does not guarantee more tokens per second on a small model.

## Configuration (`ai_models/config.json`)

| Key | Default | Meaning |
|-----|---------|---------|
| `draft_max_ngram`, `draft_num_pred` | 4, 16 | longest n-gram matched, most tokens per draft |
| `draft_mode` | `consensus` | or `lookup` (single most recent occurrence) |
| `draft_len_by_match` | `[3, 8]` | draft caps for 1- and 2-token matches |
| `draft_max_candidates` | 8 | continuations pooled per lookup |
| `draft_adaptive` | false | adaptive draft length |
| `draft_retrieval`, `draft_retrieval_k`, `draft_retrieval_min_sim` | true, 2, 0.65 | retrieval drafts |
| `route_context`, `route_budget_tokens` | true, 768 | KV-block routing and its budget |
| `route_keep_recent`, `route_low_water`, `route_recall_sim` | 2, 0.6, 0.8 | routing policy |
| `lsh_bands`, `lsh_band_bits` | 10, 6 | SimHash banding (640 possible bucket headers) |
| `bucket_difficulty`, `bucket_max_nonces` | 6, 4096 | nonce search per bucket |
| `seal_async` | true | seal in the background |

## Caveats

* Attention pairs are logical causal pairs. llama.cpp pads the KV length its kernels iterate
  over, so actual kernel work is somewhat higher. The ratios between configurations are what
  matter.
* On real multi-core silicon, the first *reported* valid nonce depends on which core finds one
  first. For stable bucket IDs, a real-hardware hasher-server must return the lowest valid nonce
  in the searched range. The virtual chip and the host fallback search upward from 0, so they
  are deterministic.
* The stand-in model's numbers show mechanics, not real-text acceptance. The tokens/sec it
  prints are meaningless.
