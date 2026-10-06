"""Chat engine: a local GGUF LLM on the CPU; native BM1387 nonce search on the ASIC.

Per turn (the device in brackets is what the trace reports):

    [host] fingerprint    SHA-256 content address of (model, conversation, sampling) -> response
                          cache; an exact repeat is answered without running the LLM
    [ASIC] route.buckets  LSH bucket IDs for the new message (and any history exchange not yet
                          bucketed) by native nonce search - one batched request, mostly cache
                          hits after warm-up
    [host] route          KV-block routing: which history exchanges enter the prompt, and so
                          the KV cache the CPU attends over (only once the prompt outgrows the
                          budget)
    [host] retrieve       past replies to similar prompts, found by bucket, become draft
                          candidates
    [host] llm            llama.cpp prefill, verification passes and sampling, with exact counts
                          of evaluated tokens, verification passes and attention pairs
    [host] draft          the n-gram draft index (token-tuple keys, consensus drafts)
    [ASIC] seal           proof-of-work nonce over the turn's chained transcript digest, mined in
                          the background - the reply does not wait for it

Only nonce searches count as ASIC work. A BM1387 cannot hash arbitrary data, so fingerprint
and digest hashing (and the draft index) run on the host and the trace says so. When no
hasher-server or miner is attached, the nonce searches run on the host too and are reported
as such.
"""
from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ai_asic.chat.accelerator import HashAccelerator
from ai_asic.chat.draft import HashDraftIndex, make_llama_draft_model
from ai_asic.chat.lsh import BucketHasher
from ai_asic.chat.metering import EvalStats, LlamaMeter
from ai_asic.chat.models_dir import cache_dir, ensure_layout, load_config, resolve_llm
from ai_asic.chat.retrieval import DraftStore
from ai_asic.chat.routing import ContextRouter, RouteResult
from ai_asic.stratum import protocol as stratum_protocol
from ai_asic.workloads.split_model import (DEVICE_ASIC, DEVICE_HOST, ExecutionTrace, Stage,
                                           _software_mine)

_GENESIS = "00" * 32
_NBITS = 0x1D00FFFF


class _SealStage:
    """Adds the seal stage to a turn's trace once the background seal finishes (idempotent)."""

    def __init__(self, trace: ExecutionTrace, difficulty: int):
        self.trace = trace
        self.difficulty = difficulty
        self._done = False
        self._lock = threading.Lock()

    def __call__(self, fut: Future) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
            try:
                seal = fut.result()
            except Exception as exc:
                self.trace.add(Stage("seal", DEVICE_HOST, 0, 0.0, f"failed: {exc}"))
                return
            on_miner = seal.get("kind") == "stratum"
            detail = f"nonce 0x{seal['nonce']:08x} at {self.difficulty} bits"
            if on_miner:
                detail += (f", {_leading_zeros_display(seal['pow_hash'])} zero bits from "
                           f"{seal['miner']} in {seal['miner_ms']:.0f} ms")
            device = DEVICE_ASIC if on_miner else seal.get("nonce_device", DEVICE_HOST)
            self.trace.add(Stage("seal", device, ops=int(seal.get("hashes", 0)),
                                 seconds=float(seal.get("seal_ms", 0.0)) / 1000.0,
                                 detail=detail + " (background)"))


@dataclass
class TurnResult:
    text: str
    cached: bool
    trace: ExecutionTrace
    seal_future: Optional[Future] = None
    tokens: int = 0            # generated tokens (the reply re-tokenized, not stream chunks)
    tokens_per_s: float = 0.0  # tokens / whole generation time, prompt processing included
    ttft_s: float = 0.0        # time to first token: prompt processing + first forward pass
    decode_tps: float = 0.0    # tokens after the first / time after the first
    draft_lookups: int = 0     # ~ verification forward passes when drafting
    draft_proposed: int = 0
    draft_scored: int = 0
    draft_accepted: int = 0
    eval: Optional[EvalStats] = None      # exact CPU work, when the model exposes eval/sample
    route: Optional[RouteResult] = None
    retrieved: int = 0                    # past replies used as draft references
    _seal_stage: Optional[_SealStage] = field(default=None, repr=False)

    @property
    def acceptance(self) -> float:
        return self.draft_accepted / self.draft_scored if self.draft_scored else 0.0

    @property
    def tokens_per_pass(self) -> Optional[float]:
        """Accepted tokens per verification pass (exact, from the eval meter)."""
        if self.eval is not None and self.eval.decode_passes:
            return self.eval.tokens_per_pass
        return None

    @property
    def attention_pairs(self) -> Optional[int]:
        return self.eval.attention_pairs if self.eval is not None else None

    def wait_seal(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """Block until the background seal is written; returns its record (None if sealing
        was off for this turn)."""
        if self.seal_future is None:
            return None
        rec = self.seal_future.result(timeout)
        if self._seal_stage is not None:
            self._seal_stage(self.seal_future)
        return rec

    @property
    def seal(self) -> Optional[Dict]:
        return self.wait_seal()


# --- response cache ----------------------------------------------------------

class ResponseCache:
    """SHA-256-addressed reply cache, stored as append-only JSONL (one line per reply; the last
    line for a key wins), so storing a reply costs one appended line rather than rewriting the
    whole file. A legacy ``responses.json`` next to it is imported once."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: Dict[str, Dict] = {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        self._data[rec.pop("key")] = rec
                    except (ValueError, KeyError, TypeError, AttributeError):
                        continue
        except OSError:
            self._import_legacy(self.path.with_suffix(".json"))

    def _import_legacy(self, legacy: Path) -> None:
        if legacy == self.path:
            return
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self._data = {k: v for k, v in data.items() if isinstance(v, dict) and "text" in v}
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                for k, v in self._data.items():
                    fh.write(json.dumps({"key": k, **v}) + "\n")
        except OSError:
            pass

    def get(self, key: str) -> Optional[str]:
        hit = self._data.get(key)
        return hit["text"] if hit else None

    def put(self, key: str, text: str, model: str) -> None:
        rec = {"text": text, "model": model, "time": int(time.time())}
        self._data[key] = rec
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, **rec}) + "\n")
        except OSError:
            pass

    def __len__(self) -> int:
        return len(self._data)


# --- sealed transcript log ---------------------------------------------------

def _turn_bytes(turn: Dict) -> bytes:
    return json.dumps(turn, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _seal_header(prev: bytes, digest: bytes, ntime: int) -> bytearray:
    header = bytearray(80)
    struct.pack_into("<I", header, 0, 0x20000000)
    header[4:36] = prev
    header[36:68] = digest
    struct.pack_into("<I", header, 68, ntime & 0xFFFFFFFF)
    struct.pack_into("<I", header, 72, _NBITS)
    return header


def _leading_zeros_display(hash_hex: str) -> int:
    """Leading zero bits of a hash given in display (big-endian) hex."""
    return _leading_zeros(bytes.fromhex(hash_hex))


def _leading_zeros(digest: bytes) -> int:
    v = int.from_bytes(digest, "big")
    return 256 - v.bit_length() if v else 256


class TranscriptLog:
    """Append-only chat log where each turn is chained to the previous one by SHA-256 and
    sealed with a mined nonce - the ASIC-native operation - so any later edit is detectable.

    Seals are mined in the background on one thread per log file, in submission order. The
    chain digest is fixed synchronously when a turn is submitted, so the next turn can chain
    to it before the previous seal's nonce is found. One instance exists per file
    (:meth:`for_path`), so several engines writing the same log share one chain.
    """

    _registry: Dict[str, "TranscriptLog"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def for_path(cls, path: Path) -> "TranscriptLog":
        key = str(Path(path).resolve())
        with cls._registry_lock:
            log = cls._registry.get(key)
            if log is None:
                log = cls._registry[key] = cls(Path(path))
            return log

    def __init__(self, path: Path):
        self.path = Path(path)
        self._last: Optional[str] = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seal")

    def _read_last(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                last = None
                for line in fh:
                    if line.strip():
                        last = line
            return json.loads(last)["digest"] if last else _GENESIS
        except (OSError, ValueError, KeyError):
            return _GENESIS

    def last_digest(self) -> str:
        with self._lock:
            if self._last is None:
                self._last = self._read_last()
            return self._last

    def seal_async(self, turn: Dict, acc: HashAccelerator, difficulty_bits: int,
                   timeout: float = 30.0) -> Future:
        with self._lock:
            if self._last is None:
                self._last = self._read_last()
            prev_hex = self._last
            prev = bytes.fromhex(prev_hex)
            digest = acc.hash(prev + _turn_bytes(turn), stage="seal")
            self._last = digest.hex()
            return self._executor.submit(self._mine_and_write, turn, prev_hex, prev, digest,
                                         acc, difficulty_bits, timeout)

    def seal(self, turn: Dict, acc: HashAccelerator, difficulty_bits: int,
             timeout: float = 30.0) -> Dict:
        return self.seal_async(turn, acc, difficulty_bits, timeout).result()

    def flush(self, timeout: Optional[float] = None) -> None:
        """Wait until every submitted seal is written."""
        self._executor.submit(lambda: None).result(timeout)

    def _mine_and_write(self, turn: Dict, prev_hex: str, prev: bytes, digest: bytes,
                        acc: HashAccelerator, difficulty_bits: int, timeout: float) -> Dict:
        t0 = time.perf_counter()
        ntime = int(turn.get("time", 0))
        try:
            share = acc.pool_seal(prev, digest, ntime, difficulty_bits, timeout=timeout,
                                  stage="seal")
        except Exception:
            share = None
        if share is not None:
            # Mined by a real miner through the private pool: the header commits to the digest
            # via the coinbase, so the proof carries both (see ai_asic.stratum.protocol).
            record = {
                "turn": turn, "prev": prev_hex, "digest": digest.hex(), "kind": "stratum",
                "nonce": share.nonce, "pow_hash": share.pow_hash_hex,
                "difficulty": difficulty_bits, "found": share.leading_zeros >= difficulty_bits,
                "coinbase": share.coinbase.hex(), "header": share.header.hex(),
                "device": acc.label, "miner": share.miner,
                "miner_ms": round(share.seconds * 1000, 1),
                "nonce_device": DEVICE_ASIC, "hashes": acc.pool_share_hashes,
            }
        else:
            header = bytes(_seal_header(prev, digest, ntime))
            try:
                out = acc.mine(header, difficulty_bits, stage="seal")
                device, hashes = out.device, out.hashes_tried
            except Exception:
                out = _software_mine(header, difficulty_bits, 1 << 24)
                device, hashes = DEVICE_HOST, out.hashes_tried
            record = {
                "turn": turn, "prev": prev_hex, "digest": digest.hex(),
                "nonce": out.nonce, "pow_hash": out.hash_hex, "difficulty": difficulty_bits,
                "found": out.found, "device": acc.label,
                "nonce_device": device, "hashes": hashes,
            }
        record["seal_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record


def verify_transcripts(path: Path) -> Tuple[bool, int, List[str]]:
    """Re-check every seal in a transcript log with ``hashlib`` alone (no accelerator).
    Returns ``(ok, turns_checked, problems)``."""
    problems: List[str] = []
    prev_hex = _GENESIS
    n = 0
    try:
        lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return True, 0, []
    for i, line in enumerate(lines, start=1):
        n = i
        try:
            rec = json.loads(line)
        except ValueError:
            problems.append(f"turn {i}: not valid JSON")
            break
        if rec.get("prev") != prev_hex:
            problems.append(f"turn {i}: chain broken (prev digest mismatch)")
        prev = bytes.fromhex(rec.get("prev", _GENESIS))
        digest = hashlib.sha256(prev + _turn_bytes(rec["turn"])).digest()
        if digest.hex() != rec.get("digest"):
            problems.append(f"turn {i}: content altered (digest mismatch)")
        if rec.get("kind") == "stratum":
            try:
                err = stratum_protocol.verify_seal(
                    prev, digest, bytes.fromhex(rec["coinbase"]), bytes.fromhex(rec["header"]),
                    int(rec.get("difficulty", 0)))
            except (KeyError, ValueError):
                err = "malformed miner proof"
            if err:
                problems.append(f"turn {i}: miner seal invalid ({err})")
            prev_hex = rec.get("digest", "")
            continue
        header = _seal_header(prev, digest, int(rec["turn"].get("time", 0)))
        struct.pack_into("<I", header, 76, int(rec.get("nonce", 0)) & 0xFFFFFFFF)
        pow_hash = hashlib.sha256(hashlib.sha256(bytes(header)).digest()).digest()
        if _leading_zeros(pow_hash) < int(rec.get("difficulty", 0)):
            problems.append(f"turn {i}: seal nonce does not meet its difficulty")
        prev_hex = rec.get("digest", "")
    return not problems, n, problems


# --- engine ------------------------------------------------------------------

# GGML tensor type ids accepted by llama-cpp-python's type_k / type_v.
_KV_TYPES = {"f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8}


def llama_kwargs(cfg: Dict, llama_cls=None) -> Dict:
    """llama.cpp performance settings from the chat config (see ``models_dir.DEFAULT_CONFIG``).
    Settings left null keep llama-cpp-python's defaults; with ``llama_cls``, settings its
    constructor does not accept (older versions) are dropped."""
    kw: Dict = {"n_ctx": int(cfg["n_ctx"])}
    for key in ("n_threads", "n_threads_batch", "n_batch", "n_ubatch"):
        if cfg.get(key):
            kw[key] = int(cfg[key])
    if cfg.get("flash_attn") is not None:
        kw["flash_attn"] = bool(cfg["flash_attn"])
    kv = str(cfg.get("kv_cache_type") or "f16").lower()
    if kv not in _KV_TYPES:
        raise ValueError(f"kv_cache_type must be one of {sorted(_KV_TYPES)}, not {kv!r}")
    if kv != "f16":
        kw["type_k"] = kw["type_v"] = _KV_TYPES[kv]
        kw.setdefault("flash_attn", True)  # llama.cpp needs it for a quantized V cache
    if llama_cls is not None:
        try:
            import inspect

            params = inspect.signature(llama_cls.__init__).parameters
            if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
                kw = {k: v for k, v in kw.items() if k in params}
        except (TypeError, ValueError):
            pass
    return kw


class ChatEngine:
    def __init__(self, model_path: Optional[str] = None,
                 accelerator: Optional[HashAccelerator] = None,
                 root: Optional[Path] = None, llm=None, **overrides):
        self.root = ensure_layout(root)
        self.cfg = {**load_config(self.root), **overrides}
        resolved = resolve_llm(model_path, self.root)
        self.model_path: Optional[Path] = resolved if resolved else (
            Path(model_path) if model_path else None)
        self.acc = accelerator or HashAccelerator()
        cfg = self.cfg
        self.draft = HashDraftIndex(
            self.acc, max_ngram=int(cfg["draft_max_ngram"]), num_pred=int(cfg["draft_num_pred"]),
            mode=str(cfg["draft_mode"]), adaptive=bool(cfg["draft_adaptive"]),
            max_candidates=int(cfg["draft_max_candidates"]),
            len_by_match=cfg["draft_len_by_match"])
        self.buckets = BucketHasher(self.acc, difficulty_bits=int(cfg["bucket_difficulty"]),
                                    max_nonces=int(cfg["bucket_max_nonces"]))
        self.router = ContextRouter(self.buckets, budget_tokens=int(cfg["route_budget_tokens"]),
                                    keep_recent=int(cfg["route_keep_recent"]),
                                    n_bands=int(cfg["lsh_bands"]),
                                    band_bits=int(cfg["lsh_band_bits"]),
                                    low_water=float(cfg["route_low_water"]),
                                    recall_similarity=float(cfg["route_recall_sim"]))
        if cfg.get("lsh_prefetch_table", True) and (cfg["route_context"]
                                                    or cfg["draft_retrieval"]):
            # The whole LSH bucket table on the ASIC, in the background: replies then never
            # wait on the device for a bucket ID.
            self.buckets.prefetch_table(int(cfg["lsh_bands"]), int(cfg["lsh_band_bits"]))
        self.store = DraftStore(Path(cfg["draft_store_path"]) if cfg.get("draft_store_path")
                                else cache_dir(self.root) / "draft_store.jsonl")
        self._llm = None
        self._draft_wired = False  # the draft index is actually the model's draft source
        self.meter: Optional[LlamaMeter] = None
        self._tok_cache: Dict[str, int] = {}
        if llm is not None:
            self._adopt(llm)
        self._lock = threading.Lock()
        self.cache = ResponseCache(cache_dir(self.root) / "responses.jsonl")
        self.log = TranscriptLog.for_path(cache_dir(self.root) / "transcripts.jsonl")
        self.new_chat()

    @property
    def model_name(self) -> str:
        return self.model_path.name if self.model_path else "(no model)"

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def set_accelerator(self, acc: HashAccelerator) -> None:
        if acc is not self.acc:
            self.acc.close()  # stop its worker and release its hasher-server connection
        self.acc = acc
        # Digests and bucket IDs are device-independent, so the index and cache stay valid.
        self.draft.acc = acc
        self.buckets.acc = acc

    def new_chat(self) -> None:
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg["system_prompt"]}]
        self.draft.reset()
        self.router.reset()

    def flush(self, timeout: Optional[float] = None) -> None:
        """Wait for background seals to be written."""
        self.log.flush(timeout)

    def _adopt(self, llm) -> None:
        self._llm = llm
        self.meter = LlamaMeter(llm)
        # A stand-in model built without a draft model gets the index attached directly
        # (a real Llama receives it at construction; see load()).
        if self.cfg["use_draft"] and getattr(llm, "draft_model", "absent") is None:
            llm.draft_model = self.draft
        self._draft_wired = self.cfg["use_draft"] and getattr(llm, "draft_model", None) is not None

    def load(self) -> None:
        if self._llm is not None:
            return
        if not self.model_path or not self.model_path.is_file():
            raise FileNotFoundError(
                "No GGUF model found. Put a .gguf file in ai_models/llm/ or run "
                "'python -m ai_asic.cli models download qwen2.5-0.5b'.")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install the CPU wheel with:\n"
                "  pip install llama-cpp-python --only-binary=:all: "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            ) from exc
        draft = make_llama_draft_model(self.draft) if self.cfg["use_draft"] else None
        self._adopt(Llama(model_path=str(self.model_path), draft_model=draft, verbose=False,
                          **llama_kwargs(self.cfg, Llama)))

    # -- one turn -------------------------------------------------------------
    def reply(self, user_text: str, on_token: Optional[Callable[[str], None]] = None,
              max_tokens: Optional[int] = None,
              temperature: Optional[float] = None) -> TurnResult:
        with self._lock:
            return self._reply(user_text, on_token, max_tokens, temperature)

    def _reply(self, user_text, on_token, max_tokens, temperature) -> TurnResult:
        max_tokens = int(max_tokens or self.cfg["max_tokens"])
        temperature = float(self.cfg["temperature"] if temperature is None else temperature)
        acc = self.acc
        acc.reset_meters()
        trace = ExecutionTrace(asic_label=acc.label, asic_is_hardware=acc.is_hardware)
        msgs = self.messages + [{"role": "user", "content": user_text}]

        # [host] fingerprint -> response cache
        t0 = time.perf_counter()
        key_src = json.dumps({"model": self.model_name, "messages": msgs,
                              "max_tokens": max_tokens, "temperature": temperature},
                             sort_keys=True).encode("utf-8")
        key = acc.hash(key_src, stage="fingerprint").hex()
        cached = self.cache.get(key) if self.cfg["use_cache"] else None
        trace.add(Stage("fingerprint", DEVICE_HOST, ops=1, seconds=time.perf_counter() - t0,
                        detail=f"{key[:16]}... cache {'HIT' if cached else 'miss'}"))
        result = TurnResult(text="", cached=cached is not None, trace=trace)

        if cached is not None:
            text = cached
            if on_token:
                on_token(text)
        else:
            self.load()
            text = self._generate(msgs, user_text, on_token, max_tokens, temperature, result,
                                  key)

        self.messages = msgs + [{"role": "assistant", "content": text}]
        result.text = text

        # [ASIC] seal, in the background
        difficulty = int(self.cfg.get("seal_difficulty", 0))
        if difficulty > 0:
            turn = {"time": int(time.time()), "model": self.model_name,
                    "user": user_text, "assistant": text}
            fut = self.log.seal_async(turn, acc, difficulty,
                                      timeout=float(self.cfg.get("seal_timeout", 30)))
            result.seal_future = fut
            result._seal_stage = _SealStage(trace, difficulty)
            fut.add_done_callback(result._seal_stage)
            if not self.cfg.get("seal_async", True):
                result.wait_seal()
        # The device may have fallen back mid-turn; report what actually finished the turn.
        trace.asic_label, trace.asic_is_hardware = acc.label, acc.is_hardware
        return result

    def _generate(self, msgs, user_text, on_token, max_tokens, temperature,
                  result: TurnResult, cache_key: str) -> str:
        cfg, trace = self.cfg, result.trace
        use_draft = bool(cfg["use_draft"])
        retrieval = use_draft and self._draft_wired and bool(cfg["draft_retrieval"])

        # [ASIC] bucket IDs + [host] KV-block routing
        route = self.router.route(msgs, self._msg_tokens, enabled=bool(cfg["route_context"]),
                                  want_query_buckets=retrieval)
        result.route = route
        b = route.buckets
        if b.searches or b.cache_hits or b.waited:
            trace.add(Stage("route.buckets", b.device, ops=b.hashes, seconds=b.seconds,
                            detail=f"{b.searches} nonce searches in {b.round_trips} round "
                                   f"trip(s), {b.cache_hits} cached"
                                   + (f", {b.waited} from the table still in flight"
                                      if b.waited else "") + " (LSH bucket IDs)"))
        if cfg["route_context"]:
            detail = (f"{route.reason}; prompt {route.tokens_full} -> {route.tokens_routed} "
                      f"tokens" if route.routed else
                      f"{route.reason}; {route.tokens_full} prompt tokens")
            trace.add(Stage("route", DEVICE_HOST, ops=0, seconds=route.select_seconds,
                            detail=detail))
        sent = self._fit_context(route.messages, max_tokens)

        # [host] retrieval drafts
        d = self.draft
        if retrieval and route.query_buckets is not None:
            t0 = time.perf_counter()
            hits = self.store.lookup(route.query_sig, route.query_buckets,
                                     k=int(cfg["draft_retrieval_k"]),
                                     min_similarity=float(cfg["draft_retrieval_min_sim"]))
            d.set_references([self._llm.tokenize(t.encode("utf-8"), add_bos=False)
                              for _, t in hits])
            result.retrieved = len(hits)
            sims = ", ".join(f"{s:.2f}" for s, _ in hits)
            trace.add(Stage("retrieve", DEVICE_HOST, ops=len(hits),
                            seconds=time.perf_counter() - t0,
                            detail=(f"{len(hits)} past repl{'y' if len(hits) == 1 else 'ies'} as "
                                    f"draft candidates (similarity {sims})" if hits else
                                    f"no similar past reply ({len(self.store)} stored)")))
        elif use_draft:
            d.set_references([])

        # [host] llama.cpp generation, metered
        d.discard_pending()
        before = (d.calls, d.proposed, d.scored, d.accepted, d.candidates, d.ref_drafts)
        draft_s0, draft_ops0 = d.seconds, d.windows
        metered = self.meter is not None and self.meter.available
        if metered:
            self.meter.begin()
        parts: List[str] = []
        t_first: Optional[float] = None
        t0 = time.perf_counter()
        for chunk in self._llm.create_chat_completion(
                messages=sent, stream=True, max_tokens=max_tokens, temperature=temperature):
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                if t_first is None:
                    t_first = time.perf_counter()
                parts.append(delta)
                if on_token:
                    on_token(delta)
        t_end = time.perf_counter()
        d.discard_pending()  # the final draft of a turn is never verified
        elapsed = t_end - t0
        raw = "".join(parts)
        ntok = self._count_tokens(raw, len(parts))
        stats = self.meter.end(fallback_sampled=ntok) if metered else None
        result.eval = stats
        result.tokens = ntok
        result.tokens_per_s = ntok / elapsed if elapsed > 0 else 0.0
        if t_first is not None:
            result.ttft_s = t_first - t0
            if ntok > 1 and t_end > t_first:
                result.decode_tps = (ntok - 1) / (t_end - t_first)
        lookups, proposed, scored, accepted, cands, ref_drafts = (
            a - b for a, b in zip(
                (d.calls, d.proposed, d.scored, d.accepted, d.candidates, d.ref_drafts), before))
        result.draft_lookups, result.draft_proposed = lookups, proposed
        result.draft_scored, result.draft_accepted = scored, accepted
        draft_secs = d.seconds - draft_s0

        detail = (f"{ntok} tokens, {result.decode_tps:.1f} tok/s decode, first token "
                  f"{result.ttft_s * 1000:.0f} ms")
        if stats is not None:
            detail += (f"; {stats.decode_passes} verification passes, "
                       f"{stats.evaluated_tokens} tokens evaluated ({stats.reused_tokens} KV "
                       f"reused), {stats.attention_pairs:,} attention pairs")
        trace.add(Stage("llm", DEVICE_HOST, ops=ntok, seconds=max(0.0, elapsed - draft_secs),
                        detail=detail))
        if use_draft:
            rate = f"{accepted / scored:.0%}" if scored else "n/a"
            if stats is not None and stats.decode_passes:
                per_pass = (f"{stats.tokens_per_pass:.2f} tok/pass, "
                            f"{stats.rejected_tokens} unused draft tokens")
            else:
                per_pass = f"~{ntok / lookups:.2f} tok/pass" if lookups else "no passes"
            trace.add(Stage("draft", DEVICE_HOST, ops=d.windows - draft_ops0,
                            seconds=draft_secs,
                            detail=f"{lookups} lookups, {proposed} proposed from {cands} "
                                   f"candidates, {accepted}/{scored} accepted ({rate}), "
                                   f"{per_pass}"
                                   + (f", {ref_drafts} drafts from retrieved replies"
                                      if ref_drafts else "")))

        text = raw.strip()
        if cfg["use_cache"]:
            self.cache.put(cache_key, text, self.model_name)
        # Index this turn for later: retrieval by the prompt's buckets, and the exchange's
        # buckets searched in the background so routing finds them cached next turn.
        if retrieval and route.query_buckets is not None:
            self.store.add(route.query_sig, route.query_buckets, text)
        if cfg["route_context"]:
            self.router.prefetch(user_text + "\n" + text)
        return text

    def _count_tokens(self, text: str, fallback: int) -> int:
        """Generated-token count. Stream chunks are not tokens (llama-cpp-python holds back
        partial UTF-8 and merges/splits pieces), so re-tokenize the reply. That can differ from
        the sampled tokens by a token or so, but identically across runs of the same reply."""
        if not text:
            return 0
        try:
            return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            return fallback

    def _msg_tokens(self, m: Dict[str, str]) -> int:
        content = m.get("content", "")
        n = self._tok_cache.get(content)
        if n is None:
            n = len(self._llm.tokenize(content.encode("utf-8"), add_bos=False)) + 6
            if len(self._tok_cache) > 4096:
                self._tok_cache.clear()
            self._tok_cache[content] = n
        return n

    def _fit_context(self, msgs: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Drop the oldest exchanges (keeping the system prompt) until the prompt plus the
        reply budget fits the context window."""
        budget = int(self.cfg["n_ctx"]) - max_tokens - 32

        def size(ms):
            return sum(self._msg_tokens(m) for m in ms)

        system, rest = msgs[:1], msgs[1:]
        while len(rest) > 1 and size(system + rest) > budget:
            rest = rest[2:] if len(rest) > 2 else rest[1:]
        return system + rest
