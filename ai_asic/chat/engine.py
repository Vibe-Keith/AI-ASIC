"""Chat engine: a local GGUF LLM on the CPU, with the SHA-256 stages on the hash accelerator.

Per turn:

    [ASIC] fingerprint  SHA-256 content address of (model, conversation, sampling) -> response
                        cache; an exact repeat is answered without running the LLM
    [CPU]  llm          tokenize, forward passes, draft verification, sampling (llama.cpp)
    [ASIC] draft        SHA-256-keyed n-gram index proposing speculative tokens (inside the
                        llm loop; its time is reported separately and excluded from "llm")
    [ASIC] seal         mine a proof-of-work nonce over the turn's chained transcript digest,
                        appended to ai_models/cache/transcripts.jsonl (verify with
                        :func:`verify_transcripts`). On a real miner this goes through the
                        host's private stratum pool (ai_asic.stratum), and fingerprint/draft
                        hashing runs on the host - a mining ASIC only hashes block headers.

Every stage reports the device that actually ran it, in the same ExecutionTrace the
Workload tab uses.
"""
from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ai_asic.chat.accelerator import HashAccelerator
from ai_asic.chat.draft import HashDraftIndex, make_llama_draft_model
from ai_asic.chat.models_dir import cache_dir, ensure_layout, load_config, resolve_llm
from ai_asic.stratum import protocol as stratum_protocol
from ai_asic.workloads.split_model import DEVICE_ASIC, DEVICE_HOST, ExecutionTrace, Stage

_GENESIS = "00" * 32
_NBITS = 0x1D00FFFF


@dataclass
class TurnResult:
    text: str
    cached: bool
    trace: ExecutionTrace
    seal: Optional[Dict] = None
    tokens: int = 0            # generated tokens (the reply re-tokenized, not stream chunks)
    tokens_per_s: float = 0.0  # tokens / whole generation time, prompt processing included
    ttft_s: float = 0.0        # time to first token: prompt processing + first forward pass
    decode_tps: float = 0.0    # tokens after the first / time after the first
    draft_lookups: int = 0     # ~ verification forward passes when drafting
    draft_proposed: int = 0
    draft_scored: int = 0
    draft_accepted: int = 0

    @property
    def acceptance(self) -> float:
        return self.draft_accepted / self.draft_scored if self.draft_scored else 0.0


# --- response cache ----------------------------------------------------------

class ResponseCache:
    def __init__(self, path: Path):
        self.path = path
        try:
            self._data: Dict[str, Dict] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._data = {}

    def get(self, key: str) -> Optional[str]:
        hit = self._data.get(key)
        return hit["text"] if hit else None

    def put(self, key: str, text: str, model: str) -> None:
        self._data[key] = {"text": text, "model": model, "time": int(time.time())}
        try:
            self.path.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
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
    sealed with a mined nonce - the ASIC-native operation - so any later edit is detectable."""

    def __init__(self, path: Path):
        self.path = path

    def last_digest(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                last = None
                for line in fh:
                    if line.strip():
                        last = line
            return json.loads(last)["digest"] if last else _GENESIS
        except (OSError, ValueError, KeyError):
            return _GENESIS

    def seal(self, turn: Dict, acc: HashAccelerator, difficulty_bits: int,
             timeout: float = 30.0) -> Dict:
        prev_hex = self.last_digest()
        prev = bytes.fromhex(prev_hex)
        digest = acc.hash(prev + _turn_bytes(turn), stage="seal")
        ntime = int(turn.get("time", 0))
        share = acc.pool_seal(prev, digest, ntime, difficulty_bits, timeout=timeout,
                              stage="seal")
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
            }
        else:
            header = _seal_header(prev, digest, ntime)
            out = acc.mine(bytes(header), difficulty_bits, stage="seal")
            record = {
                "turn": turn, "prev": prev_hex, "digest": digest.hex(),
                "nonce": out.nonce, "pow_hash": out.hash_hex, "difficulty": difficulty_bits,
                "found": out.found, "device": acc.label,
            }
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
        self.draft = HashDraftIndex(self.acc, max_ngram=int(self.cfg["draft_max_ngram"]),
                                    num_pred=int(self.cfg["draft_num_pred"]))
        self._llm = llm
        self._lock = threading.Lock()
        self.cache = ResponseCache(cache_dir(self.root) / "responses.json")
        self.log = TranscriptLog(cache_dir(self.root) / "transcripts.jsonl")
        self.new_chat()

    @property
    def model_name(self) -> str:
        return self.model_path.name if self.model_path else "(no model)"

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def set_accelerator(self, acc: HashAccelerator) -> None:
        if acc is not self.acc:
            self.acc.close()  # release its persistent hasher-server connection
        self.acc = acc
        self.draft.acc = acc  # digests are device-independent, so the index stays valid

    def new_chat(self) -> None:
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg["system_prompt"]}]
        self.draft.reset()

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
        self._llm = Llama(model_path=str(self.model_path), n_ctx=int(self.cfg["n_ctx"]),
                          draft_model=draft, verbose=False)

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

        # [ASIC] fingerprint -> response cache
        t0 = time.perf_counter()
        key_src = json.dumps({"model": self.model_name, "messages": msgs,
                              "max_tokens": max_tokens, "temperature": temperature},
                             sort_keys=True).encode("utf-8")
        key = acc.hash(key_src, stage="fingerprint").hex()
        cached = self.cache.get(key) if self.cfg["use_cache"] else None
        trace.add(Stage("fingerprint", acc.device, ops=1, seconds=time.perf_counter() - t0,
                        detail=f"{key[:16]}... cache {'HIT' if cached else 'miss'}"))

        ntok, tps, ttft, decode_tps = 0, 0.0, 0.0, 0.0
        draft_stats = (0, 0, 0, 0)
        if cached is not None:
            text = cached
            if on_token:
                on_token(text)
        else:
            self.load()
            msgs = self._fit_context(msgs, max_tokens)
            d = self.draft
            d.discard_pending()
            before = (d.calls, d.proposed, d.scored, d.accepted)
            parts: List[str] = []
            t_first: Optional[float] = None
            t0 = time.perf_counter()
            for chunk in self._llm.create_chat_completion(
                    messages=msgs, stream=True, max_tokens=max_tokens,
                    temperature=temperature):
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
            tps = ntok / elapsed if elapsed > 0 else 0.0
            if t_first is not None:
                ttft = t_first - t0
                if ntok > 1 and t_end > t_first:
                    decode_tps = (ntok - 1) / (t_end - t_first)
            draft_stats = tuple(a - b for a, b in zip(
                (d.calls, d.proposed, d.scored, d.accepted), before))
            lookups, proposed, scored, accepted = draft_stats
            dm = acc.meter("draft")
            trace.add(Stage("llm", DEVICE_HOST, ops=ntok, seconds=max(0.0, elapsed - dm.seconds),
                            detail=f"{ntok} tokens, {decode_tps:.1f} tok/s decode, "
                                   f"first token {ttft * 1000:.0f} ms"))
            if self.cfg["use_draft"]:
                rate = f"{accepted / scored:.0%}" if scored else "n/a"
                per_pass = ntok / lookups if lookups else 0.0
                trace.add(Stage("draft", acc.device, ops=dm.ops, seconds=dm.seconds,
                                detail=f"{lookups} lookups, {proposed} proposed, "
                                       f"{accepted}/{scored} accepted ({rate}), "
                                       f"~{per_pass:.2f} tok/pass"))
            text = raw.strip()
            if self.cfg["use_cache"]:
                self.cache.put(key, text, self.model_name)

        self.messages = msgs + [{"role": "assistant", "content": text}]

        # [ASIC] seal
        seal = None
        difficulty = int(self.cfg.get("seal_difficulty", 0))
        if difficulty > 0:
            t0 = time.perf_counter()
            turn = {"time": int(time.time()), "model": self.model_name,
                    "user": user_text, "assistant": text}
            seal = self.log.seal(turn, acc, difficulty,
                                 timeout=float(self.cfg.get("seal_timeout", 30)))
            sm = acc.meter("seal")
            on_miner = seal.get("kind") == "stratum"
            detail = f"nonce 0x{seal['nonce']:08x} at {difficulty} bits"
            if on_miner:
                detail += (f", {_leading_zeros_display(seal['pow_hash'])} zero bits from "
                           f"{seal['miner']} in {seal['miner_ms']:.0f} ms")
            trace.add(Stage("seal", DEVICE_ASIC if on_miner else acc.device, ops=sm.ops,
                            seconds=time.perf_counter() - t0, detail=detail))
        # The device may have fallen back mid-turn; report what actually finished the turn.
        trace.asic_label, trace.asic_is_hardware = acc.label, acc.is_hardware
        lookups, proposed, scored, accepted = draft_stats
        return TurnResult(text=text, cached=cached is not None, trace=trace, seal=seal,
                          tokens=ntok, tokens_per_s=tps, ttft_s=ttft, decode_tps=decode_tps,
                          draft_lookups=lookups, draft_proposed=proposed,
                          draft_scored=scored, draft_accepted=accepted)

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

    def _fit_context(self, msgs: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Drop the oldest exchanges (keeping the system prompt) until the prompt plus the
        reply budget fits the context window."""
        budget = int(self.cfg["n_ctx"]) - max_tokens - 32
        tok = self._llm.tokenize

        def size(ms):
            return sum(len(tok(m["content"].encode("utf-8"), add_bos=False)) + 6 for m in ms)

        system, rest = msgs[:1], msgs[1:]
        while len(rest) > 1 and size(system + rest) > budget:
            rest = rest[2:] if len(rest) > 2 else rest[1:]
        return system + rest
