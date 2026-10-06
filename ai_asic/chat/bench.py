"""Speculative-draft benchmark: is drafting actually faster for this model on this machine?

Runs the same prompts through the chat engine with drafting off and with one or more draft
shapes (``max_ngram x num_pred``), greedy (temperature 0) so every configuration should produce
the same reply, and reports decode speed, time to first token and draft acceptance side by side.

Drafting is not free in llama-cpp-python: it keeps logits for every position and checks drafts
in Python, so on a small model whose forward pass is already fast it can lose. Measure first.
"""
from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ai_asic.chat.accelerator import HashAccelerator
from ai_asic.chat.engine import ChatEngine

# (name, prompt). The first two invite the reply to reuse context (where prompt lookup shines);
# the last is free-form, where drafts rarely hit and only their overhead shows.
DEFAULT_PROMPTS: List[Tuple[str, str]] = [
    ("echo",
     "Repeat this paragraph exactly, word for word:\n\n"
     "The miner hashes block headers by rolling a 32-bit nonce. Each chip runs many SHA-256 "
     "cores in parallel, and the control board collects any nonce whose double hash falls "
     "below the target. The host only sends work and checks the results."),
    ("code",
     "Rewrite this Python function with the variable `x` renamed to `total`. Output only the "
     "code.\n\n"
     "def summarize(values):\n"
     "    x = 0\n"
     "    for v in values:\n"
     "        if v > 0:\n"
     "            x += v\n"
     "    print('positive sum:', x)\n"
     "    return x\n"),
    ("free", "In three sentences, explain why the sky looks blue."),
]

# A configuration: None = drafting off, else (max_ngram, num_pred).
Config = Optional[Tuple[int, int]]


def parse_configs(spec: str) -> List[Config]:
    """``"off,3x10,4x16"`` -> ``[None, (3, 10), (4, 16)]``."""
    out: List[Config] = []
    for part in (p.strip().lower() for p in spec.split(",")):
        if not part:
            continue
        if part == "off":
            out.append(None)
            continue
        try:
            ng, npred = part.split("x")
            out.append((int(ng), int(npred)))
        except ValueError:
            raise ValueError(f"bad config {part!r}; use 'off' or NGRAMxTOKENS like 3x10")
    if not out:
        raise ValueError("no configurations given")
    return out


def config_label(cfg: Config) -> str:
    return "off" if cfg is None else f"{cfg[0]}x{cfg[1]}"


@dataclass
class BenchRow:
    config: str
    prompt: str
    tokens: int
    ttft_s: float
    decode_tps: float
    tokens_per_s: float
    acceptance: Optional[float]   # None when drafting is off or nothing was scored
    tok_per_pass: Optional[float]
    text_sha: str
    scored: int = 0
    accepted: int = 0


@dataclass
class BenchSummary:
    config: str
    tokens: int
    decode_tps: float             # all decode tokens / all decode time
    ttft_s: float                 # mean
    acceptance: Optional[float]   # pooled over prompts
    speedup: Optional[float]      # decode_tps vs the first configuration
    same_text: int                # prompts whose reply matches the first configuration's
    prompts: int


def run_bench(model_path: Optional[str], configs: Sequence[Config],
              prompts: Sequence[Tuple[str, str]] = DEFAULT_PROMPTS, max_tokens: int = 128,
              runs: int = 1, accelerator: Optional[HashAccelerator] = None,
              make_llm: Optional[Callable[[], Any]] = None,
              on_row: Optional[Callable[[BenchRow], None]] = None,
              **overrides) -> List[BenchRow]:
    """Run every prompt ``runs`` times under each configuration. Loads one model at a time.
    ``make_llm`` substitutes a stand-in model (tests)."""
    acc = accelerator or HashAccelerator()
    rows: List[BenchRow] = []
    for cfg in configs:
        opts = dict(overrides, use_draft=cfg is not None, use_cache=False, seal_difficulty=0)
        if cfg is not None:
            opts.update(draft_max_ngram=cfg[0], draft_num_pred=cfg[1])
        llm = make_llm() if make_llm else None
        engine = ChatEngine(model_path, accelerator=acc, llm=llm, **opts)
        engine.load()
        try:
            for _ in range(runs):
                for name, prompt in prompts:
                    engine.new_chat()
                    r = engine.reply(prompt, max_tokens=max_tokens, temperature=0.0)
                    row = BenchRow(
                        config=config_label(cfg), prompt=name, tokens=r.tokens,
                        ttft_s=r.ttft_s, decode_tps=r.decode_tps, tokens_per_s=r.tokens_per_s,
                        acceptance=r.acceptance if r.draft_scored else None,
                        tok_per_pass=(r.tokens / r.draft_lookups
                                      if cfg is not None and r.draft_lookups else None),
                        text_sha=hashlib.sha256(r.text.encode("utf-8")).hexdigest(),
                        scored=r.draft_scored, accepted=r.draft_accepted)
                    rows.append(row)
                    if on_row:
                        on_row(row)
        finally:
            close = getattr(engine._llm, "close", None)
            if callable(close) and make_llm is None:
                close()
            engine._llm = None
            del engine
            gc.collect()
    return rows


def summarize(rows: Sequence[BenchRow]) -> List[BenchSummary]:
    order: List[str] = []
    for r in rows:
        if r.config not in order:
            order.append(r.config)
    base_text = {}
    for r in rows:
        if r.config == order[0]:
            base_text.setdefault(r.prompt, r.text_sha)
    out: List[BenchSummary] = []
    base_tps: Optional[float] = None
    for cfg in order:
        rs = [r for r in rows if r.config == cfg]
        dec_tokens = sum(max(0, r.tokens - 1) for r in rs if r.decode_tps > 0)
        dec_time = sum((r.tokens - 1) / r.decode_tps for r in rs if r.decode_tps > 0)
        tps = dec_tokens / dec_time if dec_time > 0 else 0.0
        if base_tps is None:
            base_tps = tps
        scored = sum(r.scored for r in rs)
        acceptance = sum(r.accepted for r in rs) / scored if scored else None
        prompts = {r.prompt: r.text_sha for r in rs}
        out.append(BenchSummary(
            config=cfg, tokens=sum(r.tokens for r in rs), decode_tps=tps,
            ttft_s=sum(r.ttft_s for r in rs) / len(rs), acceptance=acceptance,
            speedup=tps / base_tps if base_tps else None,
            same_text=sum(1 for p, sha in prompts.items() if base_text.get(p) == sha),
            prompts=len(prompts)))
    return out


def format_row(r: BenchRow) -> str:
    acc = f"{r.acceptance:>8.0%}" if r.acceptance is not None else f"{'-':>8}"
    tpp = f"{r.tok_per_pass:>8.2f}" if r.tok_per_pass is not None else f"{'-':>8}"
    return (f"{r.config:<8}{r.prompt:<8}{r.tokens:>6}{r.ttft_s * 1000:>9.0f}"
            f"{r.decode_tps:>9.1f}{acc}{tpp}")


ROW_HEADER = (f"{'Config':<8}{'Prompt':<8}{'Tokens':>6}{'TTFT ms':>9}{'tok/s':>9}"
              f"{'Accept':>8}{'Tok/pass':>8}")


def format_summary(summaries: Sequence[BenchSummary]) -> List[str]:
    lines = [f"{'Config':<8}{'Tokens':>7}{'Decode tok/s':>14}{'Speedup':>9}{'TTFT ms':>9}"
             f"{'Accept':>8}  Same reply as {summaries[0].config if summaries else '-'}"]
    for s in summaries:
        acc = f"{s.acceptance:>8.0%}" if s.acceptance is not None else f"{'-':>8}"
        sp = f"{s.speedup:>8.2f}x" if s.speedup is not None else f"{'-':>9}"
        lines.append(f"{s.config:<8}{s.tokens:>7}{s.decode_tps:>14.1f}{sp}"
                     f"{s.ttft_s * 1000:>9.0f}{acc}  {s.same_text}/{s.prompts}")
    return lines
