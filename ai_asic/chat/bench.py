"""Benchmarks for the decoding pipeline: does it cut CPU work for this model on this machine?

Two suites, both greedy (temperature 0) so every configuration should produce the same reply:

**Draft suite** (:func:`run_bench`) - the same prompts with drafting off and with one or more
draft configurations. Reports the two numbers that predict a real tokens/sec gain on a CPU:

  * accepted tokens per verification pass (``tok/pass``): each pass is one batched forward
    pass of the full model, so more tokens per pass means fewer passes per reply;
  * attention pairs (query-key dot products per layer and head, prefill + decode): the
    attention work the CPU did, including draft tokens it evaluated and then rejected;

plus decode speed, time to first token, unused draft tokens and whether the reply matched.

**Routing suite** (:func:`run_route_bench`) - a live multi-turn chat (eight exchanges on
different topics, then questions that refer back to early ones) with the full history and with
KV-block routing at one or more budgets, on one model instance so llama.cpp's KV prefix reuse
behaves exactly as in the app. Reports, summed over all turns, prompt tokens prefilled and
attention pairs, and whether the final replies matched the full-context ones (a strict quality
check: a changed reply is a changed reply).

Drafting is not free in llama-cpp-python: it keeps logits for every position and checks drafts
in Python, so on a small model whose forward pass is already fast it can lose. Measure first.
"""
from __future__ import annotations

import gc
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ai_asic.chat.accelerator import HashAccelerator
from ai_asic.chat.engine import ChatEngine

# (name, prompt). The first two invite the reply to reuse context (where prompt lookup shines);
# the last is free-form, where drafts rarely hit and only their overhead shows.
_PARA = ("The miner hashes block headers by rolling a 32-bit nonce. Each chip runs many SHA-256 "
         "cores in parallel, and the control board collects any nonce whose double hash falls "
         "below the target. The host only sends work and checks the results.")
_CODE = ("def summarize(values):\n"
         "    x = 0\n"
         "    for v in values:\n"
         "        if v > 0:\n"
         "            x += v\n"
         "    print('positive sum:', x)\n"
         "    return x\n")
DEFAULT_PROMPTS: List[Tuple[str, str]] = [
    ("echo", "Repeat this paragraph exactly, word for word:\n\n" + _PARA),
    ("code",
     "Rewrite this Python function with the variable `x` renamed to `total`. Output only the "
     "code.\n\n" + _CODE),
    ("free", "In three sentences, explain why the sky looks blue."),
]

# Scripted replies for --sim (the weightless stand-in model): what a model would plausibly say.
SIM_REPLIES: Dict[str, str] = {
    DEFAULT_PROMPTS[0][1]: _PARA,
    DEFAULT_PROMPTS[1][1]: (_CODE.replace(" x ", " total ").replace("x +=", "total +=")
                            .replace(" x)", " total)").replace("return x", "return total")),
    DEFAULT_PROMPTS[2][1]: (
        "Sunlight contains all colours, and when it enters the atmosphere the short blue "
        "wavelengths are scattered by air molecules far more than red ones. This Rayleigh "
        "scattering sends blue light toward our eyes from every direction of the sky. At "
        "sunset the light crosses more air, so the blue is scattered away and the sky turns "
        "red."),
}

# A configuration: None = drafting off, else (max_ngram, num_pred) or (max_ngram, num_pred,
# flags) where flags may contain "l" (legacy single-occurrence lookup, no length caps) and "r"
# (retrieval drafts from replies earlier in the same benchmark run).
Config = Optional[Union[Tuple[int, int], Tuple[int, int, str]]]
_FLAGS = set("lr")


def parse_configs(spec: str) -> List[Config]:
    """``"off,3x10l,4x16,4x16r"`` -> ``[None, (3, 10, 'l'), (4, 16), (4, 16, 'r')]``."""
    out: List[Config] = []
    for part in (p.strip().lower() for p in spec.split(",")):
        if not part:
            continue
        if part == "off":
            out.append(None)
            continue
        try:
            ng, rest = part.split("x")
            digits = rest.rstrip("lr")
            flags = rest[len(digits):]
            if not digits or set(flags) - _FLAGS:
                raise ValueError
            cfg: Config = (int(ng), int(digits), "".join(sorted(set(flags)))) if flags else (
                int(ng), int(digits))
            out.append(cfg)
        except ValueError:
            raise ValueError(f"bad config {part!r}; use 'off' or NGRAMxTOKENS[l][r] like 4x16")
    if not out:
        raise ValueError("no configurations given")
    return out


def config_label(cfg: Config) -> str:
    if cfg is None:
        return "off"
    return f"{cfg[0]}x{cfg[1]}" + (cfg[2] if len(cfg) > 2 else "")


def _config_overrides(cfg: Config, store_dir: Path) -> Dict[str, Any]:
    if cfg is None:
        return {"use_draft": False, "draft_retrieval": False}
    flags = cfg[2] if len(cfg) > 2 else ""
    opts: Dict[str, Any] = {"use_draft": True, "draft_max_ngram": cfg[0],
                            "draft_num_pred": cfg[1], "draft_retrieval": "r" in flags,
                            "draft_store_path": str(store_dir / f"store-{config_label(cfg)}.jsonl")}
    if "l" in flags:
        opts.update(draft_mode="lookup", draft_len_by_match=[])
    return opts


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
    passes: Optional[int] = None          # verification passes (exact, needs the eval meter)
    sampled: Optional[int] = None
    rejected: Optional[int] = None        # draft tokens evaluated but not kept (unused)
    attention: Optional[int] = None       # attention pairs, prefill + decode
    evaluated: Optional[int] = None       # tokens the CPU evaluated
    prefill: Optional[int] = None         # prompt tokens evaluated


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
    tok_per_pass: Optional[float] = None   # pooled: decode tokens / verification passes
    passes: Optional[int] = None
    rejected: Optional[int] = None
    attention: Optional[int] = None
    attention_ratio: Optional[float] = None  # vs the first configuration


def _row(cfg_label: str, prompt: str, r, drafting: bool) -> BenchRow:
    ev = r.eval
    exact = ev is not None and ev.decode_passes > 0
    return BenchRow(
        config=cfg_label, prompt=prompt, tokens=r.tokens, ttft_s=r.ttft_s,
        decode_tps=r.decode_tps, tokens_per_s=r.tokens_per_s,
        acceptance=r.acceptance if r.draft_scored else None,
        tok_per_pass=(ev.tokens_per_pass if exact else
                      (r.tokens / r.draft_lookups if drafting and r.draft_lookups else None)),
        text_sha=hashlib.sha256(r.text.encode("utf-8")).hexdigest(),
        scored=r.draft_scored, accepted=r.draft_accepted,
        passes=ev.decode_passes if ev else None, sampled=ev.sampled if ev else None,
        rejected=ev.rejected_tokens if ev else None,
        attention=ev.attention_pairs if ev else None,
        evaluated=ev.evaluated_tokens if ev else None,
        prefill=ev.prefill_tokens if ev else None)


def _close(engine: ChatEngine, stand_in: bool) -> None:
    close = getattr(engine._llm, "close", None)
    if callable(close) and not stand_in:
        close()
    engine._llm = None


def run_bench(model_path: Optional[str], configs: Sequence[Config],
              prompts: Sequence[Tuple[str, str]] = DEFAULT_PROMPTS, max_tokens: int = 128,
              runs: int = 1, accelerator: Optional[HashAccelerator] = None,
              make_llm: Optional[Callable[[], Any]] = None,
              on_row: Optional[Callable[[BenchRow], None]] = None,
              **overrides) -> List[BenchRow]:
    """Run every prompt ``runs`` times under each configuration. Loads one model at a time.
    ``make_llm`` substitutes a stand-in model (tests, ``--sim``)."""
    acc = accelerator or HashAccelerator()
    rows: List[BenchRow] = []
    with tempfile.TemporaryDirectory(prefix="ai-asic-bench-") as tmp:
        for cfg in configs:
            opts = dict(overrides, use_cache=False, seal_difficulty=0, route_context=False,
                        **_config_overrides(cfg, Path(tmp)))
            llm = make_llm() if make_llm else None
            engine = ChatEngine(model_path, accelerator=acc, llm=llm, **opts)
            engine.load()
            try:
                for _ in range(runs):
                    for name, prompt in prompts:
                        engine.new_chat()
                        r = engine.reply(prompt, max_tokens=max_tokens, temperature=0.0)
                        row = _row(config_label(cfg), name, r, cfg is not None)
                        rows.append(row)
                        if on_row:
                            on_row(row)
            finally:
                _close(engine, make_llm is not None)
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
    base_attn: Optional[int] = None
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
        metered = all(r.passes is not None for r in rs)
        passes = sum(r.passes for r in rs) if metered else None
        attention = sum(r.attention for r in rs) if metered else None
        if base_attn is None and attention is not None:
            base_attn = attention
        tpp = (sum(max(0, r.sampled - 1) for r in rs) / passes if metered and passes else None)
        out.append(BenchSummary(
            config=cfg, tokens=sum(r.tokens for r in rs), decode_tps=tps,
            ttft_s=sum(r.ttft_s for r in rs) / len(rs), acceptance=acceptance,
            speedup=tps / base_tps if base_tps else None,
            same_text=sum(1 for p, sha in prompts.items() if base_text.get(p) == sha),
            prompts=len(prompts), tok_per_pass=tpp, passes=passes,
            rejected=sum(r.rejected for r in rs) if metered else None,
            attention=attention,
            attention_ratio=(attention / base_attn if attention is not None and base_attn
                             else None)))
    return out


def _fmt(v, spec: str, width: int) -> str:
    return f"{v:{width}{spec}}" if v is not None else f"{'-':>{width}}"


def format_row(r: BenchRow) -> str:
    return (f"{r.config:<9}{r.prompt:<8}{r.tokens:>6}{r.ttft_s * 1000:>9.0f}"
            f"{r.decode_tps:>9.1f}{_fmt(r.acceptance, '.0%', 8)}{_fmt(r.tok_per_pass, '.2f', 9)}"
            f"{_fmt(r.passes, 'd', 8)}{_fmt(r.rejected, 'd', 9)}{_fmt(r.attention, ',d', 13)}")


ROW_HEADER = (f"{'Config':<9}{'Prompt':<8}{'Tokens':>6}{'TTFT ms':>9}{'tok/s':>9}"
              f"{'Accept':>8}{'Tok/pass':>9}{'Passes':>8}{'Unused':>9}{'Attn pairs':>13}")


def format_summary(summaries: Sequence[BenchSummary]) -> List[str]:
    base = summaries[0].config if summaries else "-"
    lines = [f"{'Config':<9}{'Tokens':>7}{'Decode tok/s':>13}{'Speedup':>9}{'TTFT ms':>9}"
             f"{'Accept':>8}{'Tok/pass':>9}{'Passes':>8}{'Unused':>9}{'Attn vs ' + base:>13}"
             f"  Same reply as {base}"]
    for s in summaries:
        sp = f"{s.speedup:>8.2f}x" if s.speedup is not None else f"{'-':>9}"
        ratio = f"{s.attention_ratio:>12.2f}x" if s.attention_ratio is not None else f"{'-':>13}"
        lines.append(f"{s.config:<9}{s.tokens:>7}{s.decode_tps:>13.1f}{sp}"
                     f"{s.ttft_s * 1000:>9.0f}{_fmt(s.acceptance, '.0%', 8)}"
                     f"{_fmt(s.tok_per_pass, '.2f', 9)}{_fmt(s.passes, 'd', 8)}"
                     f"{_fmt(s.rejected, 'd', 9)}{ratio}  {s.same_text}/{s.prompts}")
    return lines


# --- routing suite -----------------------------------------------------------------

# A long conversation on distinct topics; the questions refer back to early parts of it.
ROUTE_HISTORY: List[Tuple[str, str]] = [
    ("What does a BM1387 chip receive from the host?",
     "The BM1387 receives pre-hashed work: a 32-byte SHA-256 midstate of the first 64 header "
     "bytes plus a 12-byte tail with the merkle-root end, ntime and nbits. It rolls the 4-byte "
     "nonce itself and reports any nonce whose double hash meets the target."),
    ("How should I align a two-mirror beam steering setup?",
     "Use the first mirror to place the beam on a target near the second mirror, then use the "
     "second mirror to set the angle on a far target. Iterate between the near and far targets; "
     "each pass converges because position and angle are decoupled at those two planes."),
    ("What causes stray light in a lens system?",
     "Stray light comes from ghost reflections between surfaces, scatter from surface roughness "
     "and contamination, and light hitting mounts or baffles. Anti-reflection coatings, black "
     "baffles with sharp vanes, and field stops at intermediate images all reduce it."),
    ("How does speculative decoding speed up an LLM?",
     "A cheap drafter proposes several tokens and the large model verifies them in one batched "
     "forward pass, keeping the longest correct prefix. When drafts are good, each expensive "
     "pass yields several tokens instead of one, so fewer passes are needed per reply."),
    ("What is the difference between TCP and UDP?",
     "TCP is connection-oriented and delivers a reliable ordered byte stream with "
     "retransmission and congestion control. UDP sends independent datagrams with no delivery "
     "guarantee, which keeps latency low for games, voice and DNS."),
    ("Give me a quick bread recipe.",
     "Mix 500 g flour, 350 g warm water, 10 g salt and 7 g instant yeast. Knead ten minutes, "
     "let it rise an hour, shape it, rise another 45 minutes, then bake at 230 C for about 35 "
     "minutes until it sounds hollow."),
    ("How do I measure laser beam divergence?",
     "Measure the beam diameter at two distances well beyond the Rayleigh range, for example "
     "with a camera profiler using the 1/e^2 width. Divergence is the change in diameter over "
     "the separation, or use a focusing lens and measure the spot size at its focal plane."),
    ("Why is the sky blue?",
     "Air molecules scatter short wavelengths much more strongly than long ones (Rayleigh "
     "scattering goes as one over wavelength to the fourth power), so scattered skylight is "
     "dominated by blue."),
]
ROUTE_QUESTIONS: List[Tuple[str, str, str]] = [
    # (name, question, scripted --sim reply)
    ("midstate", "Remind me what the BM1387 midstate and 12-byte tail contain.",
     "The midstate is the SHA-256 state after the first 64 header bytes, and the 12-byte tail "
     "holds the merkle-root end, ntime and nbits."),
    ("mirrors", "For the two-mirror beam steering alignment, which mirror sets the angle?",
     "The second mirror sets the angle on the far target; the first mirror places the beam on "
     "the near target."),
    ("baffles", "Which stray light fixes did you suggest for the lens system?",
     "Anti-reflection coatings, black baffles with sharp vanes, and field stops at "
     "intermediate images."),
]


@dataclass
class RouteRow:
    config: str
    turn: int
    question: str
    final: bool                      # one of the questions that refer back (vs history turns)
    prompt_full: int                 # prompt tokens with the whole history
    prompt_sent: int                 # prompt tokens actually sent
    kept: str
    compacted: bool
    prefill: Optional[int]           # prompt tokens evaluated this turn
    reused: Optional[int]            # KV prefix reused from the previous turn
    attention: Optional[int]         # attention pairs, prefill + decode
    decode_attention: Optional[int]
    tokens: int
    ttft_s: float
    decode_tps: float
    text_sha: str


@dataclass
class RouteSummary:
    config: str
    turns: int
    compactions: int
    prefill: Optional[int]
    attention: Optional[int]
    decode_attention: Optional[int]
    attention_ratio: Optional[float]         # vs the first configuration
    decode_attention_ratio: Optional[float]
    ttft_s: float
    same_final: int                          # final replies identical to the first config's
    finals: int


def run_route_bench(model_path: Optional[str], budgets: Sequence[Optional[int]],
                    history: Sequence[Tuple[str, str]] = ROUTE_HISTORY,
                    questions: Sequence[Tuple[str, str, str]] = ROUTE_QUESTIONS,
                    max_tokens: int = 96, accelerator: Optional[HashAccelerator] = None,
                    make_llm: Optional[Callable[[], Any]] = None,
                    on_row: Optional[Callable[[RouteRow], None]] = None,
                    **overrides) -> List[RouteRow]:
    """A live chat per configuration: every history question, then every final question, as
    real turns on one model instance (so llama.cpp's KV prefix reuse works exactly as in the
    app). Routing off (``None``) or on at each token budget. With a real model the history
    replies are generated, so later turns may differ between configurations; with ``--sim``
    the replies are scripted and identical."""
    acc = accelerator or HashAccelerator()
    rows: List[RouteRow] = []
    turns = [(f"h{i + 1}", q, False) for i, (q, _) in enumerate(history)]
    turns += [(name, q, True) for name, q, _ in questions]
    for budget in budgets:
        label = "full" if budget is None else f"route{budget}"
        opts = dict(overrides, use_cache=False, seal_difficulty=0, use_draft=False,
                    draft_retrieval=False, route_context=budget is not None,
                    route_budget_tokens=budget if budget is not None else 10 ** 9)
        engine = ChatEngine(model_path, accelerator=acc, llm=make_llm() if make_llm else None,
                            **opts)
        engine.load()
        _cold(engine._llm)
        try:
            for n, (name, question, final) in enumerate(turns, start=1):
                r = engine.reply(question, max_tokens=max_tokens, temperature=0.0)
                rt, ev = r.route, r.eval
                row = RouteRow(
                    config=label, turn=n, question=name, final=final,
                    prompt_full=rt.tokens_full if rt else 0,
                    prompt_sent=rt.tokens_routed if rt else 0,
                    kept=f"{rt.blocks_kept}/{rt.blocks_total}" if rt else "-",
                    compacted=bool(rt and rt.compacted),
                    prefill=ev.prefill_tokens if ev else None,
                    reused=ev.reused_tokens if ev else None,
                    attention=ev.attention_pairs if ev else None,
                    decode_attention=ev.decode_attention if ev else None,
                    tokens=r.tokens, ttft_s=r.ttft_s, decode_tps=r.decode_tps,
                    text_sha=hashlib.sha256(r.text.encode("utf-8")).hexdigest())
                rows.append(row)
                if on_row:
                    on_row(row)
        finally:
            _close(engine, make_llm is not None)
            del engine
            gc.collect()
    return rows


def _cold(llm) -> None:
    reset = getattr(llm, "reset", None)
    if callable(reset):
        reset()
    elif hasattr(llm, "n_tokens"):
        llm.n_tokens = 0


def summarize_route(rows: Sequence[RouteRow]) -> List[RouteSummary]:
    order: List[str] = []
    for r in rows:
        if r.config not in order:
            order.append(r.config)
    base_final = {r.question: r.text_sha for r in rows if r.config == order[0] and r.final}
    out: List[RouteSummary] = []
    base_attn = base_dec = None
    for cfg in order:
        rs = [r for r in rows if r.config == cfg]
        metered = all(r.attention is not None for r in rs)
        attn = sum(r.attention for r in rs) if metered else None
        dec = sum(r.decode_attention for r in rs) if metered else None
        if base_attn is None:
            base_attn, base_dec = attn, dec
        finals = [r for r in rs if r.final]
        out.append(RouteSummary(
            config=cfg, turns=len(rs), compactions=sum(r.compacted for r in rs),
            prefill=sum(r.prefill for r in rs) if metered else None,
            attention=attn, decode_attention=dec,
            attention_ratio=attn / base_attn if attn is not None and base_attn else None,
            decode_attention_ratio=dec / base_dec if dec is not None and base_dec else None,
            ttft_s=sum(r.ttft_s for r in rs) / len(rs),
            same_final=sum(1 for r in finals if base_final.get(r.question) == r.text_sha),
            finals=len(finals)))
    return out


ROUTE_HEADER = (f"{'Config':<10}{'Turn':<6}{'Prompt':>7}{'Sent':>6}{'Kept':>6}{'Compact':>8}"
                f"{'Reused':>8}{'Prefill':>8}{'Attn pairs':>12}{'Decode attn':>12}"
                f"{'TTFT ms':>9}{'tok/s':>7}")


def format_route_row(r: RouteRow) -> str:
    return (f"{r.config:<10}{r.question:<6}{r.prompt_full:>7}{r.prompt_sent:>6}{r.kept:>6}"
            f"{'yes' if r.compacted else '':>8}{_fmt(r.reused, 'd', 8)}{_fmt(r.prefill, 'd', 8)}"
            f"{_fmt(r.attention, ',d', 12)}{_fmt(r.decode_attention, ',d', 12)}"
            f"{r.ttft_s * 1000:>9.0f}{r.decode_tps:>7.1f}")


def format_route_summary(summaries: Sequence[RouteSummary]) -> List[str]:
    base = summaries[0].config if summaries else "-"
    lines = [f"{'Config':<10}{'Turns':>6}{'Compactions':>12}{'Prefill':>9}{'Attn pairs':>13}"
             f"{'vs ' + base:>9}{'Decode attn':>13}{'vs ' + base:>9}{'TTFT ms':>9}"
             f"  Same final replies as {base}"]

    def ratio(v):
        return f"{v:>8.2f}x" if v is not None else f"{'-':>9}"

    for s in summaries:
        lines.append(f"{s.config:<10}{s.turns:>6}{s.compactions:>12}{_fmt(s.prefill, 'd', 9)}"
                     f"{_fmt(s.attention, ',d', 13)}{ratio(s.attention_ratio)}"
                     f"{_fmt(s.decode_attention, ',d', 13)}{ratio(s.decode_attention_ratio)}"
                     f"{s.ttft_s * 1000:>9.0f}  {s.same_final}/{s.finals}")
    return lines


def sim_llm_factory(replies: Optional[Dict[str, str]] = None) -> Callable[[], Any]:
    """A factory for the weightless stand-in model with the benchmark's scripted replies."""
    from ai_asic.chat.simllm import ScriptedLlama

    table = dict(SIM_REPLIES)
    table.update(dict(ROUTE_HISTORY))
    table.update({q: a for _, q, a in ROUTE_QUESTIONS})
    if replies:
        table.update(replies)
    return lambda: ScriptedLlama(table)
