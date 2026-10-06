"""Exact CPU-side work counters for llama-cpp-python generation.

:class:`LlamaMeter` wraps a model's ``eval`` and ``sample`` methods (llama-cpp-python's
``generate`` calls both through ``self``), so it sees every token the CPU actually evaluates:

  * the first ``eval`` of a generation is the prompt prefill - only the part after the KV prefix
    reused from the previous call;
  * every later ``eval`` is one speculative verification pass: the last sampled token plus the
    draft being checked;
  * each ``sample`` is one generated token.

Attention work is counted as query-key pairs per layer and head: a batch of ``k`` tokens
evaluated on top of ``n`` cached positions attends ``k*n + k*(k+1)/2`` pairs (causal). This is
the quantity KV routing shrinks and rejected drafts inflate. (llama.cpp pads the KV length it
computes over; this counts the logical pairs.)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalStats:
    prefill_tokens: int = 0            # prompt tokens evaluated (after KV prefix reuse)
    reused_tokens: int = 0             # KV prefix kept from the previous call
    decode_passes: int = 0             # verification passes (evals after the prefill)
    decode_tokens: int = 0             # tokens those passes evaluated (1 + draft each)
    sampled: int = 0                   # tokens generated
    prefill_attention: int = 0         # query-key pairs, per layer and head
    decode_attention: int = 0
    peak_context: int = 0              # largest KV length reached

    @property
    def attention_pairs(self) -> int:
        return self.prefill_attention + self.decode_attention

    @property
    def evaluated_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens

    @property
    def tokens_per_pass(self) -> float:
        """Generated tokens per verification pass after the first token (the mean accepted
        length: 1.0 without drafting, 1 + accepted drafts per pass with it)."""
        return (self.sampled - 1) / self.decode_passes if self.decode_passes else 0.0

    @property
    def rejected_tokens(self) -> int:
        """Draft tokens evaluated in verification passes but not kept: rejected, or past the
        end of the reply (the last draft can run beyond the final token)."""
        return max(0, self.decode_tokens - max(0, self.sampled - 1))

    def add(self, other: "EvalStats") -> "EvalStats":
        return EvalStats(*(getattr(self, f) + getattr(other, f) if f != "peak_context" else
                           max(self.peak_context, other.peak_context)
                           for f in self.__dataclass_fields__))


class LlamaMeter:
    """Counts evaluated tokens, verification passes and attention pairs on a llama-like model.
    ``available`` is False when the model exposes no ``eval``/``n_tokens`` (e.g. test doubles
    that only stream text)."""

    def __init__(self, llm):
        self.llm = llm
        self.stats = EvalStats()
        self._first = True
        self.available = False
        self.counts_samples = False
        orig_eval = getattr(llm, "eval", None)
        if callable(orig_eval) and hasattr(llm, "n_tokens"):
            def eval_wrapper(tokens, *a, _orig=orig_eval, **kw):
                self._on_eval(len(tokens))
                return _orig(tokens, *a, **kw)

            llm.eval = eval_wrapper
            self.available = True
        orig_sample = getattr(llm, "sample", None)
        if self.available and callable(orig_sample):
            def sample_wrapper(*a, _orig=orig_sample, **kw):
                self.stats.sampled += 1
                return _orig(*a, **kw)

            llm.sample = sample_wrapper
            self.counts_samples = True

    def begin(self) -> None:
        self.stats = EvalStats()
        self._first = True

    def _on_eval(self, k: int) -> None:
        n0 = int(getattr(self.llm, "n_tokens", 0) or 0)
        pairs = k * n0 + k * (k + 1) // 2
        s = self.stats
        s.peak_context = max(s.peak_context, n0 + k)
        if self._first:
            self._first = False
            s.reused_tokens = n0
            s.prefill_tokens += k
            s.prefill_attention += pairs
        else:
            s.decode_passes += 1
            s.decode_tokens += k
            s.decode_attention += pairs

    def end(self, fallback_sampled: int = 0) -> EvalStats:
        if not self.counts_samples or self.stats.sampled == 0:
            self.stats.sampled = fallback_sampled
        return self.stats
