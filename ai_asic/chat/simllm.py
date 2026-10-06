"""A weightless stand-in for ``llama_cpp.Llama`` that runs llama-cpp-python's decode loop.

:class:`ScriptedLlama` tokenizes on whitespace and "predicts" a scripted reply for each prompt,
deterministically (like greedy sampling). Everything around that prediction mirrors
llama-cpp-python 0.3's ``Llama.generate``:

  * KV prefix reuse against the previous call's tokens (only the new suffix is evaluated),
  * ``eval`` of the prompt, then one ``eval`` per step of the last token plus the draft,
  * ``sample`` at each position up to ``n_tokens``, accepting draft tokens while they match
    and rewinding the KV state at the first mismatch,
  * the draft model called with the full token sequence each step.

So draft acceptance, verification passes and attention pairs measured on it follow the same
mechanics as on a real model. What it cannot tell you is real speed or real acceptance on real
text - the reply is scripted. Use it to test and to sanity-check the pipeline; benchmark on a
GGUF model for numbers that matter.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Sequence, Union

Replies = Union[Dict[str, str], Callable[[List[Dict[str, str]]], str]]


class ScriptedLlama:
    def __init__(self, replies: Optional[Replies] = None, default_reply: str = "OK.",
                 n_ctx: int = 8192):
        self.replies = replies or {}
        self.default_reply = default_reply
        self._n_ctx = n_ctx
        self._vocab: Dict[str, int] = {}
        self._words: List[str] = []
        self.eos = self._id("</s>")
        self.input_ids: List[int] = []
        self.n_tokens = 0
        self.draft_model = None
        self.calls = 0
        self._target: List[int] = []
        self._prompt_len = 0

    # -- tokenizer ----------------------------------------------------------
    def _id(self, word: str) -> int:
        i = self._vocab.get(word)
        if i is None:
            i = self._vocab[word] = len(self._words)
            self._words.append(word)
        return i

    def tokenize(self, data: bytes, add_bos: bool = False, special: bool = False) -> List[int]:
        return [self._id(w) for w in data.decode("utf-8", errors="replace").split()]

    def detokenize(self, ids: Sequence[int]) -> bytes:
        return " ".join(self._words[i] for i in ids).encode("utf-8")

    def n_ctx(self) -> int:
        return self._n_ctx

    def _format(self, messages: Sequence[Dict[str, str]]) -> List[int]:
        toks: List[int] = []
        for m in messages:
            toks += self.tokenize(f"<|{m['role']}|> {m['content']} <|end|>".encode("utf-8"))
        return toks + self.tokenize(b"<|assistant|>")

    def _reply_for(self, messages: List[Dict[str, str]]) -> str:
        if callable(self.replies):
            return self.replies(messages)
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return self.replies.get(last, self.default_reply)

    # -- llama-cpp-python surface --------------------------------------------
    def create_chat_completion(self, messages, stream: bool = True, max_tokens: int = 128,
                               temperature: float = 0.0, **_kw) -> Iterator[Dict]:
        self.calls += 1
        msgs = list(messages)
        prompt = self._format(msgs)
        self._prompt_len = len(prompt)
        self._target = self.tokenize(self._reply_for(msgs).encode("utf-8")) + [self.eos]
        produced = 0
        for tok in self.generate(prompt):
            if tok == self.eos:
                break
            word = self._words[tok]
            yield {"choices": [{"delta": {"content": word if produced == 0 else " " + word}}]}
            produced += 1
            if produced >= max_tokens:
                break

    def eval(self, tokens: Sequence[int]) -> None:
        n_past = self.n_tokens
        self.input_ids[n_past:] = list(tokens)
        self.n_tokens = n_past + len(tokens)

    def sample(self, idx: int = 0, **_kw) -> int:
        # The logits at position idx predict position idx + 1.
        pos = idx + 1 - self._prompt_len
        return self._target[pos] if 0 <= pos < len(self._target) else self.eos

    def generate(self, tokens: Sequence[int]) -> Iterator[int]:
        tokens = list(tokens)
        if self.n_tokens > 0:
            longest = 0
            for a, b in zip(self.input_ids[:self.n_tokens], tokens[:-1]):
                if a != b:
                    break
                longest += 1
            if longest > 0:
                tokens = tokens[longest:]
                self.n_tokens = longest
            else:
                self.n_tokens = 0
        sample_idx = self.n_tokens + len(tokens) - 1
        while True:
            self.eval(tokens)
            while sample_idx < self.n_tokens:
                token = self.sample(idx=sample_idx)
                sample_idx += 1
                yield token
                tokens.clear()
                tokens.append(token)
                if sample_idx < self.n_tokens and token != self.input_ids[sample_idx]:
                    self.n_tokens = sample_idx  # rewind past the rejected draft
                    break
            if self.draft_model is not None:
                self.input_ids[self.n_tokens:] = tokens
                draft = self.draft_model(self.input_ids[:self.n_tokens + len(tokens)])
                room = self._n_ctx - self.n_tokens - len(tokens)
                tokens.extend(int(t) for t in list(draft)[:max(0, room)])
