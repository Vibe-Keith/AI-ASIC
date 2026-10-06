"""Chatbot: a local LLM on the host CPU, with native BM1387 nonce search on the ASIC.

The language model's neural maths (matrix multiplies, attention, sampling) runs on the CPU via
llama.cpp - a SHA-256 mining ASIC cannot do it. The pipeline is built to make the CPU do less
work per reply:

  * speculative drafts (``draft.py``): consensus over every earlier n-gram occurrence, length
    capped by match strength, plus past replies to similar prompts (``retrieval.py``);
  * KV-block routing (``routing.py``): long chats attend over the relevant and recent history
    only, with sticky selection so llama.cpp keeps reusing its KV prefix;
  * exact metering (``metering.py``): verification passes, accepted tokens per pass and
    attention pairs, from the model's own eval/sample calls.

The ASIC does the one thing a BM1387 can: nonce search. It computes the LSH bucket IDs that
routing and retrieval key on (``lsh.py``) and the proof-of-work seal on each turn's transcript,
asynchronously and in batches (``accelerator.py``). Only that work is counted as ASIC work.

``llama-cpp-python`` is an optional dependency; everything except :class:`ChatEngine`'s model
loading works without it (``simllm.py`` is a weightless stand-in for tests and ``bench --sim``).
See ``docs/ASIC_DECODING.md``.
"""
