"""Chatbot: a local LLM on the host CPU with SHA-256 stages offloaded to the ASIC.

The language model's neural maths (matrix multiplies, attention, sampling) runs on the CPU via
llama.cpp - a SHA-256 ASIC cannot do it. Every stage of the chat pipeline that *is* SHA-256
work runs on the hasher-server when one is reachable:

  * prompt fingerprinting for the response cache,
  * the content-addressed n-gram index that drafts tokens for speculative decoding,
  * proof-of-work seals over each turn's transcript.

``llama-cpp-python`` is an optional dependency; everything except :class:`ChatEngine`'s model
loading works without it.
"""
