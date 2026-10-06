import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

from ai_asic.chat.accelerator import HashAccelerator
from ai_asic.chat.download import DownloadError, ModelSpec, download
from ai_asic.chat.draft import HashDraftIndex
from ai_asic.chat.engine import ChatEngine, verify_transcripts
from ai_asic.chat.models_dir import ensure_layout, load_config, resolve_llm
from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.server import HasherServer, VirtualAsicDevice
from ai_asic.workloads.split_model import DEVICE_ASIC, DEVICE_HOST

REAL_MODEL = (Path(__file__).resolve().parents[1] / "ai_models" / "llm"
              / "qwen2.5-0.5b-instruct-q4_k_m.gguf")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ASIC_MODELS", str(tmp_path / "ai_models"))
    return ensure_layout(tmp_path / "ai_models")


@pytest.fixture
def hserver():
    srv = HasherServer(device=VirtualAsicDevice("Antminer S9"), port=0).start()
    try:
        yield srv
    finally:
        srv.stop()


class FakeLlama:
    """Stands in for llama_cpp.Llama: streams a canned reply word by word."""

    def __init__(self, reply="Hello from the fake model.", by_char=False):
        self.reply = reply
        self.by_char = by_char  # stream one character per chunk (chunks != tokens)
        self.calls = 0

    def create_chat_completion(self, messages, stream, max_tokens, temperature):
        self.calls += 1
        if self.by_char:
            for ch in self.reply:
                yield {"choices": [{"delta": {"content": ch}}]}
            return
        words = self.reply.split(" ")
        for i, w in enumerate(words):
            yield {"choices": [{"delta": {"content": w if i == 0 else " " + w}}]}

    def tokenize(self, data, add_bos=False):
        return data.split()


# --- ai_models layout ----------------------------------------------------------

def test_layout_and_config(root):
    for sub in ("llm", "hasher", "cache"):
        assert (root / sub).is_dir()
    assert (root / "README.md").is_file()
    cfg = load_config(root)
    assert cfg["n_ctx"] == 2048 and cfg["use_draft"] is True
    assert resolve_llm(root=root) is None
    (root / "llm" / "tiny.gguf").write_bytes(b"GGUF")
    assert resolve_llm(root=root).name == "tiny.gguf"
    assert resolve_llm("tiny.gguf", root=root).name == "tiny.gguf"


# --- downloader ----------------------------------------------------------------

def _spec_for(path: Path, sha: str, size: int) -> ModelSpec:
    return ModelSpec("t", "model.gguf", path.as_uri(), size, sha, "test")


def test_download_verifies_sha256(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 5000)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    out = download(_spec_for(src, sha, 5000), dest_dir=tmp_path / "dl", chunk_size=1024)
    assert out.read_bytes() == src.read_bytes()


def test_download_rejects_bad_sha256(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"y" * 100)
    with pytest.raises(DownloadError):
        download(_spec_for(src, "0" * 64, 100), dest_dir=tmp_path / "dl")
    assert not (tmp_path / "dl" / "model.gguf").exists()
    assert not (tmp_path / "dl" / "model.gguf.part").exists()


# --- accelerator ------------------------------------------------------------------

def test_accelerator_hashing_is_host_work_nonce_search_is_asic(hserver):
    local = HashAccelerator()
    asic = HashAccelerator("127.0.0.1", hserver.port)
    # Arbitrary-data SHA-256 is not BM1387 work: it runs (and is counted) on the host.
    assert local.device == DEVICE_HOST and asic.device == DEVICE_HOST
    assert local.nonce_device == DEVICE_HOST and asic.nonce_device == DEVICE_ASIC
    data = [i.to_bytes(4, "little") for i in range(600)]
    before = hserver.metrics.snapshot()["cpu_hashes"]
    assert asic.hash_batch(data, stage="t") == local.hash_batch(data, stage="t")
    assert asic.meter("t").ops == 600 and asic.meter("t").device == DEVICE_HOST
    assert hserver.metrics.snapshot()["cpu_hashes"] == before  # no RPC for host hashing
    assert asic.hash(b"abc") == hashlib.sha256(b"abc").digest()


def test_accelerator_mine_verifies(hserver):
    asic = HashAccelerator("127.0.0.1", hserver.port)
    header = prepare_asic_job([3] * 12, 0, timestamp=0)
    out = asic.mine(header, 10)
    full = bytearray(header)
    struct.pack_into("<I", full, 76, out.nonce)
    assert hashlib.sha256(hashlib.sha256(bytes(full)).digest()).hexdigest() == out.hash_hex


def test_accelerator_falls_back_when_server_dies():
    srv = HasherServer(device=VirtualAsicDevice(), port=0).start()
    acc = HashAccelerator("127.0.0.1", srv.port, keep_warm=False)
    assert acc.is_hardware
    srv.stop()
    out = acc.mine(prepare_asic_job([1] * 12, 0, timestamp=0), 6)
    assert out.found and out.device == DEVICE_HOST
    assert not acc.is_hardware and "unreachable" in acc.label
    assert acc.hash(b"still works") == hashlib.sha256(b"still works").digest()


# --- speculative draft index -------------------------------------------------------

def test_draft_proposes_continuation_of_repeated_ngram():
    idx = HashDraftIndex(HashAccelerator(), max_ngram=3, num_pred=4)
    draft = idx([1, 2, 3, 4, 5, 9, 1, 2, 3])
    assert draft == [4, 5, 9, 1]  # what followed the earlier "1 2 3"
    assert idx([7, 8]) == []      # a rewrite with no repeats proposes nothing


def test_draft_index_handles_rewind():
    acc = HashAccelerator()
    rewound = HashDraftIndex(acc)
    rewound([5, 6, 7, 8, 9, 5, 6])
    rewound([5, 6, 7, 1, 2])            # llama.cpp rejected a draft and rewrote the tail
    fresh = HashDraftIndex(acc)
    fresh([5, 6, 7, 1, 2])
    assert rewound.tokens == fresh.tokens
    assert rewound._index == fresh._index
    assert rewound._by_end == fresh._by_end


def test_draft_acceptance_follows_llama_call_pattern():
    idx = HashDraftIndex(HashAccelerator(), max_ngram=3, num_pred=4)
    ctx = [1, 2, 3, 4, 5, 1, 2]
    assert idx(ctx) == [3, 4, 5, 1]
    # llama.cpp kept 3, 4, rejected 5 and sampled 9 instead; next call shows that.
    idx(ctx + [3, 4, 9])
    assert (idx.accepted, idx.scored) == (2, 4)
    # A fully accepted draft arrives followed by one freshly sampled token.
    seq = [7, 8, 7]
    idx.reset()
    assert idx(seq) == [8, 7]
    idx(seq + [8, 7, 6])
    assert (idx.accepted, idx.scored) == (4, 6)  # counters are cumulative across reset()


def test_draft_acceptance_ignores_new_generation_and_discarded():
    idx = HashDraftIndex(HashAccelerator(), max_ngram=2, num_pred=3)
    assert idx([1, 2, 1])
    idx([5, 6, 7])                 # not an extension: a new generation, nothing scored
    assert idx.scored == 0
    assert idx([5, 6, 7, 5]) == [6, 7, 5]
    idx.discard_pending()          # generation ended; its last draft is never verified
    idx([5, 6, 7, 5, 6, 7])
    assert idx.scored == 0


def test_draft_same_on_host_and_asic(hserver):
    seq = [11, 12, 13, 14, 11, 12, 13, 99, 11, 12]
    host = HashDraftIndex(HashAccelerator())(seq)
    asic = HashDraftIndex(HashAccelerator("127.0.0.1", hserver.port))(seq)
    assert host == asic and host


# --- engine (fake LLM) --------------------------------------------------------------

def test_engine_reply_and_trace(root, hserver):
    fake = FakeLlama("The chip rolls the nonce.")
    eng = ChatEngine(accelerator=HashAccelerator("127.0.0.1", hserver.port), root=root,
                     llm=fake, seal_difficulty=8)
    streamed = []
    r = eng.reply("What does the chip do?", on_token=streamed.append)
    assert r.text == "The chip rolls the nonce." and "".join(streamed) == r.text
    assert r.seal and r.seal["found"]  # waits for the background seal
    stages = {s.name: s for s in r.trace.stages}
    assert stages["fingerprint"].device == DEVICE_HOST  # hashing data is not ASIC work
    assert stages["llm"].device == DEVICE_HOST
    assert stages["seal"].device == DEVICE_ASIC        # the nonce search is
    assert stages["seal"].ops == r.seal["hashes"] > 0
    assert eng.messages[-1] == {"role": "assistant", "content": r.text}


def test_engine_counts_tokens_not_stream_chunks(root):
    eng = ChatEngine(root=root, llm=FakeLlama("one two three four", by_char=True),
                     seal_difficulty=0, use_cache=False)
    r = eng.reply("hi")
    assert r.tokens == 4               # 18 chunks, 4 tokens (FakeLlama tokenizes on spaces)
    assert r.ttft_s >= 0 and r.decode_tps > 0
    assert "4 tokens" in {s.name: s for s in r.trace.stages}["llm"].detail


def test_bench_runs_and_summarizes(root):
    from ai_asic.chat.bench import parse_configs, run_bench, summarize

    assert parse_configs("off, 3x10,4x16") == [None, (3, 10), (4, 16)]
    with pytest.raises(ValueError):
        parse_configs("fast")
    prompts = [("a", "say a"), ("b", "say b")]
    rows = run_bench(None, [None, (3, 10)], prompts=prompts, max_tokens=8, root=root,
                     make_llm=lambda: FakeLlama("same reply every time"))
    assert [(r.config, r.prompt) for r in rows] == [
        ("off", "a"), ("off", "b"), ("3x10", "a"), ("3x10", "b")]
    assert all(r.tokens == 4 for r in rows)
    s = summarize(rows)
    assert [x.config for x in s] == ["off", "3x10"]
    assert s[0].speedup == pytest.approx(1.0)
    assert s[1].same_text == s[1].prompts == 2


def test_engine_draft_shape_from_config(root):
    eng = ChatEngine(root=root, llm=FakeLlama(), draft_max_ngram=6, draft_num_pred=24)
    assert eng.draft.max_ngram == 6 and eng.draft.num_pred == 24
    assert load_config(root)["draft_num_pred"] == 16  # default stays put


def test_set_accelerator_closes_previous_connection(root, hserver):
    old = HashAccelerator("127.0.0.1", hserver.port)
    eng = ChatEngine(root=root, llm=FakeLlama(), accelerator=old)
    client = old._client
    assert client._sock is not None
    eng.set_accelerator(HashAccelerator())
    assert client._sock is None and old._client is None


def test_engine_cache_hit_skips_llm(root):
    fake = FakeLlama()
    eng = ChatEngine(root=root, llm=fake, seal_difficulty=0)
    first = eng.reply("hi")
    eng.new_chat()
    second = eng.reply("hi")
    assert not first.cached and second.cached
    assert second.text == first.text
    assert fake.calls == 1
    assert [s.name for s in second.trace.stages] == ["fingerprint"]


def test_engine_cache_can_be_disabled(root):
    fake = FakeLlama()
    eng = ChatEngine(root=root, llm=fake, seal_difficulty=0, use_cache=False)
    eng.reply("hi"); eng.new_chat(); eng.reply("hi")
    assert fake.calls == 2


def test_transcript_seals_verify_and_detect_tampering(root):
    eng = ChatEngine(root=root, llm=FakeLlama("ok"), seal_difficulty=8, use_cache=False)
    eng.reply("one")
    eng.reply("two")
    eng.flush()  # seals are mined in the background
    log = root / "cache" / "transcripts.jsonl"
    ok, n, problems = verify_transcripts(log)
    assert ok and n == 2 and not problems

    lines = log.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["turn"]["assistant"] = "edited after the fact"
    log.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n", encoding="utf-8")
    ok, _, problems = verify_transcripts(log)
    assert not ok and any("altered" in p for p in problems)


def test_context_trimming_keeps_system_prompt(root):
    eng = ChatEngine(root=root, llm=FakeLlama(), n_ctx=120, max_tokens=40)
    msgs = [eng.messages[0]]
    for i in range(30):
        msgs += [{"role": "user", "content": f"question {i} " * 3},
                 {"role": "assistant", "content": f"answer {i} " * 3}]
    msgs.append({"role": "user", "content": "latest"})
    fitted = eng._fit_context(msgs, 40)
    assert fitted[0]["role"] == "system"
    assert fitted[-1]["content"] == "latest"
    assert len(fitted) < len(msgs)


# --- real model (only when downloaded) ----------------------------------------------

@pytest.mark.skipif(not REAL_MODEL.is_file() or importlib.util.find_spec("llama_cpp") is None,
                    reason="real GGUF model or llama-cpp-python not available")
def test_real_model_replies(root, hserver):
    eng = ChatEngine(str(REAL_MODEL), accelerator=HashAccelerator("127.0.0.1", hserver.port),
                     root=root, temperature=0.0, max_tokens=16, seal_difficulty=0)
    r = eng.reply("Reply with the single word: yes")
    assert r.text.strip()
    assert {s.name for s in r.trace.stages} >= {"fingerprint", "llm", "draft"}
