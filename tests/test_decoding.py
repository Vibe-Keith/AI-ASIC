"""Tests for the ASIC-assisted decoding pipeline: batched/async native nonce search, honest
host-vs-ASIC accounting, LSH buckets, KV-block routing, speculative-draft quality, retrieval
drafts, exact attention/verification metering, and background transcript seals."""
import hashlib
import struct
import time

import pytest

from ai_asic.chat import lsh
from ai_asic.chat.accelerator import HashAccelerator
from ai_asic.chat.draft import HashDraftIndex
from ai_asic.chat.engine import ChatEngine, TranscriptLog, verify_transcripts
from ai_asic.chat.metering import LlamaMeter
from ai_asic.chat.models_dir import ensure_layout
from ai_asic.chat.retrieval import DraftStore
from ai_asic.chat.routing import ContextRouter
from ai_asic.chat.simllm import ScriptedLlama
from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.server import HasherClient, HasherServer, VirtualAsicDevice
from ai_asic.workloads.split_model import DEVICE_ASIC, DEVICE_HOST

PARA = ("The miner hashes block headers by rolling a 32-bit nonce. Each chip runs many SHA-256 "
        "cores in parallel, and the control board collects any nonce whose double hash falls "
        "below the target. The host only sends work and checks the results.")
ECHO = "Repeat this paragraph exactly, word for word:\n\n" + PARA


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


def _engine(root, llm, **kw):
    opts = dict(seal_difficulty=0, use_cache=False)
    opts.update(kw)
    return ChatEngine(root=root, llm=llm, **opts)


# --- device / server: native nonce search ---------------------------------------------

def test_fast_virtual_chip_matches_cycle_honest_model():
    fast, slow = VirtualAsicDevice(fast=True), VirtualAsicDevice(fast=False)
    for seed in range(4):
        header = prepare_asic_job([seed] * 12, 0, timestamp=seed)
        a, b = fast.mine(header, 7, max_nonces=1 << 14), slow.mine(header, 7, max_nonces=1 << 14)
        assert (a.nonce, a.hash_hex, a.hashes_tried, a.found) == \
               (b.nonce, b.hash_hex, b.hashes_tried, b.found)


def test_mine_batch_one_round_trip_and_native_metrics(hserver):
    client = HasherClient("127.0.0.1", hserver.port)
    headers = [prepare_asic_job([i] * 12, 0, timestamp=0) for i in range(6)]
    before = hserver.metrics.snapshot()
    batch = client.mine_batch([(h, 8, 1 << 16) for h in headers])
    after = hserver.metrics.snapshot()
    assert after["total_requests"] - before["total_requests"] == 1
    singles = [client.mine(h, 8, max_nonces=1 << 16) for h in headers]
    assert [(r.nonce, r.hash_hex) for r in batch] == [(r.nonce, r.hash_hex) for r in singles]
    for h, r in zip(headers, batch):
        full = bytearray(h)
        struct.pack_into("<I", full, 76, r.nonce)
        assert hashlib.sha256(hashlib.sha256(bytes(full)).digest()).hexdigest() == r.hash_hex
    assert after["native_hashes"] - before["native_hashes"] == sum(r.hashes_tried for r in batch)
    client.compute_hash(b"x")
    m = hserver.metrics.snapshot()
    assert m["cpu_hashes"] == before["cpu_hashes"] + 1  # data hashing is never "native"


def test_client_falls_back_when_server_lacks_mine_batch(hserver):
    orig = hserver.dispatch

    def old_server(req):
        if req.get("method") == "MineBatch":
            return {"ok": False, "error": "unknown method: 'MineBatch'"}
        return orig(req)

    hserver._server.dispatch = old_server
    client = HasherClient("127.0.0.1", hserver.port)
    out = client.mine_batch([(prepare_asic_job([1] * 12, 0, timestamp=0), 6, 1 << 12)] * 2)
    assert len(out) == 2 and all(r.found for r in out) and not client._batch_supported


# --- accelerator: async, batched, warm, honest ----------------------------------------

def test_submitted_searches_share_one_round_trip(hserver):
    acc = HashAccelerator("127.0.0.1", hserver.port, keep_warm=False)
    headers = [prepare_asic_job([i] * 12, 0, timestamp=1) for i in range(10)]
    trips = acc.round_trips
    futures = acc.submit_mine_batch(headers, 6, max_nonces=4096, stage="t")
    results = [f.result(timeout=10) for f in futures]
    assert acc.round_trips - trips == 1
    assert all(r.found and r.device == DEVICE_ASIC for r in results)
    m = acc.meter("t")
    assert m.device == DEVICE_ASIC and m.jobs == 10 and m.calls == 1
    assert m.ops == sum(r.hashes_tried for r in results)
    acc.close()


def test_closed_accelerator_never_strands_a_search(hserver):
    acc = HashAccelerator("127.0.0.1", hserver.port)
    acc.close()
    out = acc.submit_mine(prepare_asic_job([2] * 12, 0, timestamp=0), 6).result(timeout=5)
    assert out.found and out.device == DEVICE_HOST


def test_seal_survives_accelerator_swap(root, hserver):
    acc = HashAccelerator("127.0.0.1", hserver.port)
    eng = _engine(root, ScriptedLlama(default_reply="ok"), accelerator=acc, seal_difficulty=10)
    r = eng.reply("hello")
    eng.set_accelerator(HashAccelerator())  # closes acc while its seal may still be mining
    assert r.wait_seal(timeout=10)["found"]
    eng.flush(timeout=10)
    assert verify_transcripts(root / "cache" / "transcripts.jsonl")[0]


def test_mine_batch_chunks_past_the_server_cap(hserver):
    client = HasherClient("127.0.0.1", hserver.port)
    jobs = [(prepare_asic_job([i] * 12, 0, timestamp=0), 1, 64) for i in range(300)]
    assert len(client.mine_batch(jobs)) == 300


def test_keep_warm_runs_while_idle_and_is_not_turn_work(hserver):
    acc = HashAccelerator("127.0.0.1", hserver.port, warm_interval=0.05)
    deadline = time.time() + 5
    while acc.warm.jobs < 3 and time.time() < deadline:
        time.sleep(0.02)
    assert acc.warm.jobs >= 3 and acc.warm.device == DEVICE_ASIC
    assert "warm" not in acc.meters
    acc.close()


def test_turn_trace_counts_only_nonce_search_as_asic(root, hserver):
    acc = HashAccelerator("127.0.0.1", hserver.port, keep_warm=False)
    eng = _engine(root, ScriptedLlama({ECHO: PARA}), accelerator=acc, seal_difficulty=8)
    r = eng.reply(ECHO, max_tokens=200)
    r.wait_seal()
    by = {s.name: s for s in r.trace.stages}
    assert by["seal"].device == DEVICE_ASIC and by["route.buckets"].device == DEVICE_ASIC
    for name in ("fingerprint", "llm", "draft", "route", "retrieve"):
        assert by[name].device == DEVICE_HOST, name
    assert r.trace.asic_ops == by["seal"].ops + by["route.buckets"].ops
    acc.close()


# --- LSH ------------------------------------------------------------------------------

def test_simhash_separates_related_from_unrelated():
    a = lsh.simhash("Explain the BM1387 chip's midstate and the 12-byte tail")
    b = lsh.simhash("What is the midstate the BM1387 chip receives and what is in the 12 byte "
                    "tail?")
    c = lsh.simhash("Give me a quick bread recipe with flour and yeast")
    assert lsh.similarity(a, b) > lsh.similarity(a, c) + 0.15


def test_bucket_ids_are_cached_and_device_independent(hserver):
    host = lsh.BucketHasher(HashAccelerator())
    acc = HashAccelerator("127.0.0.1", hserver.port, keep_warm=False)
    asic = lsh.BucketHasher(acc)
    bl = [lsh.bands(lsh.simhash("routing test text")), lsh.bands(lsh.simhash("another one"))]
    (h1, h2), _ = host.buckets(bl)
    (a1, a2), st = asic.buckets(bl)
    assert (h1, h2) == (a1, a2) and st.searches == 20 and st.device == DEVICE_ASIC
    _, st2 = asic.buckets(bl)
    assert st2.searches == 0 and st2.cache_hits == 20
    assert lsh.collide(h1, h1) and not lsh.collide(h1, [None] * len(h1))
    acc.close()


# --- routing --------------------------------------------------------------------------

def _words(m):
    return len(m["content"].split()) + 6


def _chat(n, words=40):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs += [{"role": "user", "content": f"topic{i} question " * 5},
                 {"role": "assistant", "content": f"topic{i} answer " * (words // 2)}]
    return msgs


def test_router_is_noop_under_budget():
    r = ContextRouter(lsh.BucketHasher(HashAccelerator()), budget_tokens=10_000)
    msgs = _chat(3) + [{"role": "user", "content": "hi"}]
    res = r.route(msgs, _words)
    assert not res.routed and res.messages == msgs and res.reason == "fits budget"


def test_router_compacts_keeps_relevant_old_exchange_then_appends():
    router = ContextRouter(lsh.BucketHasher(HashAccelerator()), budget_tokens=400,
                           keep_recent=2, low_water=0.6)
    msgs = _chat(10)
    old = {"role": "user", "content": "How do I align the two mirror beam steering setup?"}
    old_a = {"role": "assistant", "content": "Use the first mirror for position and the second "
                                             "mirror for angle, iterating near and far targets."}
    msgs[3:5] = [old, old_a]  # exchange 1 is about mirror alignment
    q = {"role": "user", "content": "For the two mirror beam steering alignment, which mirror "
                                    "sets the angle?"}
    res = router.route(msgs + [q], _words)
    assert res.compacted and res.routed and res.tokens_routed <= 0.6 * 400
    assert 1 in res.kept                               # recalled by bucket match
    assert res.kept[-2:] == [8, 9]                     # the recent window
    assert res.messages[0]["role"] == "system" and res.messages[-1] == q
    # Next turn: same selection plus the new exchange appended (KV prefix reusable).
    nxt = msgs + [q, {"role": "assistant", "content": "The second mirror."},
                  {"role": "user", "content": "thanks"}]
    res2 = router.route(nxt, _words)
    assert not res2.compacted and res2.kept == res.kept + [10]
    assert res2.messages[:len(res.messages) - 1] == res.messages[:-1]


@pytest.mark.parametrize("turns,limit", [(16, 0.8), (24, 0.25)])
def test_routing_cuts_attention_in_a_long_live_chat(root, turns, limit):
    """Under the context window routing still attends over less; past it, the old
    oldest-first trim re-prefills the whole prompt every turn and routing avoids that."""
    hist = [(f"Question {i} about subject{i} and detail{i}?",
             " ".join(f"word{i}x{j}" for j in range(40)) + ".") for i in range(turns)]
    totals = {}
    for route in (False, True):
        llm = ScriptedLlama(dict(hist))
        eng = _engine(root, llm, use_draft=False, draft_retrieval=False, route_context=route,
                      route_budget_tokens=400, n_ctx=1024, max_tokens=64)
        totals[route] = sum(eng.reply(q).eval.attention_pairs for q, _ in hist)
    assert totals[True] < limit * totals[False]


# --- metering -------------------------------------------------------------------------

def test_meter_counts_attention_pairs_exactly():
    llm = ScriptedLlama({"hi": "a b c"})
    m = LlamaMeter(llm)
    m.begin()
    list(llm.create_chat_completion([{"role": "user", "content": "hi"}], max_tokens=10))
    s = m.end()
    prompt = len(llm.tokenize(b"<|user|> hi <|end|> <|assistant|>"))
    assert s.prefill_tokens == prompt and s.reused_tokens == 0
    assert s.prefill_attention == prompt * (prompt + 1) // 2
    # 3 words + EOS sampled; each decode pass evaluates one token on a growing cache
    assert s.sampled == 4 and s.decode_passes == 3 and s.tokens_per_pass == 1.0
    assert s.decode_attention == sum(prompt + i + 1 for i in range(3))


def test_drafts_raise_tokens_per_pass_and_keep_the_reply(root):
    off = _engine(root, ScriptedLlama({ECHO: PARA}), use_draft=False).reply(ECHO, max_tokens=200)
    on = _engine(root, ScriptedLlama({ECHO: PARA})).reply(ECHO, max_tokens=200)
    assert on.text == off.text == PARA
    assert off.tokens_per_pass == pytest.approx(1.0)
    assert on.tokens_per_pass > 5 and on.eval.decode_passes < off.eval.decode_passes / 5


def test_strength_caps_waste_less_than_legacy_lookup_on_free_text(root):
    q = "In three sentences, explain why the sky looks blue."
    a = ("Sunlight contains all colours, and when it enters the atmosphere the short blue "
         "wavelengths are scattered by air molecules far more than red ones. This scattering "
         "sends blue light toward our eyes from every direction of the sky.")
    legacy = _engine(root, ScriptedLlama({q: a}), draft_mode="lookup", draft_len_by_match=[],
                     draft_max_ngram=3, draft_num_pred=10).reply(q, max_tokens=200)
    new = _engine(root, ScriptedLlama({q: a})).reply(q, max_tokens=200)
    assert new.text == legacy.text
    assert new.eval.rejected_tokens < legacy.eval.rejected_tokens / 2
    assert new.eval.attention_pairs < legacy.eval.attention_pairs


# --- retrieval drafts -----------------------------------------------------------------

def test_retrieval_drafts_from_a_past_reply(root):
    q1 = "Explain how the BM1387 chip uses the midstate and the 12-byte tail."
    q2 = "Explain how the BM1387 chip uses its midstate and 12 byte tail."
    ans = ("The host hashes the first 64 header bytes into a midstate and sends it with the "
           "12-byte tail; the chip rolls the nonce and finishes the second block itself.")
    llm = ScriptedLlama({q1: ans, q2: ans})
    eng = _engine(root, llm)
    first = eng.reply(q1, max_tokens=100)
    eng.new_chat()
    second = eng.reply(q2, max_tokens=100)
    assert first.retrieved == 0 and second.retrieved == 1
    assert second.tokens_per_pass > 2 * first.tokens_per_pass


def test_draft_store_persists_and_ranks(tmp_path):
    path = tmp_path / "store.jsonl"
    store = DraftStore(path)
    sig = lsh.simhash("alpha beta gamma delta")
    b = [1, 2, 3]
    store.add(sig, b, "reply one")
    store.add(sig ^ 0xFFFF, b, "reply two")
    again = DraftStore(path)
    hits = again.lookup(sig, b, k=2, min_similarity=0.5)
    assert [t for _, t in hits] == ["reply one", "reply two"] and hits[0][0] == 1.0
    assert again.lookup(sig, [9, 9, 9]) == []


# --- background seals -----------------------------------------------------------------

def test_background_seals_stay_chained_and_ordered(root):
    eng = _engine(root, ScriptedLlama(default_reply="ok"), seal_difficulty=10, use_draft=False)
    results = [eng.reply(f"turn {i}") for i in range(6)]
    eng.flush()
    ok, n, problems = verify_transcripts(root / "cache" / "transcripts.jsonl")
    assert ok and n == 6, problems
    assert [r.seal["turn"]["user"] for r in results] == [f"turn {i}" for i in range(6)]


def test_engines_on_one_log_share_the_chain(root):
    a = _engine(root, ScriptedLlama(default_reply="a"), seal_difficulty=6)
    b = _engine(root, ScriptedLlama(default_reply="b"), seal_difficulty=6)
    assert a.log is b.log is TranscriptLog.for_path(root / "cache" / "transcripts.jsonl")
    a.reply("one"); b.reply("two"); a.reply("three")
    a.flush()
    ok, n, _ = verify_transcripts(root / "cache" / "transcripts.jsonl")
    assert ok and n == 3
