"""AI-ASIC command-line interface.

Cross-platform (works on Windows via `py -m ai_asic.cli` or the `ai-asic` console script).

Commands:
  detect       Detect the attached ASIC miner and print its profile/capabilities.
  profiles     List all supported miner models.
  infer        Run recursive inference on an input string.
  encode-work  Build a BM1387 (S9) work frame from an 80-byte header (hex).
  simulate     Run a virtual BM1387 chip end-to-end and verify it actually mines.
  serve        Run the on-device hasher-server (compute + mining) over a device backend.
  workload     Run a hash-based model split across the host CPU and an ASIC miner.
  models       List or download the AI files in ai_models/.
  chat         Chat with a local LLM; SHA-256 stages offload to the hasher-server.
"""
from __future__ import annotations

import argparse
import sys
import time

from ai_asic import __version__
from ai_asic.hardware import bm1387
from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.hardware.device_detector import detect_asic, detection_summary
from ai_asic.hardware.miner_profiles import all_profiles, detect_profile
from ai_asic.hardware.simulator import SimConfig, simulate_header
from ai_asic.server import (
    DEFAULT_PORT as HASHER_PORT,
    ChainAsicDevice,
    HasherServer,
    VirtualAsicDevice,
    auto_device,
)
from ai_asic.workloads.split_model import MiningBackend, SplitModel
from ai_asic.hashing.methods import ASICHashMethod, SoftwareHashMethod
from ai_asic.hashing.network import HashNetwork
from ai_asic.hashing.recursive import RecursiveEngine


def _cmd_detect(args: argparse.Namespace) -> int:
    print(detection_summary(args.host))
    return 0


def _cmd_profiles(args: argparse.Namespace) -> int:
    print(f"{'Model':<20}{'Chip':<10}{'Protocol':<16}{'Conn':<10}{'Hashrate':>12}")
    print("-" * 68)
    for p in all_profiles():
        hr = f"{p.nominal_hashrate / 1e12:.2f} TH/s"
        print(f"{p.model:<20}{p.chip:<10}{p.protocol.value:<16}{p.connection.value:<10}{hr:>12}")
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    net = HashNetwork(args.input_size, args.hidden1, args.hidden2, args.classes)
    if args.asic:
        method = ASICHashMethod(args.host, args.port)
        print(f"Hash method: {method.name()}")
    else:
        method = SoftwareHashMethod()
    engine = RecursiveEngine(net, passes=args.passes, jitter=args.jitter,
                             seed_rotation=args.seed_rotation, hash_method=method)
    result = engine.infer(args.text.encode("utf-8"))
    c = result.consensus
    print(f"Prediction:        class {c.prediction}")
    print(f"Consensus conf:    {c.confidence:.3f}")
    print(f"Avg per-pass conf: {c.average_confidence:.3f}")
    print(f"Valid passes:      {result.valid_passes}/{result.total_passes}")
    print(f"Latency:           {result.latency_s * 1000:.1f} ms")
    print(f"Using hardware:    {engine.is_using_hardware()}")
    return 0


def _cmd_encode_work(args: argparse.Namespace) -> int:
    if args.header:
        header = bytes.fromhex(args.header)
        if len(header) != 80:
            print("error: header must be 80 bytes (160 hex chars)", file=sys.stderr)
            return 2
    else:
        # Build a demo header from zero slots.
        header = prepare_asic_job([0] * 12, 0, timestamp=0)
    work = bm1387.new_work_from_header(header, args.work_id)
    frame = work.encode()
    print(f"work_id:   {args.work_id}")
    print(f"midstate:  {work.midstates[0].hex()}")
    print(f"data(12):  {work.data.hex()}")
    print(f"frame:     {frame.hex()}")
    print(f"frame len: {len(frame)} bytes")
    return 0


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0 s"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 90:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def _cmd_simulate(args: argparse.Namespace) -> int:
    if args.header:
        header = bytes.fromhex(args.header)
        if len(header) != 80:
            print("error: header must be 80 bytes (160 hex chars)", file=sys.stderr)
            return 2
    else:
        # Deterministic demo header: slots seeded from the work id so repeated runs differ.
        header = prepare_asic_job([args.work_id] * 12, 0, timestamp=0)

    profile = detect_profile(args.model) if args.model else None
    cfg = SimConfig(
        difficulty_bits=args.difficulty,
        chip_count=profile.chip_count if profile and profile.chip_count else args.chips,
        nominal_hashrate=float(profile.nominal_hashrate) if profile else args.hashrate,
        max_nonces=args.max_nonces,
    )

    chip_name = f"{profile.model} ({profile.chip})" if profile else "BM1387 (generic S9)"
    print(f"Virtual miner:     {chip_name}")
    print(f"Target difficulty: {cfg.difficulty_bits} leading zero bits")
    print(f"Modeled chips:     {cfg.chip_count}  @  {cfg.nominal_hashrate / 1e12:.2f} TH/s nominal")
    print("Mining from midstate + 12-byte tail only (as the real chip does)...")
    print()

    sim = simulate_header(header, work_id=args.work_id, config=cfg)
    r = sim.result

    if not r.found:
        print(f"No golden nonce in {r.hashes_tried:,} rolls "
              f"(raise --max-nonces or lower --difficulty).")
        print(f"Best run time:     {_fmt_duration(r.sim_seconds)}")
        return 1

    print(f"Golden nonce:      0x{r.nonce:08x}  ({r.nonce})")
    print(f"Midstate index:    {r.midstate_index}   (modeled chip #{r.chip_id})")
    print(f"Chip hash:         {r.hash_hex}")
    print(f"Leading zero bits: {r.leading_zeros}  (target {cfg.difficulty_bits})")
    print(f"Nonces rolled:     {r.hashes_tried:,}")
    print(f"Response frame:    {r.response_frame.hex()}  -> host parsed nonce 0x{sim.parsed.nonce:08x}")
    print()
    print(f"Simulated run:     {_fmt_duration(r.sim_seconds)} (pure-Python model)")
    print(f"Real-chip estimate:{_fmt_duration(r.est_real_seconds)} at {cfg.nominal_hashrate / 1e12:.1f} TH/s")
    print()
    print("Verification (would this drive a real S9?):")
    print(f"  midstate math == hashlib over full header : {_check(sim.midstate_matches_header)}")
    print(f"  response frame round-trips                : {_check(sim.response_roundtrips)}")
    print(f"  independent hash meets target             : {_check(r.leading_zeros >= cfg.difficulty_bits)}")
    print()
    print("  RESULT: " + ("PASS - the work/mine/nonce pipeline is correct."
                          if sim.ok else "FAIL - see mismatches above."))
    return 0 if sim.ok else 1


def _check(ok: bool) -> str:
    return "OK" if ok else "MISMATCH"


def _print_trace(trace) -> None:
    print(f"    {'Stage':<12}{'Device':<10}{'Ops':>7}{'Time':>11}  Detail")
    for s in trace.stages:
        print(f"    {s.name:<12}{s.device:<10}{s.ops:>7}{_fmt_duration(s.seconds):>11}  {s.detail}")


def _cmd_models(args: argparse.Namespace) -> int:
    from ai_asic.chat.download import KNOWN_MODELS, DownloadError, download
    from ai_asic.chat.models_dir import ensure_layout, list_llm_models, load_config

    root = ensure_layout()
    if args.action == "download":
        spec = KNOWN_MODELS.get(args.name or "")
        if spec is None:
            print(f"error: unknown model {args.name!r}; known: {', '.join(KNOWN_MODELS)}",
                  file=sys.stderr)
            return 2
        print(f"Downloading {spec.filename} ({spec.size / 2**20:.0f} MB) from Hugging Face...")
        last = [0.0]

        def progress(done: int, total: int) -> None:
            if time.time() - last[0] > 2 or done == total:
                last[0] = time.time()
                print(f"  {done / 2**20:7.1f} / {total / 2**20:.1f} MB", flush=True)

        try:
            path = download(spec, progress=progress)
        except DownloadError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Saved and SHA-256 verified: {path}")
        return 0

    cfg = load_config(root)
    print(f"AI files root:  {root}")
    print(f"Default LLM:    {cfg['default_llm']}")
    print("LLM models (llm/):")
    models = list_llm_models(root)
    for p in models:
        print(f"  {p.name:<44}{p.stat().st_size / 2**20:8.0f} MB")
    if not models:
        print("  (none - run 'models download qwen2.5-0.5b' or drop a .gguf in llm/)")
    hashers = sorted((root / "hasher").glob("*.json"))
    print(f"HASHER split-models (hasher/): {len(hashers)}")
    for p in hashers:
        print(f"  {p.name}")
    print("Downloadable:")
    for spec in KNOWN_MODELS.values():
        print(f"  {spec.key:<16}{spec.description}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from ai_asic.chat.accelerator import HashAccelerator
    from ai_asic.chat.engine import ChatEngine, verify_transcripts
    from ai_asic.chat.models_dir import cache_dir

    try:
        sys.stdout.reconfigure(errors="replace")  # model output may not fit the console codepage
    except (AttributeError, ValueError):
        pass

    if args.verify:
        path = cache_dir() / "transcripts.jsonl"
        ok, n, problems = verify_transcripts(path)
        print(f"Transcript: {path}")
        print(f"Turns checked: {n}  ->  {'ALL SEALS VALID' if ok else 'TAMPERING DETECTED'}")
        for p in problems:
            print(f"  {p}")
        return 0 if ok else 1

    pool = None
    if args.stratum:
        pool, _ = _start_pool(args)
        if args.wait_miner > 0:
            print(f"Waiting up to {args.wait_miner:.0f}s for the miner...", flush=True)
            if not pool.wait_for_miner(args.wait_miner):
                print("  no miner yet; seals run on the host until one connects")
        acc = HashAccelerator(pool=pool)
    else:
        acc = HashAccelerator() if args.no_asic else HashAccelerator(args.host, args.port)
    overrides = {"use_draft": not args.no_draft, "use_cache": not args.no_cache}
    if args.seal_difficulty is not None:
        overrides["seal_difficulty"] = args.seal_difficulty
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens
    if args.draft_ngram is not None:
        overrides["draft_max_ngram"] = args.draft_ngram
    if args.draft_tokens is not None:
        overrides["draft_num_pred"] = args.draft_tokens
    engine =ChatEngine(args.model, accelerator=acc, **overrides)

    print(f"Model:        {engine.model_name}")
    print(f"Accelerator:  {acc.label}")
    try:
        engine.load()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def turn(text: str) -> None:
        print("bot> ", end="", flush=True)
        result = engine.reply(text, on_token=lambda t: print(t, end="", flush=True))
        print()
        if args.trace or args.once:
            _print_trace(result.trace)

    if args.once:
        turn(args.once)
        return 0

    print("Type a message. Commands: /new (fresh chat), /trace (toggle stage trace), /quit")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/new":
            engine.new_chat()
            print("(new chat)")
            continue
        if text == "/trace":
            args.trace = not args.trace
            print(f"(trace {'on' if args.trace else 'off'})")
            continue
        turn(text)


def _start_pool(args: argparse.Namespace, virtual: bool = False):
    """Start the private stratum pool and tell the user how to point the miner at it.
    Returns ``(pool, url)``."""
    from ai_asic.stratum import DEFAULT_SHARE_DIFFICULTY, StratumPool
    from ai_asic.stratum.miner_setup import guess_lan_ip, lan_ip_towards, other_active_pools

    diff = args.share_difficulty
    if diff is None:
        diff = 2 ** -16 if virtual else DEFAULT_SHARE_DIFFICULTY  # pure-Python miner is slow
    listen = "127.0.0.1" if virtual else args.listen
    allow = [args.miner] if (args.miner and not virtual) else None
    pool = StratumPool(listen, args.stratum_port, share_difficulty=diff, allow=allow,
                       log=lambda m: print(f"  [pool] {m}", flush=True)).start()
    if virtual:
        return pool, f"stratum+tcp://127.0.0.1:{pool.port}"
    try:
        ip = lan_ip_towards(args.miner) if args.miner else guess_lan_ip()
    except OSError:
        ip = guess_lan_ip()
    url = f"stratum+tcp://{ip}:{pool.port}"
    print(f"Private pool:  {url}   (any worker name / password)")
    if args.miner:
        others = other_active_pools(args.miner, url)
        if others:
            print(f"  note: the miner also has {len(others)} other pool(s) configured "
                  f"({', '.join(others)}). It fails over to them if this host stops; remove "
                  "them in the miner's web UI to keep the loop fully closed.")
        if args.configure_miner:
            from ai_asic.stratum.miner_setup import point_miner_at

            try:
                print("  " + point_miner_at(args.miner, url))
            except Exception as exc:
                print(f"  could not configure the miner automatically: {exc}")
    if not (args.miner and args.configure_miner):
        print("  Set the miner's Pool 1 URL to the address above (web UI -> Miner "
              "Configuration), or rerun with --miner <ip> --configure-miner.")
    print("  Windows may ask to allow Python through the firewall; allow it on Private "
          "networks so the miner can connect.")
    return pool, url


def _cmd_hwtest(args: argparse.Namespace) -> int:
    import os

    from ai_asic.stratum import VirtualStratumMiner
    from ai_asic.stratum import protocol as P

    pool, url = _start_pool(args, virtual=args.virtual)
    vminer = VirtualStratumMiner("127.0.0.1", pool.port).start() if args.virtual else None
    try:
        print(f"Waiting up to {args.wait:.0f}s for a miner to connect...", flush=True)
        if not pool.wait_for_miner(args.wait):
            print("No miner connected. Check: the miner's pool URL is exactly "
                  f"{url}; the host firewall allows inbound TCP {pool.port}; the miner and "
                  "this host are on the same network.", file=sys.stderr)
            return 1
        m = pool.miners()[0]
        print(f"Miner:         {m.address}  {m.user_agent or '(no user agent)'}  "
              f"worker {m.worker!r}")
        print(f"Version roll:  {'mask 0x%08x' % m.version_mask if m.version_mask else 'off'}")
        share_bits = P.bits_for_difficulty(pool.share_difficulty)
        bits = share_bits if args.bits is None else max(args.bits, 0)
        print(f"\nSealing {args.seals} test digests at >= {bits} bits "
              f"(share difficulty {pool.share_difficulty:g} = {share_bits} bits per share):")
        ok_count, times = 0, []
        prev = b"\x00" * 32
        for i in range(args.seals):
            digest = os.urandom(32)
            t0 = time.perf_counter()
            share = pool.mine(prev, digest, int(time.time()), min_bits=bits,
                              timeout=args.timeout)
            dt = time.perf_counter() - t0
            if share is None:
                print(f"  seal {i + 1}: no share within {args.timeout:.0f}s")
                continue
            err = P.verify_seal(prev, digest, share.coinbase, share.header, bits)
            times.append(dt)
            ok_count += err is None
            print(f"  seal {i + 1}: {dt * 1000:7.0f} ms  nonce 0x{share.nonce:08x}  "
                  f"{share.leading_zeros} zero bits  {err or 'verified'}")
            prev = digest
        rate = pool.hashrate()
        print()
        print(f"Shares:        {m.accepted} accepted, {m.rejected} rejected"
              + (f" (last reject: {m.last_reject})" if m.rejected else ""))
        print(f"Hashrate:      {rate / 1e12:.3f} TH/s observed from shares" if rate >= 1e9
              else f"Hashrate:      {rate:,.0f} H/s observed from shares")
        if times:
            print(f"Seal latency:  {sum(times) / len(times) * 1000:.0f} ms mean, "
                  f"{max(times) * 1000:.0f} ms max")
        passed = ok_count == args.seals and m.rejected == 0
        print(f"Result:        {'PASS' if passed else 'FAIL'} - {ok_count}/{args.seals} seals "
              "mined and verified")
        if m.rejected:
            print("  Rejected shares mean the miner and pool disagree on the work; send the "
                  "reject reason above along with the miner's firmware version.")
        return 0 if passed else 1
    finally:
        if vminer:
            vminer.stop()
        pool.stop()


def _cmd_bench(args: argparse.Namespace) -> int:
    from ai_asic.chat.accelerator import HashAccelerator
    from ai_asic.chat.bench import (ROW_HEADER, format_row, format_summary, parse_configs,
                                    run_bench, summarize)

    try:
        configs = parse_configs(args.configs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    acc = HashAccelerator() if args.no_asic else HashAccelerator(args.host, args.port)
    print(f"Accelerator:  {acc.label}")
    print(f"Configs:      {args.configs}   max_tokens {args.max_tokens}, runs {args.runs}, "
          "greedy")
    print()
    print(ROW_HEADER)
    try:
        rows = run_bench(args.model, configs, max_tokens=args.max_tokens, runs=args.runs,
                         accelerator=acc, on_row=lambda r: print(format_row(r), flush=True))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print()
    for line in format_summary(summarize(rows)):
        print(line)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    if args.device == "virtual":
        device = VirtualAsicDevice(args.model)
    elif args.device == "chain":
        device = ChainAsicDevice(args.model, args.device_path)
    else:
        device = auto_device(args.model, args.device_path)

    srv = HasherServer(device=device, host=args.host, port=args.port,
                       difficulty_bits=args.difficulty).start()
    print(f"hasher-server:     {device.name()}")
    print(f"device available:  {device.available}")
    print(f"listening on:      {args.host}:{srv.port}  (difficulty {args.difficulty} bits)")
    print("Methods: ComputeHash, ComputeBatch, Mine, GetMetrics, GetDeviceInfo. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
        print("\nstopped.")
    return 0


def _cmd_workload(args: argparse.Namespace) -> int:
    if args.model_file:
        try:
            model = SplitModel.load(args.model_file)
            print(f"Loaded model:      {args.model_file}")
        except (OSError, ValueError) as exc:
            print(f"error: could not load model: {exc}", file=sys.stderr)
            return 2
    else:
        model = SplitModel(
            input_size=args.input_size, mining_neurons=args.mining_neurons,
            activation_width=args.activation_width, output_size=args.classes,
            difficulty_bits=args.difficulty, nonce_range=args.nonce_range,
        )
        print("Loaded model:      new random split-model")

    if args.save:
        model.save(args.save)
        print(f"Saved model:       {args.save}")

    use_asic = not args.no_asic
    backend = MiningBackend(
        host=args.host if use_asic else None,
        port=args.port if use_asic else None,
        difficulty_bits=model.difficulty_bits, max_nonces=model.nonce_range,
        prefer_asic=use_asic,
    )

    print(f"Mining device:     {backend.label}")
    print(f"Input:             {args.text!r}")
    print()

    result = model.infer(args.text.encode("utf-8"), backend)
    t = result.trace

    print(f"{'Stage':<10}{'Device':<10}{'Ops':>6}{'Time':>12}  Detail")
    print("-" * 72)
    for s in t.stages:
        print(f"{s.name:<10}{s.device:<10}{s.ops:>6}{_fmt_duration(s.seconds):>12}  {s.detail}")
    print("-" * 72)
    print(f"Host ops:  {t.host_ops:<6} ({_fmt_duration(t.host_seconds)})   "
          f"ASIC ops:  {t.asic_ops:<6} ({_fmt_duration(t.asic_seconds)})")
    print(f"Ran on hardware:   {t.asic_is_hardware}")
    print()
    print(f"Mined nonces:      {[hex(n) for n in result.nonces]}")
    print(f"Prediction:        class {result.prediction}  (score {result.confidence:.4f})")
    return 0


def _add_pool_args(p: argparse.ArgumentParser) -> None:
    from ai_asic.stratum import DEFAULT_STRATUM_PORT

    p.add_argument("--miner", default=None,
                   help="miner's LAN IP: only it may connect, and its API is used for setup")
    p.add_argument("--configure-miner", action="store_true",
                   help="point the miner at this host's pool via its API (addpool/switchpool)")
    p.add_argument("--listen", default="0.0.0.0", help="pool listen address (default 0.0.0.0)")
    p.add_argument("--stratum-port", type=int, default=DEFAULT_STRATUM_PORT,
                   help=f"pool port (default {DEFAULT_STRATUM_PORT})")
    p.add_argument("--share-difficulty", type=float, default=None,
                   help="pool share difficulty (default 256 for hardware; ~12 shares/s on an S9)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai-asic", description=__doc__)
    p.add_argument("--version", action="version", version=f"ai-asic {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="detect the attached ASIC miner")
    d.add_argument("--host", default=None, help="cgminer/bmminer host (default 127.0.0.1 / CGMINER_HOST)")
    d.set_defaults(func=_cmd_detect)

    pr = sub.add_parser("profiles", help="list supported miner models")
    pr.set_defaults(func=_cmd_profiles)

    inf = sub.add_parser("infer", help="run recursive inference on text")
    inf.add_argument("text", help="input text")
    inf.add_argument("--passes", type=int, default=21)
    inf.add_argument("--jitter", type=float, default=0.01)
    inf.add_argument("--seed-rotation", action="store_true")
    inf.add_argument("--input-size", type=int, default=64)
    inf.add_argument("--hidden1", type=int, default=128)
    inf.add_argument("--hidden2", type=int, default=64)
    inf.add_argument("--classes", type=int, default=10)
    inf.add_argument("--asic", action="store_true", help="use the cgminer ASIC backend")
    inf.add_argument("--host", default="127.0.0.1")
    inf.add_argument("--port", type=int, default=4028)
    inf.set_defaults(func=_cmd_infer)

    ew = sub.add_parser("encode-work", help="build a BM1387 (S9) work frame from a header")
    ew.add_argument("--header", default=None, help="80-byte header as hex (default: demo header)")
    ew.add_argument("--work-id", type=int, default=1)
    ew.set_defaults(func=_cmd_encode_work)

    sim = sub.add_parser("simulate",
                         help="run a virtual BM1387 chip and verify the mining pipeline")
    sim.add_argument("--header", default=None,
                     help="80-byte header as hex (default: deterministic demo header)")
    sim.add_argument("--work-id", type=int, default=1)
    sim.add_argument("--difficulty", type=int, default=16,
                     help="required leading zero bits in the hash (default 16; ~65k rolls)")
    sim.add_argument("--model", default=None,
                     help="miner model to model chip count / hashrate from (e.g. 'Antminer S9')")
    sim.add_argument("--chips", type=int, default=63,
                     help="modeled chip count when --model is not given (default 63)")
    sim.add_argument("--hashrate", type=float, default=13.5e12,
                     help="nominal H/s for the real-chip time estimate (default 13.5e12)")
    sim.add_argument("--max-nonces", type=int, default=1 << 24,
                     help="give up after this many nonce rolls (default 16,777,216)")
    sim.set_defaults(func=_cmd_simulate)

    sv = sub.add_parser("serve", help="run the on-device hasher-server")
    sv.add_argument("--model", default="Antminer S9", help="miner model to model the device on")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=HASHER_PORT,
                    help=f"listen port (default {HASHER_PORT})")
    sv.add_argument("--difficulty", type=int, default=16,
                    help="default mining difficulty in leading zero bits")
    sv.add_argument("--device", choices=("auto", "virtual", "chain"), default="auto",
                    help="device backend: auto picks the real chain if present, else virtual")
    sv.add_argument("--device-path", default="/dev/bitmain-asic",
                    help="chain device node (Linux control board only)")
    sv.set_defaults(func=_cmd_serve)

    wl = sub.add_parser("workload",
                        help="run a hash-based model split across host CPU and ASIC miner")
    wl.add_argument("text", help="input to classify")
    wl.add_argument("--model-file", default=None, help="load a saved split-model JSON file")
    wl.add_argument("--save", default=None, help="save the (new) model to this JSON path")
    wl.add_argument("--host", default="127.0.0.1", help="hasher-server host for the ASIC layer")
    wl.add_argument("--port", type=int, default=HASHER_PORT,
                    help=f"hasher-server port (default {HASHER_PORT})")
    wl.add_argument("--no-asic", action="store_true",
                    help="run the mining layer on the host CPU instead of the ASIC")
    wl.add_argument("--difficulty", type=int, default=12,
                    help="mining difficulty in leading zero bits (default 12)")
    wl.add_argument("--mining-neurons", type=int, default=4,
                    help="headers mined per inference = the ASIC workload (default 4)")
    wl.add_argument("--activation-width", type=int, default=8)
    wl.add_argument("--input-size", type=int, default=64)
    wl.add_argument("--classes", type=int, default=10)
    wl.add_argument("--nonce-range", type=int, default=1 << 18)
    wl.set_defaults(func=_cmd_workload)

    md = sub.add_parser("models", help="list or download AI files in ai_models/")
    md.add_argument("action", nargs="?", choices=("list", "download"), default="list")
    md.add_argument("name", nargs="?", default=None, help="model to download (e.g. qwen2.5-0.5b)")
    md.set_defaults(func=_cmd_models)

    ch = sub.add_parser("chat", help="chat with a local LLM, SHA-256 stages on the ASIC")
    ch.add_argument("--model", default=None, help="GGUF file name in ai_models/llm/ or a path")
    ch.add_argument("--once", default=None, help="send one message, print the reply, exit")
    ch.add_argument("--host", default="127.0.0.1", help="hasher-server host (the ASIC)")
    ch.add_argument("--port", type=int, default=HASHER_PORT,
                    help=f"hasher-server port (default {HASHER_PORT})")
    ch.add_argument("--no-asic", action="store_true", help="do the SHA-256 stages on the host")
    ch.add_argument("--no-draft", action="store_true", help="disable speculative drafts")
    ch.add_argument("--draft-ngram", type=int, default=None,
                    help="longest context n-gram the draft index matches (default 3)")
    ch.add_argument("--draft-tokens", type=int, default=None,
                    help="most tokens proposed per draft lookup (default 10)")
    ch.add_argument("--no-cache", action="store_true", help="disable the response cache")
    ch.add_argument("--seal-difficulty", type=int, default=None,
                    help="transcript seal difficulty in bits (0 disables sealing)")
    ch.add_argument("--temperature", type=float, default=None)
    ch.add_argument("--max-tokens", type=int, default=None)
    ch.add_argument("--trace", action="store_true", help="print the per-stage trace each turn")
    ch.add_argument("--verify", action="store_true",
                    help="verify every seal in ai_models/cache/transcripts.jsonl and exit")
    ch.add_argument("--stratum", action="store_true",
                    help="seal on a real miner through this host's private stratum pool")
    ch.add_argument("--wait-miner", type=float, default=60.0,
                    help="with --stratum, seconds to wait for the miner at startup")
    _add_pool_args(ch)
    ch.set_defaults(func=_cmd_chat)

    hw = sub.add_parser("hwtest",
                        help="closed-loop hardware test: private pool -> miner -> verified seals")
    hw.add_argument("--virtual", action="store_true",
                    help="dry run against a built-in software miner (no hardware)")
    hw.add_argument("--seals", type=int, default=5, help="test seals to mine (default 5)")
    hw.add_argument("--bits", type=int, default=None,
                    help="leading zero bits each seal must reach (default: what every share "
                         "at the pool's difficulty has, e.g. 40 at difficulty 256)")
    hw.add_argument("--wait", type=float, default=120.0,
                    help="seconds to wait for the miner to connect (default 120)")
    hw.add_argument("--timeout", type=float, default=30.0, help="seconds per seal (default 30)")
    _add_pool_args(hw)
    hw.set_defaults(func=_cmd_hwtest)

    bn = sub.add_parser("bench", help="compare chat speed with speculative drafts off and on")
    bn.add_argument("--model", default=None, help="GGUF file name in ai_models/llm/ or a path")
    bn.add_argument("--configs", default="off,3x10",
                    help="comma list: 'off' or NGRAMxTOKENS draft shapes (default off,3x10)")
    bn.add_argument("--max-tokens", type=int, default=128)
    bn.add_argument("--runs", type=int, default=1, help="passes over the prompt set")
    bn.add_argument("--host", default="127.0.0.1", help="hasher-server host (the ASIC)")
    bn.add_argument("--port", type=int, default=HASHER_PORT,
                    help=f"hasher-server port (default {HASHER_PORT})")
    bn.add_argument("--no-asic", action="store_true", help="do the draft hashing on the host")
    bn.set_defaults(func=_cmd_bench)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
