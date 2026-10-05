"""AI-ASIC command-line interface.

Cross-platform (works on Windows via `py -m ai_asic.cli` or the `ai-asic` console script).

Commands:
  detect       Detect the attached ASIC miner and print its profile/capabilities.
  profiles     List all supported miner models.
  infer        Run recursive inference on an input string.
  encode-work  Build a BM1387 (S9) work frame from an 80-byte header (hex).
"""
from __future__ import annotations

import argparse
import sys

from ai_asic import __version__
from ai_asic.hardware import bm1387
from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.hardware.device_detector import detect_asic, detection_summary
from ai_asic.hardware.miner_profiles import all_profiles, detect_profile
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

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
