"""fand-ctl: read-only inspector.

Reads /run/fand/status.json (written atomically by the daemon every poll) and
formats it for humans. No daemon-control verbs in v1 — use systemctl to start /
stop the service.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

STATUS_PATH = Path("/run/fand/status.json")
HISTORY_PATH = Path("/var/lib/fand/history.jsonl")
MODEL_PATH = Path("/var/lib/fand/model.json")


def _load_status() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text())
    except FileNotFoundError:
        print(f"error: {STATUS_PATH} does not exist — is fand running?", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"error: {STATUS_PATH} corrupt: {exc}", file=sys.stderr)
        sys.exit(2)


def cmd_status(args: argparse.Namespace) -> int:
    s = _load_status()
    age = time.time() - s.get("wall_t", 0)
    print(f"fand status @ {time.strftime('%H:%M:%S', time.localtime(s['wall_t']))} "
          f"({age:.1f}s ago)")

    if s.get("any_critical"):
        print("  *** CRITICAL TEMP — ALL FANS AT 255 ***")

    if s.get("gpu"):
        g = s["gpu"]
        print(f"  GPU: {g['temp_c']:.0f}°C  fan {g['fan_pct']:.0f}%  "
              f"{g['power_w']:.0f}W  util {g['util_pct']:.0f}%")
    if s.get("cpu_util") is not None:
        print(f"  CPU util: {s['cpu_util']:.1f}%")
    if s.get("ups"):
        u = s["ups"]
        print(f"  UPS: {u['realpower_w']:.0f}W  ({u['load_pct']:.0f}% of "
              f"{u['nominal_w']:.0f}W)  batt {u['battery_pct']:.0f}%  "
              f"{'OL' if u['on_line'] else 'OB'}")

    print()
    print(f"  {'zone':<10} {'T_z':>7} {'pwm':>5} {'fan_rpm':>8} {'source':>10} "
          f"{'trained':>9} {'r²':>6} {'n':>5}")
    print(f"  {'-'*10} {'-'*7} {'-'*5} {'-'*8} {'-'*10} {'-'*9} {'-'*6} {'-'*5}")
    fans = s.get("fans", {})
    for z in s.get("zones", []):
        # Find this zone's fan_tach by matching name pattern (best-effort)
        t = z.get("T_z")
        t_str = f"{t:.1f}°C" if t is not None else "  ?  "
        pwm = z.get("pwm_applied")
        pwm_str = f"{pwm}" if pwm is not None else "?"
        r2 = z.get("r2")
        r2_str = f"{r2:.2f}" if r2 is not None else "  -  "
        rpm = "-"  # we don't know which fan goes with which zone from status alone
        print(f"  {z['name']:<10} {t_str:>7} {pwm_str:>5} {rpm:>8} "
              f"{z.get('source','?'):>10} {'yes' if z.get('trained') else 'no':>9} "
              f"{r2_str:>6} {z.get('n_samples',0):>5}")

    print()
    print("  raw fan RPMs:", fans)

    if s.get("errors"):
        print()
        print("  errors:")
        for e in s["errors"]:
            print(f"    - {e}")
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    if not HISTORY_PATH.exists():
        print(f"error: {HISTORY_PATH} does not exist", file=sys.stderr)
        return 2
    # Read the last N lines without slurping the whole file
    n = args.n
    with HISTORY_PATH.open("rb") as f:
        f.seek(0, 2)
        end = f.tell()
        block = 8192
        data = b""
        while end > 0 and data.count(b"\n") <= n:
            read = min(block, end)
            end -= read
            f.seek(end)
            data = f.read(read) + data
    lines = data.splitlines()[-n:]
    if args.compact:
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            zs = ",".join(
                f"{z['name']}={z.get('pwm_applied','?')}({z.get('source','?')[0]})"
                for z in row.get("zones", [])
            )
            gpu = row.get("gpu") or {}
            print(
                f"{time.strftime('%H:%M:%S', time.localtime(row.get('wall_t',0)))} "
                f"gpu={gpu.get('temp_c','-')}°C/{gpu.get('util_pct','-')}% "
                f"cpu_util={row.get('cpu_util','-')}% "
                f"{zs}"
            )
    else:
        for line in lines:
            sys.stdout.write(line.decode("utf-8", "replace") + "\n")
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    if not MODEL_PATH.exists():
        print(f"error: {MODEL_PATH} does not exist — fand hasn't fit yet", file=sys.stderr)
        return 2
    payload = json.loads(MODEL_PATH.read_text())
    for name, z in payload.get("zones", {}).items():
        st = z.get("state", {})
        print(f"zone {name}: target={z['target_c']}°C  critical={z['critical_c']}°C "
              f"pwm_min={z['pwm_min']}")
        coefs = st.get("coefs")
        if coefs:
            names = ["util_gpu", "power_gpu_w", "util_cpu", "ups_w", "bias"]
            print(f"  trained: n={st['n_samples']}  r²={st['r2']:.3f}")
            for n, c in zip(names, coefs):
                print(f"    {n:<14} = {c: .4f}")
        else:
            print("  trained: NO (bootstrap curve active)")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fand-ctl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("status", help="show current zone state")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("tail", help="tail the observation history")
    sp.add_argument("-n", type=int, default=20)
    sp.add_argument("--compact", action="store_true",
                    help="one-line summary instead of raw JSON")
    sp.set_defaults(func=cmd_tail)

    sp = sub.add_parser("model", help="show learned model coefficients")
    sp.set_defaults(func=cmd_model)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
