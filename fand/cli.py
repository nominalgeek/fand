"""fand-ctl: read-only inspector.

Reads /run/fand/status.json (written atomically by the daemon every poll) and
formats it for humans. v2 schema — sensors + fans, learned coefficient
matrix. No daemon-control verbs; use systemctl to start/stop the service.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

STATUS_PATH = Path("/run/fand/status.json")
HISTORY_PATH = Path("/var/lib/fand/history.jsonl")
MODEL_PATH = Path("/var/lib/fand/model.json")


def _load_status() -> dict[str, Any]:
    try:
        s = json.loads(STATUS_PATH.read_text())
    except FileNotFoundError:
        print(f"error: {STATUS_PATH} does not exist — is fand running?", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"error: {STATUS_PATH} corrupt: {exc}", file=sys.stderr)
        sys.exit(2)
    if s.get("version") != 2:
        print(
            f"error: status.json version {s.get('version')} is not v2 — "
            "daemon needs restart after schema refactor",
            file=sys.stderr,
        )
        sys.exit(2)
    return s


def _fmt_T(T: float | None) -> str:
    return f"{T:.1f}°C" if T is not None else "  ?  "


def _fmt_pwm(pwm: int | None) -> str:
    return f"{pwm}" if pwm is not None else "?"


def _fmt_pct(stress: float | None) -> str:
    return f"{stress:.2f}" if stress is not None else "  -  "


def _fmt_r2(r2: float | None) -> str:
    return f"{r2:.2f}" if r2 is not None else "  -  "


def cmd_status(_args: argparse.Namespace) -> int:
    s = _load_status()
    age = time.time() - s.get("wall_t", 0)
    wall_t = s.get("wall_t", 0)
    print(
        f"fand status @ {time.strftime('%H:%M:%S', time.localtime(wall_t))} "
        f"({age:.1f}s ago)"
    )

    if s.get("any_critical"):
        print("  *** CRITICAL — ALL FANS AT 255 ***")

    if s.get("gpu"):
        g = s["gpu"]
        print(
            f"  GPU: {g['temp_c']:.0f}°C  fan {g['fan_pct']:.0f}%  "
            f"{g['power_w']:.0f}W  util {g['util_pct']:.0f}%"
        )
    if s.get("cpu_util") is not None:
        print(f"  CPU util: {s['cpu_util']:.1f}%")
    if s.get("ups"):
        u = s["ups"]
        print(
            f"  UPS: {u['realpower_w']:.0f}W  ({u['load_pct']:.0f}% of "
            f"{u['nominal_w']:.0f}W)  batt {u['battery_pct']:.0f}%  "
            f"{'OL' if u['on_line'] else 'OB'}"
        )

    # ---- Sensors table ----
    sensors = s.get("sensors") or []
    if sensors:
        print()
        print("  Sensors:")
        print(
            f"    {'name':<22} {'T':>7} {'target':>7} {'crit':>5} "
            f"{'stress':>6} {'trnd':>4} {'n':>5} {'rmse':>5} {'r²':>5}"
        )
        print(
            f"    {'-'*22} {'-'*7} {'-'*7} {'-'*5} {'-'*6} {'-'*4} {'-'*5} "
            f"{'-'*5} {'-'*5}"
        )
        # Sort by stress descending so the hot stuff is at the top
        sorted_sensors = sorted(sensors, key=lambda x: -(x.get("stress") or 0))
        for sn in sorted_sensors:
            stress = sn.get("stress")
            stress_str = _fmt_pct(stress)
            marker = " ←" if stress is not None and stress > 0.5 else ""
            # Sensor-read fail-safe state (daemon holds/escalates unreadable
            # sensors; see DESIGN.md "Safety properties").
            read_state = sn.get("read_state")
            if read_state and read_state != "ok":
                marker += f" [{read_state.upper()}]"
            print(
                f"    {sn['name']:<22} {_fmt_T(sn.get('T')):>7} "
                f"{sn.get('target_c', 0):>5.0f}°C "
                f"{sn.get('critical_c', 0):>3.0f}°C "
                f"{stress_str:>6} "
                f"{'yes' if sn.get('trained') else 'no':>4} "
                f"{sn.get('n_samples', 0):>5} "
                f"{_fmt_r2(sn.get('rmse')):>5} "
                f"{_fmt_r2(sn.get('r2')):>5}{marker}"
            )

    # ---- Fans table ----
    fans_state = s.get("fans_state") or []
    if fans_state:
        print()
        print("  Fans:")
        print(
            f"    {'name':<6} {'PWM':>4} {'RPM':>5} {'source':>16}  "
            f"driving sensors (top 3)"
        )
        print(f"    {'-'*6} {'-'*4} {'-'*5} {'-'*16}  {'-'*40}")
        for fan in fans_state:
            rpm = fan.get("rpm")
            rpm_str = f"{rpm}" if rpm is not None else "?"
            driving = fan.get("driving_sensors") or []
            if driving:
                drv_str = "  ".join(
                    f"{d['sensor']}(stress {d['stress']:.2f}×rel {d['relevance']:.2f})"
                    for d in driving
                )
            else:
                drv_str = "(no sensor stress)"
            print(
                f"    {fan['name']:<6} {_fmt_pwm(fan.get('pwm_applied')):>4} "
                f"{rpm_str:>5} {fan.get('source', '?'):>16}  {drv_str}"
            )

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
        skipped_v1 = 0
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("version") != 2:
                skipped_v1 += 1
                continue
            wall_t = row.get("wall_t", 0)
            gpu = row.get("gpu") or {}
            sensors = row.get("sensors") or []
            fans = row.get("fans_state") or []
            # Top-3 stressed sensors
            top = sorted(sensors, key=lambda x: -(x.get("stress") or 0))[:3]
            top_str = "  ".join(
                f"{x['name']}={x.get('T', '-')}°C({(x.get('stress') or 0):.2f})"
                for x in top if x.get("T") is not None
            )
            fan_str = "  ".join(
                f"{f['name']}={_fmt_pwm(f.get('pwm_applied'))}"
                for f in fans
            )
            print(
                f"{time.strftime('%H:%M:%S', time.localtime(wall_t))} "
                f"gpu={gpu.get('temp_c', '-')}°C/{gpu.get('util_pct', '-')}%  "
                f"sensors[ {top_str} ]  fans[ {fan_str} ]"
            )
        if skipped_v1:
            print(f"({skipped_v1} pre-v2 rows skipped)", file=sys.stderr)
    else:
        for line in lines:
            sys.stdout.write(line.decode("utf-8", "replace") + "\n")
    return 0


def cmd_model(_args: argparse.Namespace) -> int:
    if not MODEL_PATH.exists():
        print(f"error: {MODEL_PATH} does not exist — fand hasn't fit yet", file=sys.stderr)
        return 2
    payload = json.loads(MODEL_PATH.read_text())
    if payload.get("version") != 2:
        print(
            f"error: model.json version {payload.get('version')} is not v2 — "
            "daemon needs to refit (~1 hour after start)",
            file=sys.stderr,
        )
        return 2

    sensors_dict = payload.get("sensors") or {}
    if not sensors_dict:
        print("(no sensors learned yet)")
        return 0

    # ---- Per-sensor coefs ----
    print("Per-sensor models:")
    print(
        f"  {'sensor':<22} {'target':>6} {'crit':>4} {'n':>5} {'rmse':>5} "
        f"{'r²':>5} {'baseline':>9}"
    )
    print(f"  {'-'*22} {'-'*6} {'-'*4} {'-'*5} {'-'*5} {'-'*5} {'-'*9}")
    fan_names: set[str] = set()
    for name, sd in sorted(sensors_dict.items()):
        st = sd.get("state", {})
        target = sd.get("target_c", 0)
        crit = sd.get("critical_c", 0)
        n = st.get("n_samples", 0)
        r2 = st.get("r2")
        rmse = st.get("rmse")
        baseline = st.get("baseline")
        baseline_str = f"{baseline:.2f}" if baseline is not None else "-"
        print(
            f"  {name:<22} {target:>4.0f}°C {crit:>2.0f}°C {n:>5} "
            f"{_fmt_r2(rmse):>5} {_fmt_r2(r2):>5} {baseline_str:>9}"
        )
        fan_names.update((st.get("cooling_coefs") or {}).keys())

    # ---- Cooling matrix ----
    if fan_names:
        fan_list = sorted(fan_names)
        print()
        print("Cooling matrix γ_p(s) (rows = sensors, cols = fans):")
        # Header
        hdr_fans = "  ".join(f"{fn:>8}" for fn in fan_list)
        print(f"  {'sensor':<22} {hdr_fans}")
        print(f"  {'-'*22} {'-'*(10*len(fan_list))}")
        for name, sd in sorted(sensors_dict.items()):
            coefs = (sd.get("state", {}).get("cooling_coefs") or {})
            cells = "  ".join(
                f"{coefs.get(fn, 0):>8.4f}" if fn in coefs else f"{'-':>8}"
                for fn in fan_list
            )
            print(f"  {name:<22} {cells}")

    # ---- Load coefs ----
    print()
    print("Load-feature coefs α_k(s):")
    feature_names_set: set[str] = set()
    for sd in sensors_dict.values():
        feature_names_set.update((sd.get("state", {}).get("load_coefs") or {}).keys())
    feature_names_list = sorted(feature_names_set)
    if feature_names_list:
        hdr = "  ".join(f"{fn:>12}" for fn in feature_names_list)
        print(f"  {'sensor':<22} {hdr}")
        print(f"  {'-'*22} {'-'*(14*len(feature_names_list))}")
        for name, sd in sorted(sensors_dict.items()):
            coefs = (sd.get("state", {}).get("load_coefs") or {})
            cells = "  ".join(
                f"{coefs.get(fn, 0):>12.5f}" if fn in coefs else f"{'-':>12}"
                for fn in feature_names_list
            )
            print(f"  {name:<22} {cells}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fand-ctl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("status", help="show current sensor + fan state")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("tail", help="tail the observation history")
    sp.add_argument("-n", type=int, default=20)
    sp.add_argument("--compact", action="store_true",
                    help="one-line summary instead of raw JSON")
    sp.set_defaults(func=cmd_tail)

    sp = sub.add_parser("model", help="show learned per-sensor coefficients + cooling matrix")
    sp.set_defaults(func=cmd_model)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
