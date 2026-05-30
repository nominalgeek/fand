# CLAUDE.md

## What this is

`fand` is an **autonomous multi-fan multi-sensor thermal controller** for Linux hosts using the nct6799 Super-I/O chip. Runs as a root systemd service. Polls hwmon temps + nvidia-smi (GPU temp / fan% / power / util) + `/proc/stat` (CPU util) + NUT `upsc` (whole-system AC draw) at 2 s. Every fan responds to stresses across every sensor it can affect, weighted by a learned cooling-coefficient matrix. AR(1) predictive feed-forward anticipates heat surges before temps rise.

**Project goal & architecture rationale: see [DESIGN.md](DESIGN.md).** The TL;DR is that fand is for systems where the operator doesn't know which fan cools what — declare sensors and fans, let the daemon learn the relationships from runtime equilibrium samples.

This was extracted from `/opt/comfyui/fand` into its own repo (`github.com:nominalgeek/fand.git`) on 2026-05-27. The `/opt/comfyui` workspace it came out of is the ML host that motivated this — see notes about the hardware context below.

## Architecture (one-line per component)

- `fand/sensors.py` — hwmon by chip name (NOT index — kernel updates shift indices), `nvidia-smi`, `/proc/stat`, `upsc`. All readers tolerate missing sources.
- `fand/actuators.py` — PWM writer with snapshot/restore. Captures original `pwm*_enable` to `/var/lib/fand/saved_pwm.json` on startup, flips to mode 1 (manual), restores on exit. `--restore-bios` falls back to mode 5 if snapshot missing.
- `fand/model.py` — `SensorModel` (one per declared sensor): per-sensor ridge regression `T_s ≈ baseline + Σ α·feature − Σ γ·pwm` on equilibrium samples. Helpers: `aggregate_demand` (weighted-stress control law), `is_fan_equilibrium` (per-fan steady-state detector). AR(1) feed-forward per sensor (disabled while α untrained).
- `fand/daemon.py` — main loop, schema parser + auto-translator for old `zones:` format, `sd_notify` watchdog, unconditional safety floors, status JSON + history JSONL (gzip rotated daily).
- `fand/calibrate.py` — operator-supervised PWM perturbation sweep → discovers PWM↔fan-tach mapping + `pwm_min` per channel + phase-3 ΔT seed weights → writes new-schema `/etc/fand/zones.yaml`.
- `fand/cli.py` — `fand-ctl status / tail / model` — read-only, any user, no sudo. v2 status display: sensor table + fan table with stress/relevance breakdown.
- `systemd/fand.service` — `Type=notify`, `WatchdogSec=10`, `User=root`, narrow `ReadWritePaths=`, `ExecStopPost` restores BIOS modes.

## Install layout (dev checkout vs runtime)

Two locations, on purpose:

- `/opt/fand` — dev checkout. User-owned and editable. **Not** where the daemon runs from.
- `/usr/local/lib/fand` — runtime. Root-owned. `install.sh` builds a wheel from the dev checkout (`uv build`) and installs it into `.venv/` there as root via `uv pip install`. The wheel's entry points (`fand-daemon`, `fand-ctl`, `fand-calibrate`) land in `.venv/bin/`; `/usr/local/bin/fand-{ctl,calibrate}` are symlinks to them. systemd's `ExecStart` is `.venv/bin/fand-daemon`.

The daemon runs as `User=root`; if its code lived in a user-writable path, any account that can write the dev checkout could inject Python that executes as root on the next service restart. `install.sh` refuses to install if `.venv/bin/python` resolves to a non-root-owned interpreter, and the unit has `ExecStartPre=/usr/bin/test -O …` guards on `fand-daemon` and `.venv/bin/python` as belt-and-braces.

To pick up source changes: edit in `/opt/fand`, then `sudo /opt/fand/install.sh` to re-sync.

## Hardware context (load-bearing for the box this was built on)

- **GPU**: NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM. Onboard GPU fans are **not host-controllable** — `nvidia-smi` reads but cannot write fan speed on this card. The Blackwell board fan logic owns them. Treat GPU fan % as an input signal only.
- **CPU**: AMD Ryzen 9 9950X3D — read via `k10temp` (`Tctl`).
- **DDR5**: two `spd5118` modules with `HIGH=55 °C` alarm. RAM was the thermally borderline component in baseline — give it a real target (not just a critical floor).
- **Super-I/O**: nct6799 — 6 PWM channels (`pwm1-5`, `pwm7`), 5 fan tachs. PWM enable modes: `0`=full / `1`=manual / `2`=thermal cruise / `5`=BIOS SmartFan (Linux default).

If a different host has a different Super-I/O chip, code finds it by name (`find_hwmon_by_name`), so the chip name in `zones.yaml` + `actuators.Actuators(chip_name=...)` is the only place to change.

## Bring-up sequence

```
sudo /opt/fand/install.sh
sudo fand-calibrate                 # ~15 min total: phases 1+2 idle, phase 3 under load
sudo $EDITOR /etc/fand/zones.yaml   # review sensor target_c/critical_c; trim sensors:
sudo systemctl start fand
fand-ctl status
journalctl -u fand -f
```

Phase 3 of calibrate writes seed cooling weights (ΔT °C per sensor) into each fan's `cools:` dict. The daemon uses those until enough equilibrium samples accumulate to fit per-sensor `γ_p(s)` coefficients (~hours to a day under typical load).

## Common operations

| Action | Command |
|---|---|
| Build dev venv (for interactive use; runtime venv at `/usr/local/lib/fand/.venv` is built by `install.sh`) | `cd /opt/fand && uv venv .venv --python python3.13 && uv pip install --python .venv/bin/python -e .` |
| Build wheel only (no install) | `cd /opt/fand && uv build --wheel` |
| Live sensor read (no privileges needed) | `.venv/bin/python -c "from fand.sensors import Sensors; import time; s=Sensors(); s.read_all(); time.sleep(0.3); print(s.read_all())"` |
| Run daemon directly (debug, not via systemd) | `sudo .venv/bin/python -m fand.daemon` |
| Restore BIOS fan modes manually | `sudo .venv/bin/python -m fand.daemon --restore-bios` |
| Calibrate (root, ~15 min: idle for phases 1+2, load for phase 3) | `sudo fand-calibrate` |
| Service control | `sudo systemctl {start,stop,restart,status} fand` |
| Live status | `fand-ctl status` |
| Recent observation history | `fand-ctl tail --compact -n 50` |
| Learned model coefficients | `fand-ctl model` |
| Inspect raw hwmon by chip | `for h in /sys/class/hwmon/hwmon*; do echo "=== $(cat $h/name)"; ls $h; done` |
| Force restore + uninstall | See bottom of `install.sh` |

## Sandbox / sudo gotchas (Claude Code on this host)

- `uv venv` and `uv pip install` write to `~/.cache/uv` which is denied in the default sandbox — use `dangerouslyDisableSandbox: true` on those Bash calls.
- `nvidia-smi` and other GPU/system commands frequently fail in the sandbox with permission errors — same bypass.
- `ps` / `lsof` / `pgrep` in the sandbox run in a PID namespace that can't see user-started processes; **don't conclude a process is gone** from a sandboxed `ps` — re-run with sandbox off.
- All hwmon writes and `systemctl` start/stop need sudo. Reading sensors, sysfs, `fand-ctl status`, `upsc cyberpower`, and `journalctl -u fand` all work as regular user.
- Calibration spins every connected fan up and down individually for ~5 minutes — only run when system is fully idle and you're at the desk to hear what's ramping.

## Conventions

- Find hwmon devices by chip **name**, never by `hwmon3`-style index. Indices shift on kernel updates.
- Multi-instance chips (`nvme[0]`, `spd5118[1]`) require explicit indexing in `zones.yaml`'s sensor declarations — no fuzzy fallback.
- Sensor functions return `None` on failure; the daemon's safety floors (`set_all(255)` on any sensor crossing `critical_c`) are the backstop. Never raise from a sensor reader.
- Per-sensor learned state is persistent (`/var/lib/fand/model.json` — `cooling_coefs`, `load_coefs`, `baseline`, `r²`, `n_samples` per sensor) and survives daemon restarts. `equilibria.jsonl` holds the rolling training pool. `history.jsonl` rotates daily, 30 d retention.
- The daemon writes `/run/fand/status.json` every poll atomically (`tempfile` + `os.replace`). Any monitor (Prometheus exporter, dashboard) should scrape that, not the daemon directly.
- Sensor `target_c` and `critical_c` are operator-set in `zones.yaml`, **not** hardcoded — defaults are conservative but each rig wants tuning. Same for each fan's `cools:` list (initial seed weights come from `fand-calibrate`; operator can edit).
- UPS feature, NUT loopback, nvidia-smi are all **optional**. If any is missing, the daemon runs without it (load feature defaults to 0; warning logged). `Wants=` not `Requires=` in the systemd unit.

## Key files

| File | Role |
|---|---|
| `DESIGN.md` | Project goal, architectural rationale, control law derivation, learning timeline. Read this before touching `model.py`. |
| `fand/sensors.py` | Sensor poll surface; everything that reads hardware lives here. |
| `fand/actuators.py` | Everything that writes PWM lives here. Context-managed snapshot/restore. |
| `fand/model.py` | Math. Per-sensor ridge regression with non-negative `γ` cooling coefs, weighted-stress control law (`aggregate_demand`), AR(1) FF, per-fan equilibrium detection. numpy only. |
| `fand/daemon.py` | Glue + schema parsing + auto-translator + safety + persistence + sd_notify. Most behavior lives here, not in `model.py`. |
| `fand/calibrate.py` | One-shot PWM perturbation sweep + phase-3 attribution. Writes v2-schema `zones.yaml`. Operator runs once at install time. |
| `fand/cli.py` | `fand-ctl` — read-only inspector. |
| `etc/config.yaml.example` | Operator-editable tunables. Installed to `/etc/fand/config.yaml`. |
| `systemd/fand.service` | Unit template installed to `/etc/systemd/system/`. |
| `pyproject.toml` | Package metadata + deps + entry points (`fand-daemon`, `fand-ctl`, `fand-calibrate`). What `uv build` consumes. |
| `install.sh` | Root installer. Builds a wheel (`uv build`), installs it into a root-owned venv at `/usr/local/lib/fand/.venv`, writes the systemd unit, seeds `/etc/fand`, symlinks CLI shims. Does NOT start service. |

## State (outside the repo)

| Path | Owner | Purpose |
|---|---|---|
| `/etc/fand/config.yaml` | operator | Tunables (poll interval, thresholds, ff_alpha, ntfy hook) |
| `/etc/fand/zones.yaml` | calibrator → operator | v2 schema: `defaults`, `sensors:` catalog with target/critical, `fans:` dict with `cools:` seed weights |
| `/var/lib/fand/model.json` | daemon | v2: per-sensor `baseline`, `load_coefs` (α), `cooling_coefs` (γ), `r²`, `n_samples` |
| `/var/lib/fand/equilibria.jsonl` | daemon | v2: per-fan equilibrium rows with feature/PWM/temp snapshots |
| `/var/lib/fand/history.jsonl{,.YYYYMMDD.gz}` | daemon | Per-poll observation log (status JSON v2) |
| `/var/lib/fand/saved_pwm.json` | daemon | Original PWM enable modes (for restore) |
| `/run/fand/status.json` | daemon | Live atomic-write status v2 (world-readable) |
