# CLAUDE.md

## What this is

`fand` is an **adaptive system fan controller** for Linux hosts using the nct6799 Super-I/O chip. Runs as a root systemd service. Polls hwmon temps + nvidia-smi (GPU temp / fan% / power / util) + `/proc/stat` (CPU util) + NUT `upsc` (whole-system AC draw) at 2 s and drives the chip's PWM channels via a per-zone ridge-regression model on equilibrium samples plus AR(1) predictive feed-forward (anticipates heat surges from load before temps rise).

This was extracted from `/opt/comfyui/fand` into its own repo (`github.com:nominalgeek/fand.git`) on 2026-05-27. The `/opt/comfyui` workspace it came out of is the ML host that motivated this — see notes about the hardware context below.

## Architecture (one-line per component)

- `fand/sensors.py` — hwmon by chip name (NOT index — kernel updates shift indices), `nvidia-smi`, `/proc/stat`, `upsc`. All readers tolerate missing sources.
- `fand/actuators.py` — PWM writer with snapshot/restore. Captures original `pwm*_enable` to `/var/lib/fand/saved_pwm.json` on startup, flips to mode 1 (manual), restores on exit. `--restore-bios` falls back to mode 5 if snapshot missing.
- `fand/model.py` — per-zone ridge regression on equilibrium samples (form: `T_z - T_amb = (load·θ)/max(PWM - PWM₀, 1)`) + AR(1) EMA-based surge bump. Bootstrap curves used until ≥200 equilibrium samples AND R² ≥ 0.7.
- `fand/daemon.py` — main loop, `sd_notify` watchdog, unconditional safety floors, status JSON + history JSONL (gzip rotated daily).
- `fand/calibrate.py` — operator-supervised PWM perturbation sweep → discovers PWM↔fan-tach mapping + `pwm_min` per channel → template `/etc/fand/zones.yaml`.
- `fand/cli.py` — `fand-ctl status / tail / model` — read-only, any user, no sudo.
- `systemd/fand.service` — `Type=notify`, `WatchdogSec=10`, `User=root`, narrow `ReadWritePaths=`, `ExecStopPost` restores BIOS modes.

## Hardware context (load-bearing for the box this was built on)

- **GPU**: NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM. Onboard GPU fans are **not host-controllable** — `nvidia-smi` reads but cannot write fan speed on this card. The Blackwell board fan logic owns them. Treat GPU fan % as an input signal only.
- **CPU**: AMD Ryzen 9 9950X3D — read via `k10temp` (`Tctl`).
- **DDR5**: two `spd5118` modules with `HIGH=55 °C` alarm. RAM was the thermally borderline component in baseline — give it a real target (not just a critical floor).
- **Super-I/O**: nct6799 — 6 PWM channels (`pwm1-5`, `pwm7`), 5 fan tachs. PWM enable modes: `0`=full / `1`=manual / `2`=thermal cruise / `5`=BIOS SmartFan (Linux default).

If a different host has a different Super-I/O chip, code finds it by name (`find_hwmon_by_name`), so the chip name in `zones.yaml` + `actuators.Actuators(chip_name=...)` is the only place to change.

## Bring-up sequence

```
sudo /opt/fand/install.sh
sudo fand-calibrate                 # ~5 min, system MUST be idle for clean attribution
sudo $EDITOR /etc/fand/zones.yaml   # set target_sensors per zone
sudo systemctl start fand
fand-ctl status
journalctl -u fand -f
```

## Common operations

| Action | Command |
|---|---|
| Build venv | `cd /opt/fand && uv venv .venv --python python3.12 && uv pip install --python .venv/bin/python -r requirements.txt` |
| Live sensor read (no privileges needed) | `.venv/bin/python -c "from fand.sensors import Sensors; import time; s=Sensors(); s.read_all(); time.sleep(0.3); print(s.read_all())"` |
| Run daemon directly (debug, not via systemd) | `sudo .venv/bin/python -m fand.daemon` |
| Restore BIOS fan modes manually | `sudo .venv/bin/python -m fand.daemon --restore-bios` |
| Calibrate (root, system idle, ~5 min) | `sudo fand-calibrate` |
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
- Sensor functions return `None` on failure; the daemon's safety floors (`set_all(255)` on critical, per-zone 255 on sensor fault) are the backstop. Never raise from a sensor reader.
- Per-zone state is persistent (`/var/lib/fand/model.json`, `equilibria.jsonl`) and survives daemon restarts. `history.jsonl` rotates daily, 30 d retention.
- The daemon writes `/run/fand/status.json` every poll atomically (`tempfile` + `os.replace`). Any monitor (Prometheus exporter, dashboard) should scrape that, not the daemon directly.
- Critical/target temps are operator-set in `zones.yaml`, **not** hardcoded — there are no good universal defaults across cases / cooler choices.
- UPS feature, NUT loopback, nvidia-smi are all **optional**. If any is missing, the daemon runs without it (load feature defaults to 0; warning logged). `Wants=` not `Requires=` in the systemd unit.

## Key files

| File | Role |
|---|---|
| `fand/sensors.py` | Sensor poll surface; everything that reads hardware lives here. |
| `fand/actuators.py` | Everything that writes PWM lives here. Context-managed snapshot/restore. |
| `fand/model.py` | Math. Per-zone ridge regression + AR(1) feed-forward. numpy only. |
| `fand/daemon.py` | Glue + safety + persistence + sd_notify. Most behavior lives here, not in `model.py`. |
| `fand/calibrate.py` | One-shot PWM perturbation sweep. Operator runs once at install time. |
| `fand/cli.py` | `fand-ctl` — read-only inspector. |
| `etc/config.yaml.example` | Operator-editable tunables. Installed to `/etc/fand/config.yaml`. |
| `systemd/fand.service` | Unit template installed to `/etc/systemd/system/`. |
| `install.sh` | Root installer. Builds venv, installs unit + CLI shims, creates dirs. Does NOT start service. |

## State (outside the repo)

| Path | Owner | Purpose |
|---|---|---|
| `/etc/fand/config.yaml` | operator | Tunables (poll interval, thresholds, ff_alpha, ntfy hook) |
| `/etc/fand/zones.yaml` | calibrator → operator | Zone definitions: PWM channel, fan tach, target sensors, bootstrap curve |
| `/var/lib/fand/model.json` | daemon | Learned coefficients per zone |
| `/var/lib/fand/equilibria.jsonl` | daemon | Persisted equilibrium samples (survive restarts) |
| `/var/lib/fand/history.jsonl{,.YYYYMMDD.gz}` | daemon | Per-poll observation log |
| `/var/lib/fand/saved_pwm.json` | daemon | Original PWM enable modes (for restore) |
| `/run/fand/status.json` | daemon | Live atomic-write status (world-readable) |
