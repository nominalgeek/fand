# fand

Autonomous multi-fan multi-sensor thermal controller for Linux. Polls hwmon,
`nvidia-smi`, `/proc/stat`, and a NUT UPS at 2 s; drives Super-I/O PWM
channels via a *learned* per-sensor cooling-coefficient matrix and a
stress-based control law — every fan responds to whichever sensors it can
meaningfully affect, weighted by how strongly. AR(1) predictive feed-forward
anticipates heat surges from load before temps rise.

Designed for systems where the operator doesn't know which fan cools what
and would rather have the daemon figure it out from runtime data than
hand-tune curves per sensor. **See [DESIGN.md](DESIGN.md) for the project
goal, why per-zone control was the wrong abstraction, and the control law
in detail.**

Runs as a root `systemd` service. Snapshots BIOS PWM modes on startup and
restores on shutdown. Unconditional safety floors are independent of the
learned model: any sensor at/above its `critical_c` → all fans to 255 + alarm,
dead-fan detection on commanded PWM ≥ pwm_min with zero tach.

## Requirements

- Linux with `systemd`.
- A Super-I/O chip exposed via kernel hwmon. Default is `nct6799`; set
  `chip_name` in `/etc/fand/config.yaml` for other chips (the code resolves
  by the value in `/sys/class/hwmon/*/name`).
- System Python ≥ 3.12 owned by root (`install.sh` refuses to use an
  interpreter from a user homedir).
- [`uv`](https://docs.astral.sh/uv/) — used to build the wheel and the
  runtime venv. On root's PATH, or pass an explicit path:
  `sudo UV=/path/to/uv install.sh`.
- Optional: `nvidia-smi` (NVIDIA GPU temp/fan%/power/util), NUT `upsc`
  (whole-system AC draw), an `ntfy` script (push alerts). If any of these
  is missing, that feature defaults to 0 and the daemon logs a warning.

Built and tuned for: NVIDIA RTX PRO 6000 Blackwell (GPU fans are not
host-controllable on this card — fan% is treated as input only), AMD 9950X3D
(via `k10temp`/Tctl), DDR5 with `spd5118` modules, `nct6799` Super-I/O.
Should work on any Linux box with a supported Super-I/O after editing
`chip_name` and re-running `fand-calibrate`.

## Install

```bash
git clone https://github.com/nominalgeek/fand
sudo fand/install.sh
```

`install.sh` builds a wheel from the checkout, installs it into a root-owned
venv at `/usr/local/lib/fand/.venv`, writes `/etc/systemd/system/fand.service`,
seeds `/etc/fand/config.yaml`, and symlinks `fand-ctl` + `fand-calibrate` into
`/usr/local/bin/`. It does **not** start the service.

The checkout (wherever you cloned it) stays editable and is **not** where the
daemon runs from. To pick up source changes, re-run `sudo install.sh`.

## Bring up

```bash
sudo fand-calibrate                 # ~15 min total
sudo $EDITOR /etc/fand/zones.yaml   # review sensor target_c/critical_c
sudo systemctl start fand
fand-ctl status
journalctl -u fand -f
```

`fand-calibrate` runs three phases:

1. **PWM ↔ fan-tach mapping** (idle system): bounces each PWM channel and
   records which tach responds the most.
2. **`pwm_min` discovery** (idle system): walks each fan down to find its
   minimum spinning PWM.
3. **Sensor attribution** (system under typical load): drops each fan one
   at a time, measures ΔT per sensor, writes the per-fan `cools:` dict
   with seed cooling weights (in °C of ΔT).

Phases 1 and 2 want the system idle so probes aren't masked by load-driven
heat. Phase 3 *needs* heat to register — start a typical workload before
running. The whole thing takes about 15 minutes.

The output `/etc/fand/zones.yaml` is ready to use; the operator should
review the sensor `target_c`/`critical_c` defaults (which are conservative
based on hardware specs) and possibly trim the sensor catalog before
starting the service.

## Operating

| Command | Purpose |
|---|---|
| `fand-ctl status` | Per-sensor stress + per-fan PWM with driving-sensor breakdown |
| `fand-ctl tail -n 50` | Recent poll history from `history.jsonl` |
| `fand-ctl model` | Learned per-sensor coefs + cooling-coefficient matrix |
| `sudo systemctl {start,stop,restart} fand` | Service control |
| `journalctl -u fand -f` | Live daemon logs |

`fand-ctl` is read-only and needs no sudo. Live status JSON is at
`/run/fand/status.json` (world-readable) for monitoring tools.

## How it works

Every 2 s the daemon:

1. Reads every declared sensor. Applies per-sensor AR(1) feed-forward
   correction (predicts near-term ΔT from feature deltas using learned α
   coefs — disabled per-sensor while untrained).
2. Computes `stress(s) = max(0, (T - target_c) / (critical_c - target_c))`
   per sensor.
3. If any sensor is at or above its `critical_c`, drives all fans to 255
   and emits an alarm keyed by sensor name. Bypasses the rest of the loop.
4. Otherwise, per fan:
   ```
   relevance(s, p) = γ_p(s) / max_q γ_q(s)
   demand(p)      = min(1, Σ_s stress(s)^1.5 × relevance(s, p))
   PWM(p)         = pwm_min + demand × (255 − pwm_min)
   ```
   Until γ is trained, the daemon uses normalized seed weights from each
   fan's `cools:` dict (phase 3 ΔT values).
5. Detects per-fan equilibrium (fan's PWM stable + its cooled sensors at
   low dT/dt + load features stable) and appends a sample to
   `equilibria.jsonl` for the next refit.
6. Refits per-sensor coefs hourly (`fit_interval_s`), regularizing γ
   toward seed weights for low-sample-count fits and clipping γ ≥ 0
   post-solve.

See [DESIGN.md](DESIGN.md) for the matrix evolution timeline (t=0 declared
cools → t=1h initial fit → t>1d mature γ) and the configuration-vs-learned
state split.

Persistent state survives restarts:

- `/var/lib/fand/model.json` — per-sensor `baseline`, `load_coefs` (α),
  `cooling_coefs` (γ), `r²`, `n_samples`
- `/var/lib/fand/equilibria.jsonl` — equilibrium training pool, per-fan rows
- `/var/lib/fand/saved_pwm.json` — original PWM modes captured at startup
- `/var/lib/fand/history.jsonl` — per-poll observation log (gzip-rotated
  daily, 30-day retention)

## Configuration

`/etc/fand/config.yaml` — daemon tunables. The full annotated list is in
[`etc/config.yaml.example`](etc/config.yaml.example). Most-edited fields:
`chip_name`, `poll_interval_s`, `fit_interval_s`, `ridge_lambda`,
`min_samples_to_learn`, `min_r2_to_learn`, `equilibrium_pwm_stable_s`,
`ff_alpha`, `ups_name`, `ntfy_command`.

`/etc/fand/zones.yaml` — sensors + fans. Generated by `fand-calibrate`, then
operator-reviewed. Schema:

```yaml
defaults:
  pwm_min: 40

sensors:
  cpu_die:      { chip: k10temp,     label: Tctl,    target_c: 70, critical_c: 90 }
  case_ambient: { chip: nct6799,     label: AUXTIN0, target_c: 45, critical_c: 65 }
  nvidia_gpu:   { chip: gpu,         label: temp_c,  target_c: 80, critical_c: 90 }
  # ...

fans:
  pwm1:
    fan_tach: fan1_input
    cools: { case_ambient: 3.0, nvidia_gpu: 0.6 }    # ΔT °C from phase 3
  pwm4:
    fan_tach: fan4_input
    pwm_min: 50          # optional override
    cools: { cpu_die: 2.8, nvidia_gpu: 0.3 }
```

Multi-instance chips like NVMe and DDR5 SPD need explicit indexing:
`chip: "nvme[0]"`, `chip: "spd5118[1]"`. The daemon does not fuzzy-match
across instances.

If you upgrade from an older fand install with the per-zone schema, the
daemon detects it at startup, writes a translated v2 file to
`/etc/fand/zones.yaml.translated`, and exits with a `sudo mv` prompt.
After the move, re-run `sudo fand-calibrate` to replace the translator's
uniform seed weights with phase-3-measured ΔT values.

## Uninstall

```bash
sudo systemctl stop fand
sudo systemctl disable fand
sudo rm /etc/systemd/system/fand.service /usr/local/bin/fand-{ctl,calibrate}
sudo rm -rf /etc/fand /var/lib/fand /usr/local/lib/fand
sudo systemctl daemon-reload
```

## License

MIT — see [LICENSE](LICENSE).
