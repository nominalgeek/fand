# fand code review (Grok)

**For:** Claude Code / follow-up implementation  
**Reviewed:** `feat/initial-implementation` vs `origin/main` (merge-base `a104201`)  
**Date:** 2026-06-04  
**Scope:** 14 files, +3465 / −1 lines  

| Severity   | Count |
|------------|-------|
| bug        | 6     |
| suggestion | 5     |
| nit        | 1     |

---

## Executive summary

This branch is the full initial `fand` implementation: v2 `sensors` + `fans`/`cools` schema, per-sensor ridge learning with seed fallback, stress-based multi-fan control, equilibrium sampling, `fand-calibrate`, systemd `Type=notify`, and strong docs (`DESIGN.md`, `CLAUDE.md`, install layout). Architecture matches stated goals; actuator snapshot/restore and atomic state writes are well done.

**Verdict:** Suitable for supervised bring-up. Address or explicitly document the safety/ops gaps below before unattended production on thermally tight hardware (DDR5 `spd5118`, GPU, etc.).

**Dominant risk areas:**

1. Safety — unreadable sensors skip the critical floor; PWM writes are not verified.
2. Operations — systemd `WatchdogSec=10` vs tick latency (subprocess timeouts + inline refit).
3. Footguns — `restore_from_disk` without snapshot resets every PWM on the chip; calibration phase-3 baseline is stale across fans.

---

## Recommended fix order

1. **Issue 3** — Missing sensor reads must not disable critical protection.
2. **Issue 2** — Watchdog pings / refit off hot path / raise `WatchdogSec`.
3. **Issue 1** — Scope BIOS restore when snapshot absent (match managed channels only).
4. **Issue 4** — PWM read-back verification on safety and normal paths.
5. **Issue 5** — Per-fan baseline refresh in calibration phase 3.
6. **Issue 6** — Bound or compact `equilibria.jsonl` growth.
7. Issues 7–12 — Config validation, coverage warnings, degraded-mode behavior, holdout docs, install.sh text.

---

## What looks solid (do not regress)

- Hwmon discovery by chip **name**, not `hwmonN` index (`sensors.py`, `actuators.py`).
- PWM snapshot to `/var/lib/fand/saved_pwm.json`, manual mode 1 during run, restore on exit / `ExecStopPost`.
- Root-owned runtime install + `ExecStartPre` ownership checks (`install.sh`, `systemd/fand.service`).
- Per-sensor ridge with non-negative γ, seed `cools:` fallback, RMSE-gated `is_trained()`.
- Atomic writes for `/run/fand/status.json`, history JSONL, model persistence.
- Optional sources (nvidia-smi, upsc) fail soft; daemon continues.
- `DESIGN.md` control-law rationale; `fand-ctl` read-only inspector.

---

## Files in scope

```
CLAUDE.md
DESIGN.md
README.md
etc/config.yaml.example
fand/__init__.py
fand/actuators.py
fand/calibrate.py
fand/cli.py
fand/daemon.py
fand/model.py
fand/sensors.py
install.sh
pyproject.toml
systemd/fand.service
```

---

## Issues

### Issue 1 — bug — `restore_from_disk` without snapshot resets whole chip

- **File:** `fand/actuators.py:206`
- **Status:** fixed (`8c8c3a9`) — no-snapshot restore is now a no-op (snapshot-before-takeover invariant means there is nothing to undo), not a scoped mode-5 write; see `claude.review.reply.md`
- **Description:** When `saved_pwm.json` is missing or invalid, `restore_from_disk()` writes mode 5 on **every** `pwm*_enable` via `hw.glob("pwm*_enable")`, not only channels fand managed. Contradicts the comment at 189–191 (restore only managed channels). Can clobber BIOS curves on untouched PWM outputs — especially on `ExecStopPost` after crash before first snapshot.
- **Suggestion:** In the no-snapshot branch: do nothing, or restore only a known manifest of managed channels. Match `Actuators.restore()` scope.

```python
# fand/actuators.py ~206 — problematic else branch
else:
    for pwm_enable in sorted(hw.glob("pwm*_enable")):
        pwm_enable.write_text(str(DEFAULT_RESTORE_MODE))
```

---

### Issue 2 — bug — Watchdog vs tick latency

- **File:** `fand/daemon.py:644`, `systemd/fand.service:18`
- **Status:** fixed (`4351526`) — watchdog ping at loop level (covers exception ticks), WatchdogSec=30, staggered refit, nvidia-smi/upsc timeouts 3s/2s
- **Description:** `sd_notify("WATCHDOG=1")` only at end of `_tick()`. One tick can exceed `WatchdogSec=10`: `nvidia-smi` (5s) + `upsc` (3s) sequential in `sensors.read_all()`, plus hourly inline `SensorModel.fit` over up to 20k equilibria × all sensors. Tick exceptions (`log.exception("tick failed")` ~736) skip watchdog ping entirely.
- **Suggestion:** Ping before/after slow sections; move refit to worker thread or one sensor per tick; raise `WatchdogSec`; shorten subprocess timeouts under systemd notify.

---

### Issue 3 — bug — Missing sensor reads bypass critical floor

- **File:** `fand/daemon.py:519`
- **Status:** fixed (`98cb651`) — startup fail-fast on bad chip/label; runtime grace-hold then assume-critical (all fans max) with anti-flap recovery; per-sensor `on_unreadable` override; held temps excluded from learning
- **Description:** If `_temp_at()` returns `None`, sensor is omitted from `sensor_temps` / `sensor_effective_temps` and never checked against `critical_c` (530–539). Failed hwmon, `nvidia-smi` outage, or bad `chip`/`label` disables stress control **and** critical backstop for that sensor while daemon continues.
- **Suggestion:** After N consecutive failures: assume `T >= critical_c`, or `set_all(255)` with uncooled alarm. Fail fast at startup if configured paths are absent.

```python
# fand/daemon.py ~519 — current behavior
if T is None:
    self._alarm(f"missing:{name}", ...)
    continue  # sensor excluded from critical check
```

---

### Issue 4 — bug — PWM writes not verified

- **File:** `fand/daemon.py:552`
- **Status:** fixed (`3379c71`) — set_all returns failed channels (critical path re-takes-over + retries + alarms), per-fan write-failure streak, plus enable-drift detection from obs.pwm_enable
- **Description:** `set_all(255)` and `set_pwm()` ignore `Actuators.set_pwm()` return value. Partial sysfs failure (EBUSY, read-back mismatch) leaves `status.json` at commanded PWM while hardware is lower. Fan-dead detection uses commanded value, not `obs.pwm` read-back.
- **Suggestion:** Verify read-back after writes; high-priority alarm on mismatch; retry `take_over` on failure.

---

### Issue 5 — bug — Calibration phase-3 stale baseline

- **File:** `fand/calibrate.py:259`
- **Status:** fixed (`cce4e82`) — baseline re-sampled immediately before each fan's drop; needs an idle recalibration run to regenerate seeds
- **Description:** Single global `baseline = _average_temps(...)` before per-fan loop; never refreshed. Each fan perturbation heats chassis; later fans compared to pre-run baseline → inflated ΔT and wrong seed `cools:` until runtime learning corrects.
- **Suggestion:** Re-sample baseline after each fan's recovery window, or rolling pre-drop average immediately before each throttle step.

```python
# fand/calibrate.py ~259
baseline = _average_temps(sensors_obj)
for target in managed:
    ...
    post = _average_temps(sensors_obj)
    base_val = baseline.get(key)  # stale for fan 2+
```

---

### Issue 6 — bug — `equilibria.jsonl` unbounded growth

- **File:** `fand/daemon.py:478`
- **Status:** partial-by-design (`f95d3f0`, `4351526`) — severity was overstated (normal-regime growth is bounded by the per-refit rewrite); the one genuinely unbounded path was the watchdog kill-loop, closed by the staggered refit; append-counter compaction (every 2000) covers misconfigured fit_interval_s
- **Description:** `_append_equilibrium()` appends every sample to disk; in-memory deque capped at 20k (`EQUILIBRIA_MAXLEN`). File trimmed only on refit (`fit_interval_s`, default 1h). Crash or long interval → large JSONL; `_load_equilibria()` parses every line on startup before deque trim.
- **Suggestion:** Compact on append count, periodic rewrite, or append only via deque with bounded file.

---

### Issue 7 — suggestion — No `target_c < critical_c` validation

- **File:** `fand/daemon.py:363`, `fand/model.py:175`
- **Status:** fixed (`f95d3f0`) — daemon init rejects critical_c ≤ target_c, pwm_min outside [0,255], negative cools weights; not mirrored in calibrate (its generated defaults are valid by construction)
- **Description:** Invalid `critical_c <= target_c` loads silently; `stress()` uses `span = max(0.1, critical - target)` → misleading stress curve.
- **Suggestion:** Reject at daemon init; mirror in `fand-calibrate` output.

---

### Issue 8 — suggestion — Sensors not in any fan `cools`

- **File:** `fand/daemon.py:558`
- **Status:** partial-by-design (`f95d3f0`) — startup warning lists uncovered sensors; auto-linking rejected (an ambient probe wired to every fan would pollute the learned cooling map; uncovered sensors legitimately keep monitoring + critical floor only)
- **Description:** `aggregate_demand()` only uses sensors in each fan's `cools`. Declared `sensors:` with no `cools` reference show stress in status but never raise PWM until `critical_c`.
- **Suggestion:** Startup warning for uncovered sensors; optional default seed or all-fan linkage.

---

### Issue 9 — suggestion — Inline refit blocks control loop

- **File:** `fand/model.py:620`
- **Status:** fixed (`4351526`) — refit staggered one sensor per tick against a shared pool snapshot; worker thread rejected (fit() writes SensorModel.state non-atomically under readers)
- **Description:** All sensors `fit()` synchronously in main poll on refit interval. Large equilibrium sets stall ticks (interacts with Issue 2).
- **Suggestion:** Background thread with sample snapshot, or one sensor per tick.

---

### Issue 10 — suggestion — Silent degradation on tick exceptions

- **File:** `fand/daemon.py:736`
- **Status:** fixed (`4351526`) — 5 consecutive tick failures → sd_notify STATUS + exit 1 → Restart=on-failure; persistent failure leaves the unit failed with BIOS modes restored
- **Description:** Bare `except Exception` logs and continues; stale PWM, no learning, service still "active".
- **Suggestion:** Consecutive-failure counter → `sd_notify(STATUS=...)` or exit for `Restart=on-failure`; re-raise on safety-path errors.

---

### Issue 11 — suggestion — Chronological holdout for `is_trained()`

- **File:** `fand/model.py:278`
- **Status:** rejected (`e05ab3d`) — random/blocked holdout would leak time-adjacent near-duplicate equilibria into the test set and open the trained gate on overfit models; chronological errs pessimistic → fails safe onto seeds. Rationale documented at the split site and in config.yaml.example; full argument in `claude.review.reply.md`
- **Description:** First 80% train / last 20% test on time-ordered equilibria. Regime shift makes RMSE gate optimistic or pessimistic.
- **Suggestion:** Blocked/random holdout or rolling CV; document `rmse` is regime-dependent.

---

### Issue 12 — nit — `install.sh` v1 wording

- **File:** `install.sh:124`
- **Status:** fixed (`e05ab3d`) — v2 sensors+fans/cools wording; calibration duration corrected to ~15 min
- **Description:** Post-install heredoc mentions `target_sensors` (v1); codebase uses v2 `sensors` + `fans`/`cools`.
- **Suggestion:** Align with README bring-up: calibrate → edit `zones.yaml` → start.

---

## Context for implementers

- **Safety contract** (`DESIGN.md`): `critical_c` is an unconditional floor; sensor reader failures must not weaken it.
- **Hardware** (`CLAUDE.md`): nct6799, 6 PWM / 5 tach; GPU fans not host-writable; DDR5 thermally sensitive.
- **Runtime paths:** config `/etc/fand/`, state `/var/lib/fand/`, status `/run/fand/status.json`.
- **Tests:** No automated test suite observed in branch; manual verification via `fand-ctl status`, `journalctl -u fand`, supervised calibration.

---

## Claude Code usage

When fixing from this review:

1. Pick issues by **Recommended fix order** unless the user specifies otherwise.
2. Read `DESIGN.md` before changing `model.py` control law or learning gates.
3. After daemon/safety changes: `sudo systemctl restart fand` only on a test host; verify `WatchdogSec` with `systemd-analyze critical-chain fand.service`.
4. Calibration changes require a full idle `sudo fand-calibrate` run to validate phase-3 weights.
5. Mark issues resolved in this file (`Status: fixed` + PR/commit ref) if continuing across sessions.

---

*Generated by Grok branch review (`feat/initial-implementation` vs `origin/main`).*