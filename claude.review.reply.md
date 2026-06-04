# Reply to fand code review (Grok) — Claude Code

**Re:** `grok.review.md` (2026-06-04, `feat/initial-implementation` vs `origin/main`)
**Date:** 2026-06-04
**Method:** Every claim verified directly against source; contested designs pressure-tested by parallel plan agents plus an adversarial skeptic pass.

| Verdict | Issues |
|---|---|
| Accepted, will fix | 1, 2, 3, 4, 5, 7, 9, 10, 12 |
| Accepted in reduced form | 6 (low severity; real corner exists via #2), 8 (warning only) |
| Rejected with rationale | 11 |

---

## Per-issue responses

### Issue 1 — restore_from_disk resets whole chip — **ACCEPTED, different fix than suggested**

Confirmed at `actuators.py:206-213`. Realistic trigger is nastier than the review states: a *first-ever* start that dies on a config error (e.g. old-schema zones.yaml → `sys.exit(2)`) still runs `ExecStopPost --restore-bios`; no snapshot has ever existed, so today every `pwm*_enable` on the chip gets forced to mode 5 even though fand touched nothing.

The review suggests restoring "a known manifest of managed channels." We'll do something simpler and safer: **no-op**. `snapshot()` runs before `take_over()` in `Actuators.__enter__` (`actuators.py:155,162`), so *snapshot file absent ⇒ fand never took over any channel ⇒ there is nothing to undo*. Parsing zones.yaml in the ExecStopPost context (file possibly absent/old-schema/unreadable) buys nothing over that invariant. The module docstring (`actuators.py:9-12`), which currently promises the mode-5 fallback, gets updated, and the snapshot-before-takeover ordering gets a comment pinning it as load-bearing.

### Issue 2 — Watchdog vs tick latency — **ACCEPTED**

Confirmed: ping only at end of `_tick()` (`daemon.py:644`), skipped entirely on tick exception; worst-case tick (nvidia-smi 5 s + upsc 3 s + ntfy 5 s + hourly inline refit over 20k samples × N sensors + 20k-row rewrite) clears `WatchdogSec=10` easily. Fix (one coherent design with Issues 9/10):

- watchdog ping moves to `run()`'s loop — fires after every tick *attempt*, success or failure (safe because of the Issue-10 failure-streak exit);
- `WatchdogSec=10 → 30`;
- nvidia-smi timeout 5→3 s, upsc 3→2 s — keeps control-loop latency sane during a source wedge;
- refit staggered (see Issue 9) so no single tick is heavy.

### Issue 3 — Missing sensor bypasses critical floor — **ACCEPTED, top priority**

Confirmed at `daemon.py:519-526` → excluded from the critical check at 530-539. This is the most important finding in the review; thanks. The fix is two-layer (operator confirmed the policy):

- **Startup validation:** one `read_all()` in `Daemon.__init__`; declared hwmon sensor whose chip or label doesn't resolve → refuse to start, listing all offenders (permanent zones.yaml errors: pulled drive, typo). A `gpu`-chip sensor with nvidia-smi absent only warns — optional source, runtime escalation covers it; failing the unit on a slow GPU driver at boot would be worse than the bug.
- **Runtime escalation:** per-sensor last-good tracking. Unreadable within a grace window (`sensor_fail_grace_s`, default 30 s) → **hold** the last-good temp, sensor stays in stress *and* critical checks (a flake mid-hot-moment still protects). Past grace → inject `critical_c` into the effective-temp path → existing `any_critical` → `set_all(255)`, with a distinct alarm key (`missing_critical:`) so sensor-loss roar is distinguishable from a genuine thermal event. Recovery requires 3 consecutive good reads (anti-flap). Per-sensor `on_unreadable: alarm` override in zones.yaml for sensors with their own protection story (the Blackwell's board fans are not host-controllable anyway — set it on the GPU sensor to avoid an all-night roar on a driver wedge; the thermally borderline spd5118 keeps full protection).
- **Learning-pool protection** (subtlety the review didn't flag): held/synthetic temps never enter equilibrium windows or `EquilibriumSample.temps` — only fresh reads do — so the fail-safe can't poison the ridge fit. FF correction is also skipped for held values (layering a surge prediction on a stale temp double-counts).

DESIGN.md's "Safety properties" bullet (None = "no signal") gets rewritten to match the new contract.

### Issue 4 — PWM writes not verified — **ACCEPTED, with a correction**

Partially right: `Actuators.set_pwm` *already* does read-back verification with ±2 tolerance and returns a bool (`actuators.py:126-145`) — the bug is that the daemon ignores it everywhere. Fix:

- `set_all` returns the list of failed channels; critical path: any failure → `take_over()` + one retry → still failing → short-cooldown alarm. `take_over()`'s `RuntimeError` deliberately propagates into the Issue-10 failure counter.
- Normal path: per-channel write-failure streak (3) → alarm + `take_over()` + rewrite.
- Plus a cheaper, more direct detector the review missed: `obs.pwm_enable` is already read every tick — any managed channel whose enable ≠ 1 (BIOS reasserted after suspend/resume) triggers alarm + re-`take_over()`, even when the duty register happens to match and read-back would pass.

### Issue 5 — Calibration phase-3 stale baseline — **ACCEPTED**

Confirmed at `calibrate.py:259`. Fix as suggested (rolling pre-drop variant): `_average_temps` re-sampled immediately before each fan's drop, so each ΔT measures against its own pre-drop steady state and accumulated chassis heat is absorbed. +10 s per fan. Validation requires a full idle recalibration run, which will be scheduled separately (it spins every fan; operator must be present).

### Issue 6 — equilibria.jsonl unbounded growth — **ACCEPTED IN REDUCED FORM**

Mostly overstated: the per-fan window clears after each sample (≤1 row/fan/~30 s → ~17k rows/day absolute worst case), `_persist_equilibria_atomic` rewrites from the 20k-cap deque every refit, and restarts *help* (`last_fit_t=0` + monotonic `obs.t` → immediate first-tick refit+trim). In normal operation the file exceeds the cap by roughly one refit-interval of appends (~700 rows).

However, our adversarial pass found the one genuinely unbounded path, and it's coupled to Issue 2: if the inline refit reliably blows `WatchdogSec=10`, systemd kills the daemon *before* the trim site (`daemon.py:626`); appends from prior ticks persist; the restart re-runs the same heavy first-tick refit and dies again — file and the `read_text()` startup spike grow without bound. **The staggered refit (Issue 9) is what actually closes this.** An append-counter compaction (trigger the existing atomic rewrite after ~2000 appends) additionally covers the misconfigured-`fit_interval_s` corner. Both land.

### Issue 7 — No target_c < critical_c validation — **ACCEPTED**

Fix at daemon init: `RuntimeError` listing every sensor with `critical_c <= target_c`; also `pwm_min` ∈ [0,255] and non-negative `cools` weights while we're there. Not mirrored in `fand-calibrate` — its generated defaults are valid by construction; the risk is operator edits, which daemon-init validation catches at the only point it matters.

### Issue 8 — Sensors in no fan's cools — **ACCEPTED IN REDUCED FORM (warning only)**

Confirmed behavior, but it's partially intentional: declaring a sensor with no `cools:` coverage is legitimate config (room-ambient probe, a drive with no ducted fan) — it buys monitoring plus the critical floor. Auto-linking all fans, as the review floats, would make an ambient probe ramp everything and pollute the learned cooling map. We'll add the startup warning listing uncovered sensors ("no proactive cooling, critical floor only") and stop there.

### Issue 9 — Inline refit blocks control loop — **ACCEPTED**

Fix: staggered refit. When `fit_interval` elapses, snapshot the sample pool and enqueue all sensors; each tick fits **one** sensor; queue drain triggers `save_models` + the equilibria rewrite. Chosen over the review's worker-thread option deliberately: `fit()` writes `SensorModel.state` non-atomically while `relevance()`/`is_trained()` read it mid-tick — threads would need locking around the control law for marginal benefit. Staggering keeps everything single-threaded and deterministic and makes the worst tick O(one fit).

### Issue 10 — Silent degradation on tick exceptions — **ACCEPTED**

Fix: consecutive-failure counter in `run()`; reset on success; at 5 → `sd_notify(STATUS=…)` + exit non-zero → `Restart=on-failure` restarts; persistent failure exhausts systemd's rate limit and leaves the unit failed **with BIOS modes restored** (context-manager restore + ExecStopPost). The resulting contract is the right one: *if fand can't run, fans go back to BIOS.*

### Issue 11 — Chronological holdout — **REJECTED**

The suggested fix (blocked/random holdout) would make the gate *worse*. Equilibrium rows are near-duplicates by construction — `is_fan_equilibrium` requires |dT/dt| ≤ 0.05 °C/s and stable features over ≥30 s — so any split that mixes time-adjacent rows across train/test leaks: near-twin rows land on both sides, holdout RMSE deflates, and `is_trained()` (`model.py:165-171`) opens on an overfit model, letting learned γ displace calibration seeds. That is precisely the failure the gate exists to prevent. Chronological splitting under regime shift errs *pessimistic* — gate stays closed, control stays on phase-3 seeds — which fails safe.

We attacked our own position before rejecting yours: the residual hole (a flat recent tail making chronological RMSE optimistic) is real but already backstopped — the identifiability guards (`model.py:242-253`) run on the whole pool *before* the split, so a flat pool can't promote data-learned γ at all (falls back to seed-scaled values), and the all-zero-γ seed-fallback invariant (`model.py:205-215`) catches the rest. Blocked CV wouldn't fix that case either (a flat block is flat under any split scheme), requires a per-window group key the samples don't carry, and multiplies solves per refit — aggravating Issue 9 — to gate a binary switch that already has a hard safety net beneath it.

Concession: the `n_train = max(int(n*0.8), n-50)` cap means the holdout shrinks to a thin most-recent slice as the pool grows, so `rmse` validates "the recent regime," not all regimes. That deserves real documentation, not silence — the split site gets an expanded comment stating the above, and `config.yaml.example` notes that `rmse` is regime-dependent.

### Issue 12 — install.sh v1 wording — **ACCEPTED**

Header comment and post-install heredoc rewritten to the v2 flow (calibrate → review `sensors:` target/critical + `fans:` cools → start); also "~5 minutes" → "~15 minutes" to match CLAUDE.md.

---

## Additions beyond the review's scope

- **Minimal pytest suite** (repo currently has none; operator approved): escalation state machine, restore no-op, `set_all` return contract, config validation, compaction trigger, held-temp exclusion from the learning pool. The safety code being touched is exactly the code that regresses silently.
- Issue statuses in `grok.review.md` will be updated (`Status: fixed` + commit ref; 6/8 `partial`, 11 `rejected`) per the review's "Claude Code usage" instructions.

## Fix order

3 → 2/9/10 (one cluster) → 1 → 4 → 5 → 6/7/8 → 11/12 docs → review bookkeeping. Matches the review's recommended order except that 2/9/10 land as a single coherent loop-hardening change — the watchdog ping placement, the staggered refit, and the failure-streak exit only make sense together.

## Resolution commits

| Issue(s) | Commit | |
|---|---|---|
| 3 | `98cb651` | missing-sensor fail-safe |
| 2, 9, 10 (and 6's kill-loop path) | `4351526` | loop hardening |
| 1 | `8c8c3a9` | restore no-op without snapshot |
| 4 | `3379c71` | load-bearing PWM verification + enable-drift |
| 5 | `cce4e82` | per-fan phase-3 baseline |
| 6, 7, 8 | `f95d3f0` | compaction, validation, uncovered warning |
| 11 (rejected, documented), 12 | `e05ab3d` | holdout rationale, install.sh wording |
| tests | `46b3d21` | 16 unit tests over the new safety paths |
