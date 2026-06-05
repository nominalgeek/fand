# Design

Companion to [CLAUDE.md](CLAUDE.md). CLAUDE.md is operational reference; this
document is *why* — the goal, the architectural decisions, and the failure
modes they exist to prevent.

## Goal

fand is an autonomous thermal-management daemon for Linux hosts where the
operator **doesn't necessarily know which fan cools what**, and the system
behaves as a coupled multi-fan multi-sensor whole rather than a set of
independent thermostats.

The target deployment is the kind of machine where:

- Multiple physical fans share an airstream and a chassis volume — front
  intakes, rear/top exhaust, CPU cooler / AIO radiator fans, GPU exhaust
  contributing to case ambient, RAM-adjacent airflow, drive-bay flow.
- Heat sources interact — GPU exhaust warms the case, which raises CPU
  intake air, which the AIO loop has to dissipate, which the radiator fans
  push out as heat to the room.
- Sensor-to-fan relationships are non-obvious or counterintuitive — a "rear
  exhaust" fan's effect on CPU temp depends on whether the AIO water loop is
  doing the heavy lifting, whether the GPU is dumping heat into the case,
  and what the front-intake pressure looks like.
- The operator wants the system to **figure it out** instead of being
  configured with a hand-picked curve per fan against a hand-picked sensor.

The system optimizes for: keeping every sensor below its rated critical
threshold, with margin, at minimum total noise.

## Anti-goals

fand is **not**:

- A static BIOS-curve replacement. Boards that already have a sensible
  BIOS SmartFan profile, with each fan tied to one obvious heat source,
  don't need fand — they need maybe a curve tweak.
- A general-purpose `fancontrol` substitute. Use lm-sensors's `fancontrol`
  for setups where you know exactly which fan cools which sensor and just
  want a curve.
- A single-host benchmark. The learned model adapts to *one* machine's
  thermal personality. Don't expect transfer between builds.
- An aggressively-tuned over-clocker's tool. The defaults are conservative;
  the goal is "no thermal events, minimum noise, no operator decisions."

## The architectural mistake we corrected

The original fand had a *per-zone* architecture: each PWM channel was a
"zone" with one designated primary sensor, its own bootstrap curve, and an
independent ridge regression that learned PWM ↔ sensor-temp from
equilibrium samples. A "zone" treated each fan like an independent
thermostat loop.

This failed concretely. On the target host (RTX PRO 6000 Blackwell at
600 W under sustained load):

```
zone           T_z   pwm  fan_rpm     source
pwm1        51.0°C   133        -  bootstrap     (case ambient → intakes)
pwm2        63.9°C   172        -  bootstrap     (Tctl → rear exhaust)
pwm3        52.0°C   136        -  bootstrap     (SYSTIN → VRM fan)
pwm4        88.0°C   255        -  bootstrap     (GPU → rad fans)
```

The rad fans (pwm4) were saturated at 255 because the GPU was at 88 °C.
Every other fan was loafing at PWM 130–170 because its assigned sensor
was below its target. The system was at thermal limit even though three of
four fans had massive headroom.

The fans weren't reacting because nothing connected them to the heat. Each
fan saw exactly one sensor (its zone's primary) and ignored the others.
But physically, all four fans contribute to GPU cooling — case intakes
lower GPU intake air temperature, rear exhaust pulls hot air out, even the
rad fans push case airflow. The per-zone abstraction made the daemon blind
to all of that.

## The new model

Two things in the config — sensors and fans:

```yaml
sensors:
  cpu_die:      { chip: k10temp,     label: Tctl,    target_c: 70, critical_c: 90 }
  case_ambient: { chip: nct6799,     label: AUXTIN0, target_c: 45, critical_c: 65 }
  nvidia_gpu:   { chip: gpu,         label: temp_c,  target_c: 80, critical_c: 90 }
  dimm_a:       { chip: spd5118[0],  label: temp1,   target_c: 45, critical_c: 55 }
  # ...

fans:
  pwm1:
    fan_tach: fan1_input
    cools: { case_ambient: 3.0, nvidia_gpu: 0.6, dimm_a: 0.4, dimm_b: 0.4 }
  pwm4:
    fan_tach: fan4_input
    cools: { cpu_die: 2.8, nvidia_gpu: 0.3 }
```

**Sensors declare what we care about** — chip + label to find the reading,
target_c we'd like to hold below, critical_c that triggers emergency floor.

**Fans declare what they can affect** — `cools` is a dict of
sensor → seed cooling weight (ΔT in °C, from phase 3 calibration). The
weights are a starting point; the daemon learns the real coefficients from
runtime data.

There are no "zones." Each fan is a control input that contributes,
proportional to its learned cooling power, to whichever sensors are
currently under stress.

## Control law

At each 2 s poll:

1. **Read sensors.** For each declared sensor, look up its hwmon path and
   read.
2. **Compute stress.** For each sensor `s`:
   ```
   stress(s) = max(0, (T(s) − target_c(s)) / (critical_c(s) − target_c(s)))
   ```
   Stress = 0 means "at or below target, idle is fine." Stress = 1 means
   "at critical, max out." Linear interpolation between.
3. **Emergency floor.** If any sensor's measured temp is ≥ its
   `critical_c`, `set_all(255)` and emit an alarm keyed by sensor name.
   Override all other logic until the sensor recovers.
4. **Per-fan demand** (non-emergency):
   ```
   relevance(s, p) = γ_p(s) / max_q γ_q(s)
   demand(p)      = min(1, Σ_s [stress(s)^1.5 × relevance(s, p)])
   PWM(p)         = pwm_min(p) + demand(p) × (255 − pwm_min(p))
   ```
   Where `γ_p(s)` is fan `p`'s learned cooling power on sensor `s`. The
   relevance term normalizes — a fan that's only the weakest cooler of a
   sensor doesn't ramp hard for that sensor alone. The sum-with-saturation
   composes correctly when multiple sensors are stressed: a fan responsible
   for three slightly-warm sensors ramps more than one with one
   slightly-warm sensor. `stress^1.5` keeps low-stress sensors from
   piling up to demand = 1 from sheer count.
5. **Smooth the input.** Stress is computed from an EMA of the effective
   temp (`temp_smooth_tau_s`, default 15 s), not the raw reading. Sensors
   quantize in whole-°C steps at poll rate; on a low-thermal-mass die with
   a narrow target→critical span (GPU: 80→90), each ±1 °C flicker would
   jolt the demand target by tens of PWM counts. Real trends pass through;
   flicker averages away. The critical floor (step 3) always reads the raw
   instantaneous temp — safety never lags this filter.
6. **Slew-limit the output.** The computed PWM is a *target*; the applied
   PWM moves toward it at most `pwm_slew_up_per_s` counts/s upward and
   `pwm_slew_down_per_s` downward (defaults 30/5). A low-thermal-mass die
   (the GPU) swings several °C per second under bursty load; without
   output filtering every fan that cools it audibly bounces in lockstep
   with per-tick sensor noise. Asymmetric so cooling never lags a surge
   by much while decay is gentle. The critical floor (step 3) bypasses
   slew — 255 is applied immediately.

When γ is untrained (no equilibrium samples yet), the seed weights from
`cools:` substitute — normalized the same way. So at fresh install, fans
respond proportional to phase 3's measured ΔT per sensor. As the daemon
collects equilibrium samples and refits, the seed weights are replaced
with learned γ.

The same seed fallback applies if a trained sensor's learned γ comes out
all-zero (closed-loop data biases γ toward zero — the controller ramps
fans *because* temps rise, so PWM correlates positively with temperature
until load features deconfound it). The invariant is enforced per fan,
not just in aggregate: each fan's blended γ is floored at its seed value,
because observational runtime data cannot causally distinguish "this fan
doesn't cool this sensor" from controller-induced correlation, while
phase 3's perturbation is direct causal evidence that it does. Learning
may refine the cooling map upward, never erase it — a sensor under
stress always retains at least its calibration-measured fan response.
Recalibration is the way to lower the floor.

## How the cooling matrix evolves

The actual fan↔sensor cooling relationships are learned, not declared.
They live in `/var/lib/fand/model.json` (per-sensor), not in
`zones.yaml`.

**t = 0** (fresh install, post-calibrate). No equilibrium samples yet.
Each fan's `cools:` dict has phase-3-derived seed weights (e.g.
`pwm1 → case_ambient: 3.0 °C`, meaning "dropping pwm1 caused case_ambient
to rise 3.0 °C during phase 3"). Control uses these as normalized
relevance weights.

**t ≈ 1 h** (first refit). Daemon has collected ~hundreds of equilibrium
samples (poll snapshots where dT/dt is low and feature variance is low —
i.e., steady-state at some load). For each sensor `s`, fits a ridge
regression:

```
T_s ≈ baseline + Σ_k [α_k(s) · load_feature_k] − Σ_p [γ_p(s) · pwm_p]
```

`γ_p(s) ≥ 0` is the marginal cooling effect of fan `p` on sensor `s`.
Strong relationships (e.g. `pwm4 → cpu_die`, since rad fans dissipate AIO
water heat → CPU temp) emerge with large γ. Weak or coincidental
relationships (e.g. `pwm4 → dimm_a`, since rad fans don't really cool RAM
directly) trend toward zero — and get clipped to 0 post-fit.

The regression only fits coefficients the data can identify. A fan whose
PWM never varied across the pool, or a load feature pinned at one value
(GPU util sitting at 99 % through a long training run), is collinear with
the intercept and carries no information about its own coefficient —
those columns are excluded from the solve. Unidentified fans keep their
seed-derived γ; unidentified features fold their constant contribution
into the baseline. Columns re-enter automatically on later refits once
the rolling pool spans enough load regimes. The solve itself runs on
standardized columns with an unpenalized intercept, so the ridge penalty
is uniform across column scales and the baseline lands at a physical
temperature.

Until n_samples per sensor is high enough, γ stays regularized toward the
seed weights — partial trust in the learned values, partial trust in
phase 3. The trust ramp applies per fan, and only to fans the data
identified; sample count is not information about a coefficient the data
can't see.

**t > 1 d** (mature). The full (fan × sensor) γ matrix is well-fit. Seed
weights are mostly irrelevant; learned coefficients drive control. The
rolling deque of equilibrium samples (capped at 5000 per fan) gives the
matrix the ability to drift over weeks as conditions change — AC turning
on for summer, dust accumulating on rad fins (reducing γ for fans cooling
sensors that route through that radiator), fan wear, hardware additions.

The AR(1) feed-forward layer predicts ΔT_s from EMA of feature deltas
using the learned α coefficients, so fans ramp **before** sensors
actually climb under a load spike. (Disabled per-sensor while that
sensor's α isn't trained — feed-forward on uncalibrated coefficients
is worse than no feed-forward.)

## Configuration vs learned state

| | What | Source | When it changes |
|---|---|---|---|
| `/etc/fand/config.yaml` | Tunables (poll interval, fit interval, ridge λ, etc.) | Operator | Manually edit + restart |
| `/etc/fand/zones.yaml` `sensors:` | What sensors exist + their target/critical limits | Operator (or calibrate seeds) | Edit when adding hardware or revising thresholds |
| `/etc/fand/zones.yaml` `fans:` `cools:` | Phase-3-derived seed cooling weights | `fand-calibrate` | Re-run cal when hardware changes |
| `/var/lib/fand/model.json` | Learned (fan × sensor) γ matrix, α coefficients, baselines | Daemon | Refit every `fit_interval_s` (default 1 h) |
| `/var/lib/fand/equilibria.jsonl` | Training data (per-fan equilibrium samples) | Daemon | Append every time an equilibrium is detected |

**YAML defines what we care about and what tools exist.** model.json
defines how those tools actually affect those things, learned empirically.

## Safety properties (independent of the model)

The model is the optimization layer. Underneath it are unconditional safety
floors that work without any learned state:

- **Critical floor.** Any sensor at or above its `critical_c` → all fans
  to 255 immediately. Bypasses the model entirely.
- **Sensor fault.** An unreadable sensor never silently loses its critical
  floor. Startup refuses to run if a declared hwmon sensor's chip/label
  doesn't resolve (permanent config error). At runtime, a sensor that stops
  reading holds its last value for a grace window (`sensor_fail_grace_s`,
  default 30 s — it stays inside the stress and critical checks), then is
  assumed to be AT its `critical_c` — all fans max + alarm — until it reads
  again (3 consecutive good reads to clear, so an intermittent source can't
  flap the fans). Per-sensor `on_unreadable: alarm` opts out of the roar for
  sensors with their own protection story (e.g. a GPU whose board fans
  aren't host-controllable anyway). Held/synthetic temps never enter the
  equilibrium pool, so the fail-safe can't poison the learned model.
- **Fan-dead detection.** Tach reading 0 RPM while commanded PWM ≥ pwm_min
  for N consecutive ticks → alarm. (N = 3 by default to skip cold-start
  mechanical spin-up.)
- **Snapshot/restore.** PWM modes captured at startup; ExecStopPost
  restores them. If the daemon crashes or is stopped, BIOS curves
  take back over.

These are independent of `model.json` and won't be deferred if the model
is wrong or untrained.

## When fand is the wrong tool

If your build has:

- One CPU cooler fan + one rear exhaust + a clear BIOS curve that already
  handles them sensibly — use BIOS. Don't add a daemon.
- Hardware where the operator knows exactly which fan cools which sensor
  and wants a deterministic, hand-tuned curve per fan — use `fancontrol`
  with `/etc/fancontrol`. Simpler, easier to reason about.
- A use case that requires deterministic fan response (audio recording,
  precision measurement) — fand's stress-based control is non-deterministic
  by design (depends on recent learning state). Disable equilibrium learning
  (`min_samples_to_learn: 999999`) to pin it to bootstrap-from-seeds
  behavior.
- A multi-host fleet — fand learns one machine's personality. Each host
  needs its own training trajectory.

The case fand handles well is the case where you have N≥3 fans, M≥4
sensors that matter, and the fan↔sensor effectiveness map isn't obvious
without measuring it.
