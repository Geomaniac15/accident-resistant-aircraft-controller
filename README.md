# Accident-Resistant Aircraft Controller

A six-degree-of-freedom Boeing 747-8 flight simulator built to answer one
question: **how well does an autopilot survive things going badly wrong?**
It flies the aircraft through takeoff, climb, cruise and approach, injects
engine failures and control-system accidents at chosen moments, and records
whether the controller keeps the aircraft flying, lands it, or loses it.
A SLURM-shardable sweep harness runs the controller against hundreds of
thousands of failure scenarios and renders the results as survival heatmaps.

## Quick start

Requires Python 3 with `numpy` and `matplotlib` (a `.venv` works fine).

```bash
# fly a nominal takeoff
python simulator.py

# fail both left engines 12 s into the takeoff roll
python simulator.py --phase takeoff --fail-engines 1,2

# rudder hardover in cruise, with moderate turbulence
python simulator.py --phase cruise --accident rudder-hardover --turbulence 3

# play back the recorded flight
python visualiser.py flight.csv         # instrument panels
python visualiser.py flight.csv --3d    # 3D path with scrub slider

# run the regression tests
python test_simulator.py
```

## The aircraft model (`simulator.py`)

A rigid-body 6-DOF model of a Boeing 747-8 (380 t operating weight, 560 m²
wing, four GEnx-2B67 engines at 296 kN). Attitude is integrated as a
**quaternion**, so the model stays valid through large upsets — inverted,
spinning, tumbling — without gimbal lock. Integration is explicit Euler at
`dt = 0.01 s`.

**Aerodynamics.** Full-envelope longitudinal model: linear lift below stall
(α ≈ 15°), blending over ~8° into flat-plate behaviour with post-stall pitch
divergence, so a stall is a genuine departure rather than a soft ceiling.
Lateral-directional axes use standard stability derivatives (dihedral effect,
weathercock stability, roll/yaw damping, adverse yaw), which is what makes
asymmetric failures dangerous: a dead outboard engine yaws the aircraft, the
sideslip rolls it, and if the controls can't hold it the spiral tightens.
Air density follows the ISA troposphere. Flap/gear configuration (clean /
takeoff / landing) sets zero-alpha lift and parasitic drag per phase.

**Propulsion.** Four independently failable engines. Thrust responds with a
first-order spool lag (~4 s up, ~1.5 s down) — there is no instant thrust,
which is exactly why the microburst scenario is lethal on takeoff. Failed
engines wind down to a **windmilling drag** of ~12 kN at their wing station,
so an engine failure both decelerates and yaws the aircraft.

**Controls.** Surfaces have travel limits (±25° elevator/rudder, ±20°
aileron) and **actuator rate limits** (60/60/50 °/s). A hardover therefore
slews to the stop at actuator speed, and a jammed elevator freezes wherever
the surface physically was at that moment.

**Environment.** Constant NED wind, plus optional Dryden-style turbulence:
per-axis Gauss–Markov gusts with 300 m horizontal / 100 m vertical length
scales, seeded for reproducibility (`--turbulence SIGMA --seed N`; roughly
1.5 = light, 3 = moderate, 6 = severe).

**Mass.** The sweep varies weight from 270 t to 442 t; rotational inertia
scales with it.

## The autopilot

A classical cascade, deliberately failure-unaware (it never "knows" an
accident happened — it just keeps flying its loops):

- **Autothrottle** — PI on airspeed, with anti-windup at the thrust limits.
- **Vertical guidance** — altitude error → climb-rate command (clamped per
  phase) → pitch command (PI).
- **Elevator** — PD pitch-attitude hold.
- **Aileron** — heading hold via a bank command (limited to 25°), PD bank hold.
- **Rudder** — sideslip nulling plus a yaw damper.

The two guidance gains (`K_h`, `K_vs`) are tunable; `--search` grid-searches
them on the nominal takeoff, and the sweep carries a 3×3 gain grid as an axis.

## Failure modes

Injected at `--fail-time` (defaults per phase if a failure is requested):

| Failure | Effect |
|---|---|
| `--fail-engines 1..4` | any subset; thrust decays, windmill drag yaws toward the dead side |
| `rudder-hardover` | rudder drives to the stop (side set by `--side`) |
| `aileron-hardover` | ailerons drive to the stop |
| `elevator-jam` | elevator freezes at its current position |
| `runaway-trim` | nose-down pitching moment grows until it overpowers the elevator |
| `windshear` | microburst: building headwind, strong downdraft core, tailwind exit |
| `wing-loss` | 45 % lift loss plus rolling/yawing moments toward the damaged side |
| `explosion` | impulsive tumble, half the lift, 20 % control effectiveness |

## Outcomes

A run ends in one of three states:

- **survived** — still flying when the phase window closes;
- **landed** — ground contact that meets the touchdown criteria: sink rate
  under 3 m/s, bank within 5°, pitch between −3° (nosewheel-first) and +12°
  (tail strike);
- **CRASH** — any other ground contact.

Recorded metrics per run: impact/end time, peak and minimum altitude, minimum
airspeed, minimum stall margin, maximum bank and maximum angle of attack.

## Flight phases

| Phase | Start | Speed target | Altitude target | Config | Window |
|---|---|---|---|---|---|
| `takeoff` | 0 m, 85 m/s | 95 m/s | 600 m | takeoff flaps | 120 s |
| `climb` | 3 000 m | 160 m/s | 7 000 m | clean | 200 s |
| `cruise` | 10 000 m | 250 m/s | 10 000 m | clean | 320 s |
| `approach` | 900 m | 85 m/s | 0 m (touchdown) | landing flaps | 220 s |

## Visualiser (`visualiser.py`)

Flights are recorded to CSV (position, attitude, α/β, airspeed, stall speed,
control-surface positions, total thrust, and the failure time in a
`# fail_time=` header).

- **2D panels** (default): bank view, ground track, altitude, airspeed vs
  stall speed, control-surface deflections and total thrust, with the failure
  moment marked on every time axis. The aircraft glyph turns orange past 60°
  of bank and red when stalled.
- **3D view** (`--3d`): the flight path coloured by stall margin (blue = safe,
  grey = marginal, red = stalled), an attitude-true aircraft glyph, and a
  scrub slider with play/pause — drag the slider to step through the moment
  things go wrong.

Older CSVs without the control columns still play; the control panels are
simply omitted.

## Failure sweep (`sweep.py`)

Runs the controller against the cartesian product of
*phase × failure × affected unit/side × failure timing × weight × wind ×
controller gains* — ~226 000 scenarios — and writes one CSV row per run.

```bash
# run everything locally (slow) or one shard of 64:
python sweep.py --shard 0/64 --out results/shard_0.csv

# aggregate shards into a survival matrix (text):
python sweep.py --summarize 'results/shard_*.csv'

# render the survival heatmap (failure type x failure time, per phase):
python sweep.py --heatmap 'results/shard_*.csv' --heatmap-out results/heatmap.png
```

On a SLURM cluster, `sbatch run_sweep.slurm` submits a 64-task array job
(shard size and partition are set in the script; `OUTDIR=... sbatch` to
redirect output).

## Tests (`test_simulator.py`)

Physics regression tests — no framework needed:

```bash
python test_simulator.py     # or pytest, if installed
```

They pin down: nominal phases survive with positive stall margin, thrust
changes no faster than the spool lag allows, surfaces respect actuator rate
limits, a left-side engine failure yaws left, dead engines produce windmill
drag, turbulence is seed-reproducible, touchdown classification separates a
flown approach from a crash, and stall speed moves the right way with
altitude and flaps.

## Repository layout

| File | Purpose |
|---|---|
| `simulator.py` | aircraft model, autopilot, failure injection, CLI |
| `visualiser.py` | 2D / 3D flight playback |
| `sweep.py` | scenario grid, shard runner, summary table, heatmap |
| `test_simulator.py` | physics regression tests |
| `run_sweep.slurm` | SLURM array job for the full sweep |
| `*.csv` | recorded flights (generated, not tracked in git) |
| `results/` | sweep shards and heatmaps (generated) |

## Known limitations / roadmap

- **No Mach effects** — cruise at 250 m/s / 10 km is ~M 0.85, where drag rise
  and lift-curve changes are real; the model ignores compressibility.
- **No ground roll** — takeoff starts already at rotation speed; there is no
  gear, runway friction or ground effect.
- **No fuel burn** — mass is constant within a run.
- **Perfect sensors** — the autopilot sees exact state, with no noise or latency.
- **The interesting next step**: a failure-aware controller — angle-of-attack
  envelope protection, windshear escape guidance (TOGA + pitch), feedforward
  rudder trim for engine-out — swept against the same scenario grid to compare
  survival heatmaps against this failure-unaware baseline.
