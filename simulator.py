import math
import argparse

# Boeing 747-8 longitudinal flight simulator with a cascaded autopilot.
# Models takeoff, climb, cruise and approach, four independently-failable
# engines, an ISA atmosphere, and post-stall departure (loss of control).
#
# 747-8 data: MTOW 442 t, wing area 560 m2, four GEnx-2B67 at 296 kN each.

dt = 0.01

# airframe
mass = 380_000.0        # kg, representative heavy operating weight (MTOW is 442 t)
inertia = 4.5e7         # kg m^2, pitch axis Iyy (~4.5e7 for the 747)
g = 9.81
wing_area = 560.0       # m^2
chord = 8.3             # m, mean aerodynamic chord (moment arm)

# propulsion: four GEnx-2B67
n_engines = 4
thrust_per_engine = 296_000.0           # N (~66,500 lbf) each
thrust_all = n_engines * thrust_per_engine
thrust_idle_all = 0.05 * thrust_all     # flight idle (working engines only)

# aerodynamics
CL_alpha = 5.0          # lift-curve slope (per radian), swept wing
alpha_stall = math.radians(15)
k_induced = 0.045       # 1/(pi*AR*e), AR ~ 8.3
CD_max = 1.8            # broadside drag coefficient (fully separated)

# flap/gear configurations: (zero-alpha lift, parasitic drag)
CONFIGS = {
    'clean':   dict(CL0=0.20, CD0=0.020),
    'takeoff': dict(CL0=0.90, CD0=0.045),
    'landing': dict(CL0=1.60, CD0=0.090),
}

# post-stall departure (only active once the wing is stalled)
Cm_stable = 0.6         # static stability while attached
Cm_damp = 4.0           # pitch damping while attached
Cm_damp_sep = 0.4       # residual pitch damping when separated (bounds the tumble)
Cm_unstable = 6.0       # divergence once stalled: this is what makes it tumble

# ISA atmosphere (troposphere)
rho0 = 1.225
def air_density(h):
    h = max(0.0, h)
    return rho0 * (1 - 2.25577e-5 * h) ** 4.25588

def stall_speed(h, config):
    # slowest the wing can hold the weight at this altitude/configuration
    cfg = CONFIGS[config]
    CL_max = cfg['CL0'] + CL_alpha * alpha_stall
    return math.sqrt(2 * mass * g / (air_density(h) * wing_area * CL_max))

# control loops
pitch_wn = 1.5          # inner pitch loop natural frequency (heavy jet, sluggish)
pitch_zeta = 0.9
Kp_pitch = inertia * pitch_wn * pitch_wn
Kd_pitch = 2 * pitch_zeta * inertia * pitch_wn
Kp_spd = 80_000.0       # autothrottle gains (N per m/s)
Ki_spd = 30_000.0
Ki_vs = 0.02            # climb-rate loop integral gain
q_ref = 0.5 * rho0 * 70.0 ** 2   # reference dynamic pressure for control authority

# gains from the takeoff gain search (regenerate with --search)
DEFAULT_K_H = 0.15
DEFAULT_K_VS = 0.03

# flight phases: initial state, target, configuration
PHASES = {
    'takeoff':  dict(y0=0.0,     V0=85.0,  V_target=95.0,  target=600.0,   vs_max=13.0, config='takeoff', dur=120),
    'climb':    dict(y0=3000.0,  V0=160.0, V_target=160.0, target=7000.0,  vs_max=15.0, config='clean',   dur=200),
    'cruise':   dict(y0=10000.0, V0=250.0, V_target=250.0, target=10000.0, vs_max=8.0,  config='clean',   dur=320),
    'approach': dict(y0=900.0,   V0=85.0,  V_target=85.0,  target=0.0,     vs_max=5.0,  config='landing', dur=220),
}

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def aero_forces(vx, vy, pitch, pitch_rate, rho, cfg):
    # full-envelope aero: a wing while attached, a tumbling flat plate once
    # the flow separates past the stall angle. returns x/y force, pitching
    # moment, and separation fraction (0 attached, 1 fully stalled).
    V2 = vx * vx + vy * vy
    V = math.sqrt(V2)
    if V < 1e-6:
        return 0.0, 0.0, 0.0, 0.0

    gamma = math.atan2(vy, vx)
    alpha = wrap_angle(pitch - gamma)
    sep = max(0.0, min(1.0, (abs(alpha) - alpha_stall) / math.radians(8)))

    q = 0.5 * rho * V2
    CL_attached = cfg['CL0'] + CL_alpha * alpha
    CL = (1 - sep) * CL_attached + sep * 2 * math.sin(alpha) * math.cos(alpha)
    CD = cfg['CD0'] + (1 - sep) * k_induced * CL_attached * CL_attached \
        + sep * CD_max * math.sin(alpha) * math.sin(alpha)

    lift = q * wing_area * CL
    drag = q * wing_area * CD
    fx = -drag * math.cos(gamma) - lift * math.sin(gamma)
    fy = lift * math.cos(gamma) - drag * math.sin(gamma)

    Cm = (1 - sep) * (-Cm_stable * alpha) + sep * (Cm_unstable * math.sin(alpha))
    damp = Cm_damp * (1 - sep) + Cm_damp_sep * sep
    moment = q * wing_area * chord * Cm \
        - damp * q * wing_area * chord * chord * pitch_rate / (2 * max(V, 1.0))
    return fx, fy, moment, sep

def simulate(K_h, K_vs, phase='takeoff', fail_engines=(), fail_time=None, record=False):
    P = PHASES[phase]
    cfg = CONFIGS[P['config']]
    vs_max = P['vs_max']
    target = P['target']
    V_target = P['V_target']
    n_fail = len(fail_engines)
    frac = (n_engines - n_fail) / n_engines      # thrust fraction after failure

    pitch = 0.0
    pitch_rate = 0.0
    x = 0.0
    y = P['y0']
    vx = P['V0']
    vy = 0.0

    spd_integral = 0.0
    vs_integral = 0.0
    airborne = False

    total_error = 0.0
    overshoot = 0.0
    fast_climb = 0.0
    effort = 0.0
    prev_commanded = 0.0
    history = []

    t = 0.0
    while t < P['dur']:
        rho = air_density(y)
        V = math.hypot(vx, vy)

        # available thrust drops to the surviving-engine fraction after failure
        failed_now = fail_time is not None and t >= fail_time and n_fail > 0
        frac_now = frac if failed_now else 1.0
        avail_max = thrust_all * frac_now
        avail_idle = thrust_idle_all * frac_now

        # autothrottle: hold the phase target speed (anti-windup on thrust limits)
        spd_error = V_target - V
        spd_integral = spd_integral + spd_error * dt
        thrust = Kp_spd * spd_error + Ki_spd * spd_integral
        thrust_clamped = max(avail_idle, min(avail_max, thrust))
        if thrust_clamped != thrust:
            spd_integral = spd_integral - spd_error * dt
            thrust = thrust_clamped

        # outer loop: altitude error -> commanded climb rate (limited)
        alt_error = target - y
        vs_cmd = max(-vs_max, min(vs_max, K_h * alt_error))

        # middle loop: climb-rate error -> commanded pitch (with anti-windup)
        vs_error = vs_cmd - vy
        commanded_pitch = K_vs * vs_error + Ki_vs * vs_integral
        cp_clamped = max(-math.radians(15), min(math.radians(15), commanded_pitch))
        if cp_clamped == commanded_pitch:
            vs_integral = vs_integral + vs_error * dt
        commanded_pitch = cp_clamped

        aero_fx, aero_fy, aero_moment, sep = aero_forces(vx, vy, pitch, pitch_rate, rho, cfg)

        # inner loop: elevator authority fades with dynamic pressure (slow = mushy)
        # and vanishes when the tail is in separated flow (a deep stall is uncontrollable)
        q = 0.5 * rho * V * V
        authority = min(1.0, q / q_ref) * (1 - sep)
        error = commanded_pitch - pitch
        torque = authority * (Kp_pitch * error - Kd_pitch * pitch_rate) + aero_moment
        pitch_rate = pitch_rate + (torque / inertia) * dt
        pitch = wrap_angle(pitch + pitch_rate * dt)

        fx = thrust * math.cos(pitch) + aero_fx
        fy = thrust * math.sin(pitch) + aero_fy - mass * g
        vx = vx + (fx / mass) * dt
        vy = vy + (fy / mass) * dt
        x = x + vx * dt
        y = y + vy * dt

        total_error = total_error + abs(alt_error) * dt
        if y > target:
            overshoot = overshoot + (y - target) * dt
        if vy > vs_max + 1.0:
            fast_climb = fast_climb + (vy - vs_max - 1.0) * dt
        effort = effort + abs(commanded_pitch - prev_commanded)
        prev_commanded = commanded_pitch

        if record:
            history.append((t, x, y, math.degrees(commanded_pitch),
                            math.degrees(pitch), math.hypot(vx, vy),
                            stall_speed(y, P['config'])))
        t = t + dt

        # stop the run at ground impact (after the aircraft has actually flown)
        if y > P['y0'] + 20 or P['y0'] > 50:
            airborne = True
        if airborne and y <= 0:
            break

    score = total_error + 30.0 * overshoot + 40.0 * fast_climb + 20.0 * effort
    if record:
        return score, history
    return score

def write_flight(filename, history):
    with open(filename, 'w') as f:
        f.write('t,x,y,commanded_pitch,actual_pitch,airspeed,stall_speed\n')
        for row in history:
            f.write(','.join(str(v) for v in row) + '\n')

def run_gain_search():
    best_score = float('inf')
    best_gains = (DEFAULT_K_H, DEFAULT_K_VS)
    with open('results.csv', 'w') as f:
        f.write('K_h,K_vs,Score\n')
        for K_h in [0.06, 0.08, 0.10, 0.12, 0.15]:
            for K_vs in [0.02, 0.03, 0.04, 0.05, 0.07]:
                score = simulate(K_h, K_vs, phase='takeoff')
                f.write(f"{K_h},{K_vs},{score}\n")
                if score < best_score:
                    best_score = score
                    best_gains = (K_h, K_vs)
    print(f'gain search best: K_h={best_gains[0]}, K_vs={best_gains[1]}, score={best_score:.1f}')
    return best_gains

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Boeing 747-8 flight / engine-failure simulator')
    ap.add_argument('--phase', choices=list(PHASES), default='takeoff',
                    help='flight phase the failure happens in (default: takeoff)')
    ap.add_argument('--fail-engines', default='',
                    help='comma list of engines 1-4 that fail, e.g. "1,2" (default: none)')
    ap.add_argument('--fail-time', type=float, default=None,
                    help='seconds into the run when the engines fail')
    ap.add_argument('--out', default='flight.csv', help='output flight file (default: flight.csv)')
    ap.add_argument('--search', action='store_true', help='run the takeoff gain search first')
    args = ap.parse_args()

    K_h, K_vs = (run_gain_search() if args.search else (DEFAULT_K_H, DEFAULT_K_VS))

    fail_engines = tuple(sorted({int(e) for e in args.fail_engines.split(',') if e.strip()}))
    for e in fail_engines:
        if e < 1 or e > n_engines:
            raise SystemExit(f'engine {e} out of range 1-{n_engines}')
    fail_time = args.fail_time
    if fail_engines and fail_time is None:
        fail_time = {'takeoff': 12.0, 'climb': 20.0, 'cruise': 30.0, 'approach': 30.0}[args.phase]

    score, history = simulate(K_h, K_vs, phase=args.phase,
                              fail_engines=fail_engines, fail_time=fail_time, record=True)
    write_flight(args.out, history)

    P = PHASES[args.phase]
    last_t, _, last_y = history[-1][0], history[-1][1], history[-1][2]
    peak = max(r[2] for r in history)
    min_V = min(r[5] for r in history)
    crashed = last_y <= 1 and last_t < P['dur'] - 0.5
    print(f"phase={args.phase}  config={P['config']}")
    if fail_engines:
        out = n_engines - len(fail_engines)
        print(f"engines failed: {list(fail_engines)} ({len(fail_engines)} of {n_engines}) "
              f"at t={fail_time:.0f}s -> {100*out/n_engines:.0f}% thrust remaining")
    else:
        print('engines: all running (nominal flight)')
    print(f"peak altitude {peak:.0f} m, min airspeed {min_V:.0f} m/s")
    if crashed:
        print(f"OUTCOME: ground impact at t={last_t:.1f}s")
    else:
        print("OUTCOME: still flying at end of window")
    print(f"wrote {args.out}  (view with: python visualiser.py {args.out})")
