import math
import argparse
import numpy as np

# Boeing 747-8 six-degree-of-freedom flight simulator with a full autopilot.
# Models takeoff, climb, cruise and approach, four independently-failable
# engines, an ISA atmosphere, post-stall departure, and the lateral-directional
# axes (roll and yaw). An asymmetric engine failure now produces a real yawing
# moment toward the dead engine, the dihedral roll-off that follows the sideslip,
# and (if the controls cannot hold it) a spiral / spin departure.
#
# 747-8 data: MTOW 442 t, wing area 560 m2, four GEnx-2B67 at 296 kN each.
# Attitude is integrated as a quaternion so the model survives large upsets
# (inverted, spinning) without gimbal lock.

dt = 0.01

# airframe
mass = 380_000.0        # kg, representative heavy operating weight (MTOW is 442 t)
g = 9.81
Ixx = 2.43e7            # kg m^2, roll inertia
Iyy = 4.49e7            # kg m^2, pitch inertia
Izz = 6.73e7            # kg m^2, yaw inertia
wing_area = 560.0       # m^2
span = 68.4             # m
chord = 8.3             # m, mean aerodynamic chord

# propulsion: four GEnx-2B67, numbered 1-4 left to right
n_engines = 4
thrust_per_engine = 296_000.0           # N (~66,500 lbf) each
engine_y = [-21.3, -11.6, 11.6, 21.3]   # lateral position from centreline (m)
idle_fraction = 0.05

# longitudinal aerodynamics
CL_alpha = 5.0
alpha_stall = math.radians(15)
k_induced = 0.045
CD_max = 1.8            # broadside drag coefficient (fully separated)
Cm_alpha = -1.2         # pitch static stability (attached)
Cm_q = -22.0            # pitch damping
Cm_de = -1.2            # elevator power
Cm_unstable = 1.2       # post-stall pitch divergence (departure)

# lateral-directional stability derivatives (per radian)
CY_beta = -0.90
CY_dr = 0.20
Cl_beta = -0.13         # dihedral effect (sideslip -> roll)
Cl_p = -0.45            # roll damping
Cl_r = 0.12
Cl_da = 0.10            # aileron power
Cl_dr = 0.012
Cn_beta = 0.14          # weathercock (directional) stability
Cn_p = -0.03
Cn_r = -0.32            # yaw damping
Cn_da = -0.004          # adverse yaw
Cn_dr = -0.12           # rudder power

# control surface travel limits (rad)
elev_max = math.radians(25)
ail_max = math.radians(20)
rud_max = math.radians(25)

# accident parameters (each only acts after fail_time)
ACCIDENTS = ['rudder-hardover', 'elevator-jam', 'aileron-hardover',
             'windshear', 'runaway-trim', 'wing-loss', 'explosion']
MB_HEADWIND = 30.0      # microburst peak horizontal wind (m/s)
MB_DOWNDRAFT = 28.0     # microburst peak downdraft (m/s)
MB_CORE_T = 10.0        # seconds from onset to the downdraft core
MB_WIDTH = 5.0          # microburst transition width (s)
MB_DURATION = 22.0
RUNAWAY_TRIM_RATE = 0.05    # nose-down pitching-moment coeff added per second
RUNAWAY_TRIM_MAX = 0.6      # ... up to this (overpowers the elevator)
WING_LOSS_LIFT = 0.55       # fraction of lift left after losing part of a wing
WING_LOSS_ROLL = 0.18       # rolling-moment coeff toward the damaged side
WING_LOSS_YAW = 0.05        # yaw toward the damaged side (extra drag)
EXPLOSION_SPIN = (1.2, 0.8, 1.5)  # impulsive (p, q, r) rates imparted (rad/s)
EXPLOSION_LIFT = 0.5        # lift left after structural break-up
EXPLOSION_CTRL = 0.2        # control effectiveness left after the blast

_ZERO3 = np.zeros(3)

def microburst_wind(tau):
    # NED wind during a microburst (tau = seconds since onset): a headwind that
    # builds, swings to a tailwind through a strong downdraft core, then eases.
    if tau < 0 or tau > MB_DURATION:
        return _ZERO3
    s = (tau - MB_CORE_T) / MB_WIDTH
    wind_north = MB_HEADWIND * math.tanh(s)              # headwind (-) then tailwind (+)
    wind_down = MB_DOWNDRAFT * math.exp(-((tau - MB_CORE_T) / 5.0) ** 2)
    return np.array([wind_north, 0.0, wind_down])

# flap/gear configurations: (zero-alpha lift, parasitic drag)
CONFIGS = {
    'clean':   dict(CL0=0.20, CD0=0.020),
    'takeoff': dict(CL0=0.90, CD0=0.045),
    'landing': dict(CL0=1.60, CD0=0.090),
}

# ISA atmosphere (troposphere)
rho0 = 1.225
def air_density(h):
    h = max(0.0, h)
    return rho0 * (1 - 2.25577e-5 * h) ** 4.25588

def stall_speed(h, config):
    cfg = CONFIGS[config]
    CL_max = cfg['CL0'] + CL_alpha * alpha_stall
    return math.sqrt(2 * mass * g / (air_density(h) * wing_area * CL_max))

# control gains
Kp_spd = 80_000.0       # autothrottle (N per m/s)
Ki_spd = 30_000.0
Ki_vs = 0.02            # climb-rate loop integral
Ke_p = 6.0              # elevator: pitch-attitude hold
Ke_d = 4.0
Ka_p = 2.5             # aileron: bank-attitude hold
Ka_d = 2.0
Kpsi = 0.6             # heading hold -> bank command
bank_max = math.radians(25)
Kr_b = 4.0             # rudder: sideslip null
Kr_d = 1.5             # rudder: yaw damper

DEFAULT_K_H = 0.15      # altitude -> climb-rate
DEFAULT_K_VS = 0.03     # climb-rate -> pitch command

# flight phases
PHASES = {
    'takeoff':  dict(y0=0.0,     V0=85.0,  V_target=95.0,  target=600.0,   vs_max=13.0, config='takeoff', dur=120),
    'climb':    dict(y0=3000.0,  V0=160.0, V_target=160.0, target=7000.0,  vs_max=15.0, config='clean',   dur=200),
    'cruise':   dict(y0=10000.0, V0=250.0, V_target=250.0, target=10000.0, vs_max=8.0,  config='clean',   dur=320),
    'approach': dict(y0=900.0,   V0=85.0,  V_target=85.0,  target=0.0,     vs_max=5.0,  config='landing', dur=220),
}

# quaternion helpers (q maps body -> world NED, z down)
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def rot_b2n(q, v):
    return qmul(qmul(q, np.array([0.0, *v])), qconj(q))[1:]

def rot_n2b(q, v):
    return qmul(qmul(qconj(q), np.array([0.0, *v])), q)[1:]

def euler_from_q(q):
    w, x, y, z = q
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    s = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.asin(s)
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw

def q_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

def simulate(K_h, K_vs, phase='takeoff', fail_engines=(), fail_time=None,
             accident=None, accident_side='left', record=False):
    P = PHASES[phase]
    config = P['config']
    cfg = CONFIGS[config]
    CL0, CD0 = cfg['CL0'], cfg['CD0']
    V_target = P['V_target']
    target = P['target']
    vs_max = P['vs_max']
    fset = set(fail_engines)

    # initial state: wings level, small nose-up trim, heading north
    q = q_from_euler(0.0, math.radians(4 if phase != 'cruise' else 2), 0.0)
    Vb = np.array([P['V0'], 0.0, 0.0])     # body velocity (u fwd, v right, w down)
    omega = np.array([0.0, 0.0, 0.0])      # body rates (p roll, q pitch, r yaw)
    posn = np.array([0.0, 0.0, -P['y0']])  # world NED position (z down)

    spd_integral = 0.0
    vs_integral = 0.0
    airborne = False
    total_error = overshoot = fast_climb = effort = 0.0
    prev_cmd = 0.0
    jam_elev = None
    exploded = False
    history = []

    t = 0.0
    while t < P['dur']:
        h = -posn[2]
        rho = air_density(h)
        u, v, w = Vb                       # inertial (ground) velocity, body frame
        # air-relative velocity (subtract any wind) drives the aerodynamics
        if accident == 'windshear' and fail_time is not None and t >= fail_time:
            wb = rot_n2b(q, microburst_wind(t - fail_time))
        else:
            wb = _ZERO3
        ua, va, wa = u - wb[0], v - wb[1], w - wb[2]
        V = max(math.sqrt(ua*ua + va*va + wa*wa), 1e-3)   # airspeed
        alpha = math.atan2(wa, ua)
        beta = math.asin(max(-1.0, min(1.0, va / V)))
        Q = 0.5 * rho * V * V
        roll, pitch, yaw = euler_from_q(q)

        # engines + autothrottle
        failed = fail_time is not None and t >= fail_time and fset
        working = [i for i in range(n_engines) if not (failed and (i + 1) in fset)]
        n_ok = len(working)
        avail_max = thrust_per_engine * n_ok
        avail_idle = idle_fraction * thrust_per_engine * n_ok
        spd_error = V_target - V
        spd_integral = spd_integral + spd_error * dt
        thrust_cmd = Kp_spd * spd_error + Ki_spd * spd_integral
        thrust_total = max(avail_idle, min(avail_max, thrust_cmd))
        if thrust_total != thrust_cmd:
            spd_integral = spd_integral - spd_error * dt
        per_engine = thrust_total / n_ok if n_ok else 0.0
        yaw_thrust = sum(-per_engine * engine_y[i] for i in working)   # moment about body z

        # guidance: altitude -> climb rate -> pitch command
        climb = -rot_b2n(q, Vb)[2]
        alt_error = target - h
        vs_cmd = max(-vs_max, min(vs_max, K_h * alt_error))
        vs_e = vs_cmd - climb
        pitch_cmd = K_vs * vs_e + Ki_vs * vs_integral
        pitch_cmd = max(-math.radians(15), min(math.radians(15), pitch_cmd))
        if abs(pitch_cmd) < math.radians(15):
            vs_integral = vs_integral + vs_e * dt

        # control laws (elevator pitch-hold, aileron heading/bank-hold, rudder sideslip+yaw damper)
        elevator = max(-elev_max, min(elev_max, Ke_p * (pitch - pitch_cmd) + Ke_d * omega[1]))
        roll_cmd = max(-bank_max, min(bank_max, Kpsi * (0.0 - yaw)))
        aileron = max(-ail_max, min(ail_max, Ka_p * (roll_cmd - roll) - Ka_d * omega[0]))
        rudder = max(-rud_max, min(rud_max, -Kr_b * beta + Kr_d * omega[2]))

        # accident effects (jam controls, add disturbance moments / lift loss)
        acc_on = accident is not None and fail_time is not None and t >= fail_time
        sgn = -1.0 if accident_side == 'left' else 1.0
        Cl_extra = Cm_extra = Cn_extra = 0.0
        lift_factor = 1.0
        if acc_on:
            if accident == 'rudder-hardover':
                rudder = sgn * rud_max
            elif accident == 'aileron-hardover':
                aileron = sgn * ail_max
            elif accident == 'elevator-jam':
                if jam_elev is None:
                    jam_elev = elevator           # freeze at the value when it jammed
                elevator = jam_elev
            elif accident == 'runaway-trim':
                Cm_extra = -min(RUNAWAY_TRIM_MAX, RUNAWAY_TRIM_RATE * (t - fail_time))
            elif accident == 'wing-loss':
                lift_factor = WING_LOSS_LIFT
                Cl_extra = sgn * WING_LOSS_ROLL
                Cn_extra = sgn * WING_LOSS_YAW
            elif accident == 'explosion':
                lift_factor = EXPLOSION_LIFT
                if not exploded:
                    omega = omega + sgn * np.array(EXPLOSION_SPIN)   # impulsive tumble
                    exploded = True
                elevator *= EXPLOSION_CTRL
                aileron *= EXPLOSION_CTRL
                rudder *= EXPLOSION_CTRL

        # aerodynamic coefficients (full-envelope longitudinal, linear lateral)
        sep = max(0.0, min(1.0, (abs(alpha) - alpha_stall) / math.radians(8)))
        CL_att = CL0 + CL_alpha * alpha
        CL = (1 - sep) * CL_att + sep * 2 * math.sin(alpha) * math.cos(alpha)
        CD = CD0 + (1 - sep) * k_induced * CL_att * CL_att + sep * CD_max * math.sin(alpha) ** 2
        CY = CY_beta * beta + CY_dr * rudder
        phat = omega[0] * span / (2 * V)
        qhat = omega[1] * chord / (2 * V)
        rhat = omega[2] * span / (2 * V)
        Cl = Cl_beta * beta + Cl_p * phat + Cl_r * rhat + (1 - sep) * Cl_da * aileron + Cl_dr * rudder + Cl_extra
        Cm = (1 - sep) * Cm_alpha * alpha + sep * Cm_unstable * math.sin(alpha) \
            + Cm_q * qhat + (1 - sep) * Cm_de * elevator + Cm_extra
        Cn = Cn_beta * beta + Cn_p * phat + Cn_r * rhat + Cn_da * aileron + Cn_dr * rudder + Cn_extra

        lift = Q * wing_area * CL * lift_factor
        drag = Q * wing_area * CD
        side = Q * wing_area * CY

        # wind -> body forces, plus thrust along body x
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)
        Fx = -drag * ca * cb - side * ca * sb + lift * sa + thrust_total
        Fy = -drag * sb + side * cb
        Fz = -drag * sa * cb - side * sa * sb - lift * ca
        # gravity in body frame
        gb = rot_n2b(q, np.array([0.0, 0.0, mass * g]))
        Fx += gb[0]; Fy += gb[1]; Fz += gb[2]

        # moments
        Lm = Q * wing_area * span * Cl
        Mm = Q * wing_area * chord * Cm
        Nm = Q * wing_area * span * Cn + yaw_thrust

        p, qrate, r = omega
        pdot = (Lm + (Iyy - Izz) * qrate * r) / Ixx
        qdot = (Mm + (Izz - Ixx) * p * r) / Iyy
        rdot = (Nm + (Ixx - Iyy) * p * qrate) / Izz
        udot = Fx / mass - (qrate * w - r * v)
        vdot = Fy / mass - (r * u - p * w)
        wdot = Fz / mass - (p * v - qrate * u)

        Vb = Vb + np.array([udot, vdot, wdot]) * dt
        omega = omega + np.array([pdot, qdot, rdot]) * dt
        q = q + 0.5 * qmul(q, np.array([0.0, *omega])) * dt
        q = q / np.linalg.norm(q)
        posn = posn + rot_b2n(q, Vb) * dt

        # score accumulators (nominal-climb quality, used by the gain search)
        total_error += abs(alt_error) * dt
        if h > target:
            overshoot += (h - target) * dt
        if climb > vs_max + 1.0:
            fast_climb += (climb - vs_max - 1.0) * dt
        effort += abs(pitch_cmd - prev_cmd)
        prev_cmd = pitch_cmd

        t += dt
        if record:
            history.append((t, posn[0], -posn[2], posn[1],
                            math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
                            math.degrees(alpha), math.degrees(beta), V,
                            stall_speed(h, config)))
        if h > P['y0'] + 20 or P['y0'] > 50:
            airborne = True
        if airborne and h <= 0:
            break

    score = total_error + 30.0 * overshoot + 40.0 * fast_climb + 20.0 * effort
    if record:
        return score, history
    return score

def write_flight(filename, history):
    cols = ['t', 'x', 'y', 'y_lat', 'roll', 'pitch', 'yaw', 'alpha', 'beta', 'airspeed', 'stall_speed']
    with open(filename, 'w') as f:
        f.write(','.join(cols) + '\n')
        for row in history:
            f.write(','.join(str(v) for v in row) + '\n')

def run_gain_search():
    best_score = float('inf')
    best = (DEFAULT_K_H, DEFAULT_K_VS)
    with open('results.csv', 'w') as f:
        f.write('K_h,K_vs,Score\n')
        for K_h in [0.06, 0.08, 0.10, 0.12, 0.15]:
            for K_vs in [0.02, 0.03, 0.04, 0.05, 0.07]:
                score = simulate(K_h, K_vs, phase='takeoff')
                f.write(f"{K_h},{K_vs},{score}\n")
                if score < best_score:
                    best_score, best = score, (K_h, K_vs)
    print(f'gain search best: K_h={best[0]}, K_vs={best[1]}, score={best_score:.1f}')
    return best

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Boeing 747-8 6-DOF flight / engine-failure simulator')
    ap.add_argument('--phase', choices=list(PHASES), default='takeoff',
                    help='flight phase the failure happens in (default: takeoff)')
    ap.add_argument('--fail-engines', default='',
                    help='comma list of engines 1-4 that fail, e.g. "1,2" (left side). default: none')
    ap.add_argument('--accident', choices=['none'] + ACCIDENTS, default='none',
                    help='other accident type to trigger (default: none)')
    ap.add_argument('--side', choices=['left', 'right'], default='left',
                    help='affected side / hardover direction for asymmetric accidents (default: left)')
    ap.add_argument('--fail-time', type=float, default=None, help='seconds into the run when the failure occurs')
    ap.add_argument('--out', default='flight.csv', help='output flight file (default: flight.csv)')
    ap.add_argument('--search', action='store_true', help='run the takeoff gain search first')
    args = ap.parse_args()

    K_h, K_vs = (run_gain_search() if args.search else (DEFAULT_K_H, DEFAULT_K_VS))

    fail_engines = tuple(sorted({int(e) for e in args.fail_engines.split(',') if e.strip()}))
    for e in fail_engines:
        if e < 1 or e > n_engines:
            raise SystemExit(f'engine {e} out of range 1-{n_engines}')
    accident = None if args.accident == 'none' else args.accident
    fail_time = args.fail_time
    if (fail_engines or accident) and fail_time is None:
        fail_time = {'takeoff': 12.0, 'climb': 20.0, 'cruise': 30.0, 'approach': 30.0}[args.phase]

    score, history = simulate(K_h, K_vs, phase=args.phase,
                              fail_engines=fail_engines, fail_time=fail_time,
                              accident=accident, accident_side=args.side, record=True)
    write_flight(args.out, history)

    P = PHASES[args.phase]
    last = history[-1]
    peak = max(r[2] for r in history)
    min_V = min(r[9] for r in history)
    max_bank = max(abs(r[4]) for r in history)
    crashed = last[2] <= 1 and last[0] < P['dur'] - 0.5
    print(f"phase={args.phase}  config={P['config']}")
    if fail_engines:
        side = 'left' if all(e <= 2 for e in fail_engines) else ('right' if all(e >= 3 for e in fail_engines) else 'both sides')
        print(f"engines failed: {list(fail_engines)} ({len(fail_engines)} of {n_engines}, {side}) "
              f"at t={fail_time:.0f}s -> {100*(n_engines-len(fail_engines))//n_engines}% thrust remaining")
    if accident:
        print(f"accident: {accident} ({args.side}) at t={fail_time:.0f}s")
    if not fail_engines and not accident:
        print('all systems nominal')
    print(f"peak altitude {peak:.0f} m, min airspeed {min_V:.0f} m/s, max bank {max_bank:.0f} deg, "
          f"final heading {last[6]:.0f} deg")
    print("OUTCOME: " + (f"ground impact at t={last[0]:.1f}s" if crashed else "still flying at end of window"))
    print(f"wrote {args.out}  (view with: python visualiser.py {args.out})")
