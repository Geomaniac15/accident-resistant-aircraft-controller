import math

dt = 0.01

# large transport aircraft (737 / A320 class)
mass = 70_000.0          # kg
inertia = 4.3e6         # kg m^2, pitch axis (Iyy)
g = 9.81

# scenario: takeoff / initial climb-out to the acceleration altitude
target_altitude = 450.0     # m (~1500 ft)
V_liftoff = 75.0            # m/s, speed just after rotation
V_climb = 90.0             # m/s, climb speed the autothrottle holds (~175 kts)
vs_max = 13.0              # m/s, climb-rate limit (~2560 ft/min, comfortable)
pitch_limit = math.radians(15)

# aerodynamics (takeoff config, flaps extended) 
rho = 1.225            # air density
wing_area = 125.0      # reference wing area S
CL_alpha = 5.7         # lift-curve slope (per radian of angle of attack)
CL0 = 0.8              # flap lift at zero angle of attack (camber)
CD0 = 0.045            # parasitic drag (gear/flaps add drag at takeoff)
k_induced = 0.045      # induced-drag factor
alpha_stall = math.radians(15)

# stall speed: slowest the wing can still hold the weight (at max lift coefficient)
CL_max = CL0 + CL_alpha * alpha_stall
stall_speed = math.sqrt(2 * mass * g / (rho * wing_area * CL_max))   # ~63 m/s

# post-stall / departure behaviour (only active once the wing is stalled)
chord = 4.0            # mean aerodynamic chord (m), sets the moment arm
CD_max = 1.8           # broadside drag coefficient (fully separated)
Cm_stable = 0.6        # static stability while the flow is attached
Cm_damp = 4.0          # aerodynamic pitch damping while attached
Cm_unstable = 6.0      # divergence once stalled: this is what makes it tumble

def wrap_angle(a):
    # keep an angle in [-pi, pi] so the airframe can rotate all the way round
    return (a + math.pi) % (2 * math.pi) - math.pi

# inner pitch loop (sized for the large inertia: wn = 2 rad/s, zeta = 0.9) 
pitch_wn = 2.0
pitch_zeta = 0.9
Kp_pitch = inertia * pitch_wn * pitch_wn
Kd_pitch = 2 * pitch_zeta * inertia * pitch_wn

# autothrottle (holds climb speed) 
thrust_max = 220_000.0   # N, full takeoff thrust (two engines)
thrust_idle = 20_000.0
Kp_spd = 9_000.0
Ki_spd = 3_000.0

# climb-rate loop (commanded climb rate -> pitch) 
Ki_vs = 0.02

def aero_forces(vx, vy, pitch, pitch_rate):
    # full-envelope aero: behaves like a wing while attached, and like a
    # tumbling flat plate once the flow separates past the stall angle.
    # returns the x/y forces, the aerodynamic pitching moment, and the
    # separation fraction (0 = attached, 1 = fully stalled).
    V2 = vx * vx + vy * vy
    V = math.sqrt(V2)
    if V < 1e-6:
        return 0.0, 0.0, 0.0, 0.0

    gamma = math.atan2(vy, vx)             # flight-path angle (direction of travel)
    alpha = wrap_angle(pitch - gamma)      # angle of attack (full range)

    # separation: 0 below the stall angle, ramping to 1 over the next 8 degrees
    sep = max(0.0, min(1.0, (abs(alpha) - alpha_stall) / math.radians(8)))

    q = 0.5 * rho * V2                      # dynamic pressure
    CL_attached = CL0 + CL_alpha * alpha
    # blend the attached lift curve with a separated flat-plate curve
    CL = (1 - sep) * CL_attached + sep * 2 * math.sin(alpha) * math.cos(alpha)
    CD = CD0 + (1 - sep) * k_induced * CL_attached * CL_attached \
        + sep * CD_max * math.sin(alpha) * math.sin(alpha)

    lift = q * wing_area * CL
    drag = q * wing_area * CD
    fx = -drag * math.cos(gamma) - lift * math.sin(gamma)
    fy = lift * math.cos(gamma) - drag * math.sin(gamma)

    # pitching moment: stable and damped while attached, divergent once stalled
    Cm = (1 - sep) * (-Cm_stable * alpha) + sep * (Cm_unstable * math.sin(alpha))
    moment = q * wing_area * chord * Cm
    moment -= (1 - sep) * Cm_damp * q * wing_area * chord * chord \
        * pitch_rate / (2 * max(V, 1.0))
    return fx, fy, moment, sep

def simulate(K_h, K_vs, record=False, engine_fail_time=None):
    # autopilot cascade:
    #   altitude error -> commanded climb rate (limited to vs_max)   gain K_h
    #   climb-rate error -> commanded pitch (PI)                     gain K_vs, Ki_vs
    #   pitch error -> control torque (PD inner loop)
    # autothrottle separately holds the climb speed.
    pitch = 0.0
    pitch_rate = 0.0
    x = 0.0
    y = 0.0
    vx = V_liftoff
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
    while t < 120:
        V = math.hypot(vx, vy)

        # autothrottle: hold climb speed (with anti-windup on the thrust limits)
        spd_error = V_climb - V
        spd_integral = spd_integral + spd_error * dt
        thrust = Kp_spd * spd_error + Ki_spd * spd_integral
        thrust_clamped = max(thrust_idle, min(thrust_max, thrust))
        if thrust_clamped != thrust:
            spd_integral = spd_integral - spd_error * dt
            thrust = thrust_clamped

        # engine failure: total loss of thrust from this time on
        if engine_fail_time is not None and t >= engine_fail_time:
            thrust = 0.0

        # outer loop: altitude error -> commanded climb rate (limited)
        alt_error = target_altitude - y
        vs_cmd = max(-vs_max, min(vs_max, K_h * alt_error))

        # middle loop: climb-rate error -> commanded pitch (with anti-windup)
        vs_error = vs_cmd - vy
        commanded_pitch = K_vs * vs_error + Ki_vs * vs_integral
        cp_clamped = max(-pitch_limit, min(pitch_limit, commanded_pitch))
        if cp_clamped == commanded_pitch:
            vs_integral = vs_integral + vs_error * dt
        commanded_pitch = cp_clamped

        aero_fx, aero_fy, aero_moment, sep = aero_forces(vx, vy, pitch, pitch_rate)

        # inner loop: pitch -> control torque. Elevator authority fades as the
        # airspeed decays (q_ratio) and vanishes when the tail is in separated
        # flow (1 - sep), so a deep stall leaves the airframe uncontrollable.
        q_ratio = min(1.0, (V / V_liftoff) ** 2)
        authority = q_ratio * (1 - sep)
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
        if y > target_altitude:
            overshoot = overshoot + (y - target_altitude) * dt
        if vy > vs_max + 1.0:
            fast_climb = fast_climb + (vy - vs_max - 1.0) * dt   # discourage steep climb

        effort = effort + abs(commanded_pitch - prev_commanded)
        prev_commanded = commanded_pitch

        if record:
            airspeed = math.hypot(vx, vy)
            history.append((t, x, y, math.degrees(commanded_pitch), math.degrees(pitch), airspeed))
        t = t + dt

        # stop the run when the aircraft hits the ground (after it has flown)
        if y > 20:
            airborne = True
        if airborne and y <= 0:
            break

    # reward a smooth, comfortable, on-target climb
    score = total_error + 30.0 * overshoot + 40.0 * fast_climb + 20.0 * effort

    if record:
        return score, history
    return score

if __name__ == '__main__':
    with open('results.csv', 'w') as f:
        f.write('K_h,K_vs,Score\n')

        # the search
        best_score = float('inf')
        best_gains = None

        for K_h in [0.06, 0.08, 0.10, 0.12, 0.15]:
            for K_vs in [0.02, 0.03, 0.04, 0.05, 0.07]:
                score = simulate(K_h, K_vs)
                f.write(f"{K_h},{K_vs},{score}\n")
                print(f"K_h: {K_h:.3f}, K_vs: {K_vs:.3f}, Score: {score:.2f}")
                if score < best_score:
                    best_score = score
                    best_gains = (K_h, K_vs)

    print(f'\nBEST: K_h={best_gains[0]}, K_vs={best_gains[1]}, score={best_score:.1f}')
    print(f'stall speed = {stall_speed:.1f} m/s')

    def write_flight(filename, history):
        with open(filename, 'w') as f:
            f.write('t,x,y,commanded_pitch,actual_pitch,airspeed\n')
            for row in history:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")

    # nominal climb
    score, history = simulate(best_gains[0], best_gains[1], record=True)
    write_flight('flight.csv', history)

    # same climb, but both engines fail at t = 12 s, shortly after takeoff:
    # the autopilot keeps pulling for altitude, stalls, departs and tumbles in
    score, history = simulate(best_gains[0], best_gains[1], record=True, engine_fail_time=12.0)
    write_flight('flight_failure.csv', history)
    print('wrote flight.csv (nominal) and flight_failure.csv (engine failure at t=12s)')
