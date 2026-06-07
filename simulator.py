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

def aero_forces(vx, vy, pitch):
    # lift acts perpendicular to the velocity vector, drag opposes it
    V2 = vx * vx + vy * vy
    V = math.sqrt(V2)
    if V < 1e-6:
        return 0.0, 0.0

    gamma = math.atan2(vy, vx)        # flight-path angle (direction of travel)
    alpha = pitch - gamma             # angle of attack
    alpha = max(-alpha_stall, min(alpha_stall, alpha))   # stall clamp

    q = 0.5 * rho * V2                # dynamic pressure
    CL = CL0 + CL_alpha * alpha
    CD = CD0 + k_induced * CL * CL

    lift = q * wing_area * CL
    drag = q * wing_area * CD

    fx = -drag * math.cos(gamma) - lift * math.sin(gamma)
    fy = lift * math.cos(gamma) - drag * math.sin(gamma)
    return fx, fy

def simulate(K_h, K_vs, record=False):
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

        # inner loop: pitch -> control torque
        error = commanded_pitch - pitch
        torque = Kp_pitch * error - Kd_pitch * pitch_rate
        pitch_rate = pitch_rate + (torque / inertia) * dt
        pitch = pitch + pitch_rate * dt

        aero_fx, aero_fy = aero_forces(vx, vy, pitch)
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
            history.append((t, x, y, math.degrees(commanded_pitch), math.degrees(pitch)))
        t = t + dt

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

    score, history = simulate(best_gains[0], best_gains[1], record=True)
    with open('flight.csv', 'w') as f:
        f.write('t,x,y,commanded_pitch,actual_pitch\n')
        for row in history:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")
