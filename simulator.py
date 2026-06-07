import math

dt = 0.01
inertia = 5.0  
mass = 1.0
g = 9.81

Kp_pitch = 4.0
Kd_pitch = 2.0
thrust = 20.0
target_altitude = 100.0

def simulate(Kp_alt, Kd_alt, record=False):
    pitch = 0.0
    pitch_rate = 0.0
    x = 0.0
    y = 0.0
    vx = 0.0
    vy = 0.0

    total_error = 0.0
    history = []

    t = 0.0
    while t < 60:
        alt_error = target_altitude - y
        commanded_pitch = Kp_alt * alt_error - Kd_alt * vy
        commanded_pitch = max(math.radians(-30), min(math.radians(30), commanded_pitch))

        error = commanded_pitch - pitch
        torque = Kp_pitch * error - Kd_pitch * pitch_rate
        pitch_rate = pitch_rate + (torque / inertia) * dt
        pitch = pitch + pitch_rate * dt

        fx = thrust * math.cos(pitch)
        fy = thrust * math.sin(pitch) - mass * g

        vx = vx + (fx / mass) * dt
        vy = vy + (fy / mass) * dt
        x = x + vx * dt
        y = y + vy * dt

        total_error = total_error + abs(alt_error) * dt

        if record:
            history.append((t, x, y, math.degrees(commanded_pitch), math.degrees(pitch)))

        t = t + dt
    
    if record:
        return total_error, history
    return total_error

if __name__ == "__main__":
    with open('results.csv', 'w') as f:
        f.write('Kp_alt,Kd_alt,Score\n')

        # the search
        best_score = float('inf')
        best_gains = None

        for Kp_alt in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
            for Kd_alt in [0.0, 0.02, 0.05, 0.08, 0.12, 0.2]:
                score = simulate(Kp_alt, Kd_alt)
                f.write(f"{Kp_alt},{Kd_alt},{score}\n")
                print(f"Kp_alt: {Kp_alt:.3f}, Kd_alt: {Kd_alt:.3f}, Score: {score:.2f}")
                if score < best_score:
                    best_score = score
                    best_gains = (Kp_alt, Kd_alt)

    print(f'\nBEST: Kp_alt={best_gains[0]}, Kd_alt={best_gains[1]}, score={best_score:.1f}')

    score, history = simulate(0.01, 0.0, record=True)
    with open('flight.csv', 'w') as f:
        f.write('t,x,y,commanded_pitch,actual_pitch\n')
        for row in history:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

    