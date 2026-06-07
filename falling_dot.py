import matplotlib.pyplot as plt
import math

dt = 0.01
inertia = 1.0   # moment of inertia: resistance to being spun (rotational 'mass')
mass = 1.0
g = 9.81

pitch = 0.0
pitch_rate = 0.0

x = 0.0
y = 0.0
vx = 0.0
vy = 0.0

# inner loop: pitch control
Kp_pitch = 4.0
Kd_pitch = 2.0

# outer loop: altitude control
Kp_alt = 0.02
Kd_alt = 0.05

thrust = 20.0
target_altitude = 100.0

times = []
xs = []
ys = []
pitches = []
target_pitches = []

t = 0.0
while t < 60:
    # Altitude control
    alt_error = target_altitude - y
    # too low means ask for nose-up, too high means nose-down
    # derive off vy (the measurement's rate) to damp, same trick as before
    commanded_pitch = Kp_alt * alt_error - Kd_alt * vy

    commanded_pitch = max(math.radians(-30), min(math.radians(30), commanded_pitch))

    # Hold pitch
    error = commanded_pitch - pitch
    torque = Kp_pitch * error - Kd_pitch * pitch_rate
    angular_acceleration = torque / inertia
    pitch_rate = pitch_rate + angular_acceleration * dt
    pitch = pitch + pitch_rate * dt

    fx = thrust * math.cos(pitch)
    fy = thrust * math.sin(pitch)

    vx = vx + (fx / mass) * dt
    vy = vy + (fy / mass) * dt
    x = x + vx * dt
    y = y + vy * dt

    times.append(t)
    xs.append(x)
    ys.append(y)
    pitches.append(math.degrees(pitch))
    target_pitches.append(math.degrees(commanded_pitch))

    t = t + dt

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(xs, ys, label='flight path')
ax1.axhline(target_altitude, color='gray', linestyle='--', label='target altitude')
ax1.set_xlabel('x position (m)')
ax1.set_ylabel('altitude (m)')
ax1.legend()

ax2.plot(times, pitches, label='actual pitch')
ax2.plot(times, target_pitches, label='commanded pitch', linestyle='--')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('pitch (degrees)')
ax2.legend()

plt.show()