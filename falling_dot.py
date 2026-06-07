import matplotlib.pyplot as plt
import math

dt = 0.01
inertia = 5.0   # moment of inertia: resistance to being spun (rotational 'mass')
mass = 1.0
g = 9.81

pitch = 0.0
pitch_rate = 0.0

x = 0.0
y = 0.0
vx = 0.0
vy = 0.0

target_pitch = math.radians(30) 
thrust = 15.0

Kp = 8.0
Kd = 10.0

times = []
xs = []
ys = []
pitches = []

t = 0.0
while t < 10:
    error = target_pitch - pitch
    torque = Kp * error - Kd * pitch_rate
    angular_acceleration = torque / inertia
    pitch_rate = pitch_rate + angular_acceleration * dt
    pitch = pitch + pitch_rate * dt

    thrust_x = thrust * math.cos(pitch)
    thrust_y = thrust * math.sin(pitch)

    fx = thrust_x
    fy = thrust_y - mass * g

    vx = vx + (fx / mass) * dt
    vy = vy + (fy / mass) * dt
    x = x + vx * dt
    y = y + vy * dt

    times.append(t)
    xs.append(x)
    ys.append(y)
    pitches.append(math.degrees(pitch))

    t = t + dt

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(xs, ys, label='flight path')
ax1.set_xlabel('x position (m)')
ax1.set_ylabel('y position (altitude, m)')
ax1.legend()
ax1.axis('equal')

ax2.plot(times, pitches, label='pitch', color='tab:orange')
ax2.set_ylabel('pitch (degrees)')
ax2.set_xlabel('time (s)')
ax2.legend()

plt.show()