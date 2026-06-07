import matplotlib.pyplot as plt
import math

dt = 0.01
inertia = 5.0   # moment of inertia: resistance to being spun (rotational 'mass')

pitch = 0.0         # angle in radians, 0 = level
pitch_rate = 0.0    # angular velocity: how fast it's pitching

times = []
pitches = []
torques = []

t = 0.0
while t < 10:
    # command a torque for the first 2 seconds, then let go
    if t < 2.0:
        torque = 1.0
    elif t < 4.0:
        torque = -1.0
    else:
        torque = 0.0

    # torque to angular velocity to angle
    angular_acceleration = torque / inertia
    pitch_rate = pitch_rate + angular_acceleration * dt
    pitch = pitch + pitch_rate * dt

    times.append(t)
    pitches.append(math.degrees(pitch))   # store as degrees, easier to read
    torques.append(torque)

    t = t + dt

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(times, pitches, label='pitch angle')
ax1.set_ylabel('pitch (degrees)')
ax1.legend()

ax2.plot(times, torques, label='torque', color='tab:green')
ax2.set_ylabel('torque (N·m)')
ax2.set_xlabel('time (s)')
ax2.legend()

plt.show()