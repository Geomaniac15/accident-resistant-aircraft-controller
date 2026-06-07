import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from simulator import alpha_stall, stall_speed

# which flight to play: default nominal, pass a file for others
#   python visualiser.py flight_failure.csv
filename = sys.argv[1] if len(sys.argv) > 1 else 'flight.csv'

# read the flight
with open(filename, 'r') as f:
    lines = f.readlines()[1:]
    times, xs, ys, commanded_pitches, actual_pitches, airspeeds = [], [], [], [], [], []
    for line in lines:
        t, x, y, cp, ap, v = line.strip().split(',')
        times.append(float(t))
        xs.append(float(x))
        ys.append(float(y))
        commanded_pitches.append(float(cp))
        actual_pitches.append(float(ap))
        airspeeds.append(float(v))

# angle of attack = body pitch - flight-path angle.
aoas = []
for i in range(len(times)):
    j = min(i + 1, len(times) - 1)
    k = max(i - 1, 0)
    vx = (xs[j] - xs[k]) / (times[j] - times[k] + 1e-9)
    vy = (ys[j] - ys[k]) / (times[j] - times[k] + 1e-9)
    gamma = math.degrees(math.atan2(vy, vx))   # flight-path angle
    aoas.append(actual_pitches[i] - gamma)

stall_deg = math.degrees(alpha_stall)

# the plane's shape
plane_shape = [
    (1.0, 0.0),    # nose
    (-0.6, 0.4),   # top tail
    (-0.3, 0.0),   # tail notch
    (-0.6, -0.4),  # bottom tail
]

def rotate(points, angle_rad):
    out = []
    for px, py in points:
        rx = px * math.cos(angle_rad) - py * math.sin(angle_rad)
        ry = px * math.sin(angle_rad) + py * math.cos(angle_rad)
        out.append((rx, ry))
    return out

# layout: big angle-of-attack view on the left, altitude and airspeed stacked right
fig, axd = plt.subplot_mosaic([['aoa', 'alt'], ['aoa', 'spd']],
                              figsize=(12, 5), layout='constrained',
                              gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle(filename)
ax_plane, ax_alt, ax_spd = axd['aoa'], axd['alt'], axd['spd']

# angle-of-attack panel: view aligned to the airflow, plane tilts by the AoA
ax_plane.set_xlim(-2, 2)
ax_plane.set_ylim(-2, 2)
ax_plane.set_aspect('equal')
ax_plane.set_title('angle of attack')
ax_plane.axhline(0, color='lightgray', linewidth=0.8)            # relative wind line
ax_plane.annotate('', xy=(0.55, 0.0), xytext=(1.9, 0.0),
                  arrowprops=dict(arrowstyle='->', color='lightgray'))
ax_plane.text(1.9, -0.18, 'relative wind', color='gray', ha='right', va='top', fontsize=9)
plane_patch, = ax_plane.fill([], [], color='tab:blue')
aoa_text = ax_plane.text(-1.9, 1.7, '', fontsize=11, va='top')

# altitude panel
ax_alt.plot(times, ys, color='lightgray')
ax_alt.axhline(0, color='saddlebrown', linewidth=0.8)   # the ground
ax_alt.set_xlim(min(times), max(times))
ax_alt.set_ylim(min(min(ys) - 10, -10), max(ys) + 10)
ax_alt.set_title('altitude (m)')
alt_dot, = ax_alt.plot([], [], 'o', color='tab:red')

# airspeed panel, with the stall-speed reference line
ax_spd.plot(times, airspeeds, color='lightgray')
ax_spd.axhline(stall_speed, color='tab:red', linestyle='--', linewidth=1.0)
ax_spd.text(times[-1], stall_speed, f' stall {stall_speed:.0f}', color='tab:red',
            va='bottom', ha='right', fontsize=8)
ax_spd.set_xlim(min(times), max(times))
ax_spd.set_ylim(min(min(airspeeds), stall_speed) - 8, max(airspeeds) + 8)
ax_spd.set_title('airspeed (m/s)')
ax_spd.set_xlabel('time (s)')
spd_dot, = ax_spd.plot([], [], 'o', color='tab:red')

step = 20
frames = range(0, len(times), step)

def draw_frame(i):
    aoa = aoas[i]
    plane_patch.set_xy(rotate(plane_shape, math.radians(aoa)))

    # warn as the wing approaches the stall angle (either direction, for tumbles)
    if abs(aoa) >= stall_deg:
        plane_patch.set_color('tab:red')
        aoa_text.set_text(f'AoA = {aoa:6.1f} deg   STALL')
        aoa_text.set_color('tab:red')
    elif abs(aoa) >= 0.8 * stall_deg:
        plane_patch.set_color('tab:orange')
        aoa_text.set_text(f'AoA = {aoa:6.1f} deg')
        aoa_text.set_color('tab:orange')
    else:
        plane_patch.set_color('tab:blue')
        aoa_text.set_text(f'AoA = {aoa:6.1f} deg')
        aoa_text.set_color('black')

    alt_dot.set_data([times[i]], [ys[i]])

    # airspeed dot turns red below stall speed
    spd_dot.set_data([times[i]], [airspeeds[i]])
    spd_dot.set_color('tab:red' if airspeeds[i] < stall_speed else 'tab:green')

    return plane_patch, alt_dot, spd_dot, aoa_text

anim = FuncAnimation(fig, draw_frame, frames=frames, interval=30, blit=False)
plt.show()
