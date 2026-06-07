import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from simulator import alpha_stall

# read the flight
with open('flight.csv', 'r') as f:
    lines = f.readlines()[1:]
    times, xs, ys, commanded_pitches, actual_pitches = [], [], [], [], []
    for line in lines:
        t, x, y, cp, ap = line.strip().split(',')
        times.append(float(t))
        xs.append(float(x))
        ys.append(float(y))
        commanded_pitches.append(float(cp))
        actual_pitches.append(float(ap))

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
    # rotate each (x,y) about the origin by angle_rad
    out = []
    for px, py in points:
        rx = px * math.cos(angle_rad) - py * math.sin(angle_rad)
        ry = px * math.sin(angle_rad) + py * math.cos(angle_rad)
        out.append((rx, ry))
    return out

# two panels: big angle-of-attack view, small altitude strip
fig, (ax_plane, ax_alt) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={'width_ratios': [2, 1]})

# angle-of-attack panel:
ax_plane.set_xlim(-2, 2)
ax_plane.set_ylim(-2, 2)
ax_plane.set_aspect('equal')
ax_plane.set_title('angle of attack')
ax_plane.axhline(0, color='lightgray', linewidth=0.8)            # relative wind line
ax_plane.annotate('', xy=(0.55, 0.0), xytext=(1.9, 0.0),         # airflow arrow
                  arrowprops=dict(arrowstyle='->', color='lightgray'))
ax_plane.text(1.9, -0.18, 'relative wind', color='gray',
              ha='right', va='top', fontsize=9)

plane_patch, = ax_plane.fill([], [], color='tab:blue')
aoa_text = ax_plane.text(-1.9, 1.7, '', fontsize=11, va='top')

# altitude panel: the trace over time, with a moving dot
ax_alt.plot(times, ys, color='lightgray')
ax_alt.set_xlim(min(times), max(times))
ax_alt.set_ylim(min(ys) - 10, max(ys) + 10)
ax_alt.set_title('altitude')
ax_alt.set_xlabel('time (s)')
alt_dot, = ax_alt.plot([], [], 'o', color='tab:red')

step = 20
frames = range(0, len(times), step)

def draw_frame(i):
    aoa = aoas[i]
    # rotate the plane by its angle of attack, relative to the airflow
    pts = rotate(plane_shape, math.radians(aoa))
    plane_patch.set_xy(pts)

    # warn as the wing approaches the stall angle
    if aoa >= stall_deg:
        plane_patch.set_color('tab:red')
        aoa_text.set_text(f'AoA = {aoa:5.1f} deg   STALL')
        aoa_text.set_color('tab:red')
    elif aoa >= 0.8 * stall_deg:
        plane_patch.set_color('tab:orange')
        aoa_text.set_text(f'AoA = {aoa:5.1f} deg')
        aoa_text.set_color('tab:orange')
    else:
        plane_patch.set_color('tab:blue')
        aoa_text.set_text(f'AoA = {aoa:5.1f} deg')
        aoa_text.set_color('black')

    # move the dot along the altitude trace
    alt_dot.set_data([times[i]], [ys[i]])
    return plane_patch, alt_dot, aoa_text

anim = FuncAnimation(fig, draw_frame, frames=frames, interval=30, blit=False)
plt.show()
