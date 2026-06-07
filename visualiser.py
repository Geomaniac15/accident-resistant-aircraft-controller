import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from simulator import simulate

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

# the plane's shape, drawn pointing right (nose at +x), centred on (0,0)
# a simple triangle: nose, and two tail corners
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

# two panels: big attitude view, small altitude strip
fig, (ax_plane, ax_alt) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={'width_ratios': [2, 1]})

# attitude panel: fixed square, plane sits centred and only rotates
ax_plane.set_xlim(-2, 2)
ax_plane.set_ylim(-2, 2)
ax_plane.set_aspect('equal')   # now rotation looks true, no squashing
ax_plane.set_title('attitude')
ax_plane.axhline(0, color='lightgray', linewidth=0.8)  # horizon reference

plane_patch, = ax_plane.fill([], [], color='tab:red')

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
    # rotate the plane shape to the actual pitch and draw it centred
    angle = math.radians(actual_pitches[i])
    pts = rotate(plane_shape, angle)
    plane_patch.set_xy(pts)

    # move the dot along the altitude trace
    alt_dot.set_data([times[i]], [ys[i]])
    return plane_patch, alt_dot

anim = FuncAnimation(fig, draw_frame, frames=frames, interval=30, blit=False)
plt.show()