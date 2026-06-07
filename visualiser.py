import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Play back a recorded flight.
#   python visualiser.py flight.csv            # 2D instrument panels (default)
#   python visualiser.py flight.csv --3d       # 3D path + aircraft orientation
ap = argparse.ArgumentParser(description='flight playback')
ap.add_argument('file', nargs='?', default='flight.csv', help='flight CSV to play (default: flight.csv)')
ap.add_argument('--3d', dest='three_d', action='store_true', help='show the 3D view instead of the panels')
args = ap.parse_args()
filename = args.file

with open(filename, 'r') as f:
    data = [[float(v) for v in line.strip().split(',')] for line in f.readlines()[1:] if line.strip()]
cols = list(zip(*data))
times, xs, alts, lats, rolls, pitches, yaws, alphas, betas, speeds, stalls = cols

stalled = [abs(a) >= 15 for a in alphas]

def body_to_world(roll_deg, pitch_deg, yaw_deg):
    # body->NED rotation (standard ZYX aerospace Euler angles)
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
        [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
        [-sp,   sr*cp,            cr*cp],
    ])

def run_2d():
    rear_shape = [(-1.0, 0.0), (1.0, 0.0)]
    rear_fin = [(0.0, 0.0), (0.0, 0.45)]
    top_shape = [(1.2, 0.0), (-0.2, 0.9), (-0.1, 0.15), (-0.9, 0.5),
                 (-1.0, 0.0), (-0.9, -0.5), (-0.1, -0.15), (-0.2, -0.9)]

    def rot(points, ang):
        c, s = math.cos(ang), math.sin(ang)
        return [(px*c - py*s, px*s + py*c) for px, py in points]

    fig, axd = plt.subplot_mosaic([['bank', 'track'], ['alt', 'spd']],
                                  figsize=(12, 7), layout='constrained')
    fig.suptitle(filename)
    ax_bank, ax_track, ax_alt, ax_spd = axd['bank'], axd['track'], axd['alt'], axd['spd']

    ax_bank.set_xlim(-1.6, 1.6); ax_bank.set_ylim(-1.6, 1.6); ax_bank.set_aspect('equal')
    ax_bank.set_title('bank  (view from behind)')
    ax_bank.axhline(0, color='lightgray', linewidth=0.8)
    wing_line, = ax_bank.plot([], [], color='tab:blue', linewidth=4)
    fin_line, = ax_bank.plot([], [], color='tab:blue', linewidth=2)
    bank_text = ax_bank.text(-1.5, 1.4, '', fontsize=10, va='top')

    ax_track.plot(xs, lats, color='lightgray')
    ax_track.plot([xs[0]], [lats[0]], '^', color='gray', markersize=7)
    ax_track.set_aspect('equal'); ax_track.set_title('ground track  (top-down)')
    ax_track.set_xlabel('downrange (m)'); ax_track.set_ylabel('lateral (m)')
    plane_top, = ax_track.fill([], [], color='tab:blue')
    span_track = max(max(xs) - min(xs), max(lats) - min(lats), 1.0)
    plane_scale = 0.04 * span_track

    ax_alt.plot(times, alts, color='lightgray')
    ax_alt.axhline(0, color='saddlebrown', linewidth=0.8)
    ax_alt.set_xlim(min(times), max(times)); ax_alt.set_ylim(min(min(alts) - 10, -10), max(alts) + 10)
    ax_alt.set_title('altitude (m)'); ax_alt.set_xlabel('time (s)')
    alt_dot, = ax_alt.plot([], [], 'o', color='tab:red')

    ax_spd.plot(times, speeds, color='lightgray')
    ax_spd.plot(times, stalls, color='tab:red', linestyle='--', linewidth=1.0)
    ax_spd.text(times[-1], stalls[-1], ' stall', color='tab:red', va='bottom', ha='right', fontsize=8)
    ax_spd.set_xlim(min(times), max(times)); ax_spd.set_ylim(min(min(speeds), min(stalls)) - 8, max(speeds) + 8)
    ax_spd.set_title('airspeed (m/s)'); ax_spd.set_xlabel('time (s)')
    spd_dot, = ax_spd.plot([], [], 'o', color='tab:red')

    step = max(1, len(times) // 600)

    def draw(i):
        roll = math.radians(rolls[i])
        wing = rot(rear_shape, roll); fin = rot(rear_fin, roll)
        wing_line.set_data([p[0] for p in wing], [p[1] for p in wing])
        fin_line.set_data([p[0] for p in fin], [p[1] for p in fin])
        colour = 'tab:red' if stalled[i] else ('tab:orange' if abs(rolls[i]) > 60 else 'tab:blue')
        wing_line.set_color(colour); fin_line.set_color(colour)
        bank_text.set_text(f'roll  {rolls[i]:6.0f}\npitch {pitches[i]:6.0f}\nyaw   {yaws[i]:6.0f}\n'
                           f'AoA   {alphas[i]:5.1f}\nsideslip {betas[i]:5.1f}')
        bank_text.set_color('tab:red' if stalled[i] else 'black')
        heading = math.radians(yaws[i])
        pts = rot([(px*plane_scale, py*plane_scale) for px, py in top_shape], heading)
        plane_top.set_xy([(xs[i] + px, lats[i] + py) for px, py in pts])
        plane_top.set_color(colour)
        alt_dot.set_data([times[i]], [alts[i]])
        spd_dot.set_data([times[i]], [speeds[i]]); spd_dot.set_color('tab:red' if speeds[i] < stalls[i] else 'tab:green')
        return wing_line, fin_line, plane_top, alt_dot, spd_dot, bank_text

    global _anim
    _anim = FuncAnimation(fig, draw, frames=range(0, len(times), step), interval=30, blit=False)
    plt.show()

def run_3d():
    # aircraft model in body axes (x fwd, y right, z down), as line segments
    L, W = 1.0, 1.1
    parts = {
        'fuse': [(L, 0, 0), (-L, 0, 0)],
        'wing': [(0.15, -W, 0), (0.15, W, 0)],
        'htail': [(-0.85, -0.45, 0), (-0.85, 0.45, 0)],
        'vtail': [(-0.85, 0, 0), (-1.0, 0, -0.5)],
    }

    # scene extents (downrange x, lateral y, altitude z), padded and kept non-degenerate
    rx = max(xs) - min(xs); ry = max(lats) - min(lats); rz = max(alts) - min(alts)
    span = max(rx, ry, rz, 1.0)
    pad = 0.08 * span
    def lims(lo, hi, rng):
        if rng < 0.2 * span:                      # keep thin dimensions visible
            mid = 0.5 * (lo + hi); lo, hi = mid - 0.1 * span, mid + 0.1 * span
        return lo - pad, hi + pad
    xlo, xhi = lims(min(xs), max(xs), rx)
    ylo, yhi = lims(min(lats), max(lats), ry)
    zlo, zhi = lims(min(min(alts), 0.0), max(alts), rz)
    glyph = 0.06 * span

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection='3d')
    fig.suptitle(filename + '   (3D)')
    ax.plot(xs, lats, alts, color='lightgray', linewidth=1)             # full path
    ax.plot(xs, lats, [zlo] * len(xs), color='0.85', linewidth=0.8)     # ground shadow
    ax.scatter([xs[0]], [lats[0]], [alts[0]], color='gray', marker='^')
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
    ax.set_box_aspect((xhi - xlo, yhi - ylo, zhi - zlo))
    ax.set_xlabel('downrange (m)'); ax.set_ylabel('lateral (m)'); ax.set_zlabel('altitude (m)')

    trail, = ax.plot([], [], [], color='tab:blue', linewidth=2)
    seg_lines = {name: ax.plot([], [], [], color='tab:blue', linewidth=3 if name != 'vtail' else 2)[0]
                 for name in parts}
    info = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, va='top')

    step = max(1, len(times) // 400)

    def draw(i):
        R = body_to_world(rolls[i], pitches[i], yaws[i])
        pos = np.array([xs[i], lats[i], alts[i]])
        colour = 'tab:red' if stalled[i] else ('tab:orange' if abs(rolls[i]) > 60 else 'tab:blue')
        for name, pts in parts.items():
            world = []
            for pb in pts:
                ned = R @ (np.array(pb) * glyph)        # body -> NED
                world.append(pos + np.array([ned[0], ned[1], -ned[2]]))  # NED down -> altitude up
            world = np.array(world)
            seg_lines[name].set_data(world[:, 0], world[:, 1])
            seg_lines[name].set_3d_properties(world[:, 2])
            seg_lines[name].set_color(colour)
        trail.set_data(xs[:i + 1], lats[:i + 1]); trail.set_3d_properties(alts[:i + 1])
        info.set_text(f't={times[i]:5.1f}s   alt={alts[i]:6.0f} m   V={speeds[i]:4.0f} m/s\n'
                      f'roll={rolls[i]:5.0f}  pitch={pitches[i]:5.0f}  yaw={yaws[i]:5.0f}  AoA={alphas[i]:4.1f}')
        info.set_color('tab:red' if stalled[i] else 'black')
        return tuple(seg_lines.values()) + (trail, info)

    global _anim
    _anim = FuncAnimation(fig, draw, frames=range(0, len(times), step), interval=30, blit=False)
    plt.show()

if args.three_d:
    run_3d()
else:
    run_2d()
