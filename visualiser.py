import sys
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# which flight to play (default nominal):
#   python visualiser.py flight.csv
#   python visualiser.py allout_cruise.csv
filename = sys.argv[1] if len(sys.argv) > 1 else 'flight.csv'

with open(filename, 'r') as f:
    rows = [line.strip().split(',') for line in f.readlines()[1:] if line.strip()]
cols = list(zip(*[[float(v) for v in r] for r in rows]))
times, xs, alts, lats, rolls, pitches, yaws, alphas, betas, speeds, stalls = cols

# aircraft silhouettes
# rear view (looking forward): wings, fuselage, fin
rear_shape = [(-1.0, 0.0), (1.0, 0.0)]          # wing line
rear_fin = [(0.0, 0.0), (0.0, 0.45)]            # vertical tail
# top view (looking down): nose forward (+x), swept wings, tail
top_shape = [(1.2, 0.0), (-0.2, 0.9), (-0.1, 0.15), (-0.9, 0.5),
             (-1.0, 0.0), (-0.9, -0.5), (-0.1, -0.15), (-0.2, -0.9)]

def rotate(points, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(px * c - py * s, px * s + py * c) for px, py in points]

fig, axd = plt.subplot_mosaic([['bank', 'track'], ['alt', 'spd']],
                              figsize=(12, 7), layout='constrained')
fig.suptitle(filename)
ax_bank, ax_track, ax_alt, ax_spd = axd['bank'], axd['track'], axd['alt'], axd['spd']

# bank (rear view) panel
ax_bank.set_xlim(-1.6, 1.6)
ax_bank.set_ylim(-1.6, 1.6)
ax_bank.set_aspect('equal')
ax_bank.set_title('bank  (view from behind)')
ax_bank.axhline(0, color='lightgray', linewidth=0.8)   # horizon
wing_line, = ax_bank.plot([], [], color='tab:blue', linewidth=4)
fin_line, = ax_bank.plot([], [], color='tab:blue', linewidth=2)
bank_text = ax_bank.text(-1.5, 1.4, '', fontsize=10, va='top')

# ground-track (top-down) panel: downrange vs lateral
ax_track.plot(xs, lats, color='lightgray')
ax_track.plot([xs[0]], [lats[0]], '^', color='gray', markersize=7)   # start
ax_track.set_aspect('equal')
ax_track.set_title('ground track  (top-down)')
ax_track.set_xlabel('downrange (m)')
ax_track.set_ylabel('lateral (m)')
track_marker, = ax_track.plot([], [], color='none')
plane_top, = ax_track.fill([], [], color='tab:blue')
span_track = max(max(xs) - min(xs), max(lats) - min(lats), 1.0)
plane_scale = 0.04 * span_track

# altitude panel
ax_alt.plot(times, alts, color='lightgray')
ax_alt.axhline(0, color='saddlebrown', linewidth=0.8)
ax_alt.set_xlim(min(times), max(times))
ax_alt.set_ylim(min(min(alts) - 10, -10), max(alts) + 10)
ax_alt.set_title('altitude (m)')
ax_alt.set_xlabel('time (s)')
alt_dot, = ax_alt.plot([], [], 'o', color='tab:red')

# airspeed panel with the (altitude-dependent) stall reference
ax_spd.plot(times, speeds, color='lightgray')
ax_spd.plot(times, stalls, color='tab:red', linestyle='--', linewidth=1.0)
ax_spd.text(times[-1], stalls[-1], ' stall', color='tab:red', va='bottom', ha='right', fontsize=8)
ax_spd.set_xlim(min(times), max(times))
ax_spd.set_ylim(min(min(speeds), min(stalls)) - 8, max(speeds) + 8)
ax_spd.set_title('airspeed (m/s)')
ax_spd.set_xlabel('time (s)')
spd_dot, = ax_spd.plot([], [], 'o', color='tab:red')

step = max(1, len(times) // 600)
frames = range(0, len(times), step)

def draw_frame(i):
    roll = math.radians(rolls[i])
    # rear view: bank the aircraft; colour by AoA stall / inverted
    wing = rotate(rear_shape, roll)
    fin = rotate(rear_fin, roll)
    wing_line.set_data([p[0] for p in wing], [p[1] for p in wing])
    fin_line.set_data([p[0] for p in fin], [p[1] for p in fin])
    stalled = abs(alphas[i]) >= 15
    colour = 'tab:red' if stalled else ('tab:orange' if abs(rolls[i]) > 60 else 'tab:blue')
    wing_line.set_color(colour)
    fin_line.set_color(colour)
    bank_text.set_text(f'roll  {rolls[i]:6.0f}\npitch {pitches[i]:6.0f}\nyaw   {yaws[i]:6.0f}\nAoA   {alphas[i]:5.1f}\nsideslip {betas[i]:5.1f}')
    bank_text.set_color('tab:red' if stalled else 'black')

    # top-down: plane glyph at current position pointing along heading
    heading = math.radians(yaws[i])
    pts = rotate([(px * plane_scale, py * plane_scale) for px, py in top_shape], heading)
    plane_top.set_xy([(xs[i] + px, lats[i] + py) for px, py in pts])
    plane_top.set_color(colour)

    alt_dot.set_data([times[i]], [alts[i]])
    spd_dot.set_data([times[i]], [speeds[i]])
    spd_dot.set_color('tab:red' if speeds[i] < stalls[i] else 'tab:green')
    return wing_line, fin_line, plane_top, alt_dot, spd_dot, bank_text

anim = FuncAnimation(fig, draw_frame, frames=frames, interval=30, blit=False)
plt.show()
