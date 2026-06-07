import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# read the flight data
with open('flight.csv', 'r') as f:
    lines = f.readlines()[1:]  # skip header

    times = []
    xs = []
    ys = []
    commanded_pitches = []
    actual_pitches = []

    for line in lines:
        t, x, y, commanded_pitch, actual_pitch = line.strip().split(',')
        times.append(float(t))
        xs.append(float(x))
        ys.append(float(y))
        commanded_pitches.append(float(commanded_pitch))
        actual_pitches.append(float(actual_pitch))

# set up the figure
fig, ax = plt.subplots()
ax.set_xlim(min(xs), max(xs))
ax.set_ylim(min(ys) - 10, max(ys) + 10)
ax.set_xlabel('x position (m)')
ax.set_ylabel('altitude (m)')

# faint full path
ax.plot(xs, ys, color='lightgray', linewidth=1)

# trail line and plane arrow
trail, = ax.plot([], [], color='tab:blue', linewidth=2)
arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

step = 20
frames = range(0, len(times), step)

def draw_frame(i):
    trail.set_data(xs[:i+1], ys[:i+1])

    # arrow
    angle = math.radians(actual_pitches[i])
    length = (max(xs) - min(xs)) * 0.03
    nose_x = xs[i] + length * math.cos(angle)
    nose_y = ys[i] + length * math.sin(angle)
    arrow.set_position((xs[i], ys[i]))   # tail of arrow
    arrow.xy = (nose_x, nose_y)          # head of arrow
    return trail, arrow

anim = FuncAnimation(fig, draw_frame, frames=frames, interval=30, blit=False)
plt.show()
