dt = 0.01   # time per tick, in seconds
g = 9.81    # acceleration due to gravity, in m/s^2

height = 100.0  # initial height, in meters
velocity = 20.0  # initial velocity, in m/s

t = 0.0  # initial time, in seconds
while height > 0:
    # dynamics
    velocity = velocity - g * dt
    height = height + velocity * dt

    t = t + dt
    print(f't={t:.2f}s, height={height:.2f}m, velocity={velocity:.2f} m/s')