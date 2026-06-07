dt = 0.01   # time per tick, in seconds
g = 9.81    # acceleration due to gravity, in m/s^2
mass = 1.0    # mass of the dot, in kg

height = 100.0  # initial height, in meters
velocity = 0.0 # initial velocity, in m/s

throttle = g  # upward force in Newtons

t = 0.0  # initial time, in seconds
while height > 0 and t < 30.0:
    # net force: thrust - weight
    net_force = throttle - mass * g
    acceleration = net_force / mass
    
    velocity = velocity + acceleration * dt
    height = height + velocity * dt

    t = t + dt
    print(f't={t:.2f}s, height={height:.2f}m, velocity={velocity:.2f} m/s')