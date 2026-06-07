dt = 0.01   # time per tick, in seconds
g = 9.81    # acceleration due to gravity, in m/s^2
mass = 1.0    # mass of the dot, in kg

height = 0.0  # initial height, in meters
velocity = 0.0 # initial velocity, in m/s

target = 100.0  # target height, in meters
hover = mass * g  # thrust required to hover, in Newtons

Kp = 1.0  # proportional gain for the controller
Kd = 2.0  # derivative gain for the controller

previous_error = target - height  # initial error, in meters

t = 0.0  # initial time, in seconds
while t < 10.0: 
    error = target - height
    error_rate = (error - previous_error) / dt
    throttle = hover + Kp * error + Kd * error_rate
    previous_error = error

    net_force = throttle - mass * g
    acceleration = net_force / mass
    velocity = velocity + acceleration * dt
    height = height + velocity * dt

    t = t + dt
    print(f't={t:.2f}s, height={height:.2f}m, velocity={velocity:.2f} m/s')