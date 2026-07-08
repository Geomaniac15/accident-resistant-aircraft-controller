import math
import numpy as np

import simulator as sim

# Regression tests for the simulator physics. No test framework needed:
#   python test_simulator.py        (also runs under pytest if installed)

# history column indices (see write_flight)
T, ALT, YAW, V, ELEV, AIL, RUD, THR = 0, 2, 6, 9, 11, 12, 13, 14


def run(phase='takeoff', **kw):
    return sim.simulate(sim.DEFAULT_K_H, sim.DEFAULT_K_VS, phase=phase, **kw)


def test_nominal_phases_survive():
    for phase in ('takeoff', 'climb', 'cruise'):
        m = run(phase, metrics=True)
        assert not m['crashed'], f'nominal {phase} crashed'
        assert m['min_margin'] > 0, f'nominal {phase} went below stall speed'
    m = run('takeoff', metrics=True)
    assert 550 < m['peak_alt'] < 700, f'takeoff peak {m["peak_alt"]} not near 600 m target'


def test_thrust_spools_smoothly():
    # first-order spool: per-tick thrust change is bounded by the largest
    # possible command error over the fastest time constant
    _, h = run('takeoff', fail_engines=(1,), fail_time=12.0, record=True)
    thr = np.array([r[THR] for r in h])
    bound = sim.n_engines * sim.thrust_per_engine / 1000.0 * sim.dt / sim.tau_spool_dn
    assert np.max(np.abs(np.diff(thr))) < bound * 1.05, 'thrust jumped faster than spool allows'


def test_surfaces_are_rate_limited():
    _, h = run('takeoff', accident='rudder-hardover', fail_time=5.0, record=True)
    for idx, rate in ((ELEV, sim.elev_rate), (AIL, sim.ail_rate), (RUD, sim.rud_rate)):
        pos = np.array([r[idx] for r in h])
        max_step = math.degrees(rate) * sim.dt
        assert np.max(np.abs(np.diff(pos))) <= max_step * 1.01, 'surface moved faster than actuator limit'
    rud = np.array([r[RUD] for r in h])
    assert rud.min() <= -math.degrees(sim.rud_max) + 0.1, 'hardover never reached the stop'


def test_engine_out_yaws_toward_dead_side():
    _, h = run('climb', fail_engines=(1, 2), fail_time=10.0, record=True)
    yaw_after = [r[YAW] for r in h if r[T] > 12.0]
    assert min(yaw_after) < -2.0, 'left-side engine failure did not yaw left'


def test_dead_engines_windmill():
    # all engines out: thrust settles to negative (windmilling drag), not zero
    _, h = run('cruise', fail_engines=(1, 2, 3, 4), fail_time=30.0, record=True)
    thr = np.array([r[THR] for r in h])
    expected = -sim.n_engines * sim.windmill_drag / 1000.0
    assert thr.min() < expected * 0.9, 'dead engines produced no windmilling drag'


def test_turbulence_reproducible():
    _, h1 = run('takeoff', turb_sigma=2.0, seed=42, record=True)
    _, h2 = run('takeoff', turb_sigma=2.0, seed=42, record=True)
    assert h1[-1] == h2[-1], 'same seed gave different trajectories'
    _, h3 = run('takeoff', turb_sigma=2.0, seed=7, record=True)
    assert h1[-1] != h3[-1], 'different seed gave identical trajectory'


def test_touchdown_classification():
    # a flown approach is a landing; an uncontrolled ground contact is a crash
    m = run('approach', metrics=True)
    assert m['landed'] and not m['crashed'], 'nominal approach should touch down safely'
    m = run('approach', accident='wing-loss', fail_time=30.0, metrics=True)
    assert m['crashed'] and not m['landed'], 'wing loss on approach should be a crash'


def test_stall_speed_physics():
    assert sim.stall_speed(10000, 'clean') > sim.stall_speed(0, 'clean'), \
        'stall speed should rise with altitude'
    assert sim.stall_speed(0, 'landing') < sim.stall_speed(0, 'clean'), \
        'flaps should lower the stall speed'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'PASS  {fn.__name__}')
    print(f'\n{len(tests)} tests passed')
