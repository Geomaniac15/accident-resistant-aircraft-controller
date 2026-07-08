import argparse
import csv
import glob
import itertools
import os
from multiprocessing import Pool

import simulator as sim

# Failure-sweep harness. Runs the controller against a large grid of emergencies
# (phase x failure x affected unit x timing) and records whether it survives.
# Designed to shard cleanly across a SLURM array job: each task runs --shard k/N.

# --- sweep axes (the cartesian product of these is the workload) ---
PHASE_TIMES = {
    'takeoff':  [4, 6, 8, 10, 12, 15, 18],
    'climb':    [10, 30, 60, 90, 120],
    'cruise':   [30, 90, 150, 220, 280],
    'approach': [10, 20, 30, 40, 50, 60],
}
# every non-empty subset of the four engines (15 combinations)
ENGINE_SETS = [c for r in range(1, 5) for c in itertools.combinations((1, 2, 3, 4), r)]
SIDED_ACCIDENTS = {'rudder-hardover', 'aileron-hardover', 'wing-loss', 'explosion'}
WEIGHTS = [270_000, 300_000, 330_000, 360_000, 400_000, 440_000]   # kg (6)


def _winds(mags, dirs_with_cross):
    out = [((0.0, 0.0, 0.0), 'calm')]
    for m in mags:
        out.append(((float(m), 0.0, 0.0), f'tail{m}'))
        out.append(((-float(m), 0.0, 0.0), f'head{m}'))
        if dirs_with_cross:
            out.append(((0.0, float(m), 0.0), f'cross{m}'))
    return out

WINDS_LOW = _winds([10, 20, 30], dirs_with_cross=True)         # takeoff/approach (10)
WINDS_HIGH = _winds([20], dirs_with_cross=False)               # climb/cruise (3)
# controller-tuning axis: full grid of altitude and climb-rate gains (3 x 3 = 9)
GAINS = [(kh, kv) for kh in (0.10, 0.15, 0.20) for kv in (0.02, 0.03, 0.04)]

PHASES = list(sim.PHASES)
FIELDS = ['phase', 'failure', 'side', 'fail_time', 'weight', 'wind', 'K_h', 'K_vs',
          'outcome', 't_end', 'peak_alt', 'min_alt', 'min_speed', 'min_margin', 'max_bank', 'max_aoa']


def build_scenarios():
    scenarios = []
    for phase in PHASES:
        winds = WINDS_LOW if phase in ('takeoff', 'approach') else WINDS_HIGH
        for weight in WEIGHTS:
            for wvec, wlabel in winds:
                for K_h, K_vs in GAINS:
                    common = dict(phase=phase, weight=weight, wind=wvec, wind_label=wlabel,
                                  K_h=K_h, K_vs=K_vs)
                    # baseline: no failure
                    scenarios.append(dict(fail_engines=(), accident=None, side='left',
                                          fail_time=None, **common))
                    for ft in PHASE_TIMES[phase]:
                        for eng in ENGINE_SETS:
                            scenarios.append(dict(fail_engines=eng, accident=None, side='left',
                                                  fail_time=ft, **common))
                        for acc in sim.ACCIDENTS:
                            for s in (['left', 'right'] if acc in SIDED_ACCIDENTS else ['left']):
                                scenarios.append(dict(fail_engines=(), accident=acc, side=s,
                                                      fail_time=ft, **common))
    return scenarios


def label(sc):
    if sc['accident']:
        return sc['accident']
    if sc['fail_engines']:
        return 'engines:' + '+'.join(map(str, sc['fail_engines']))
    return 'nominal'


def run_one(sc):
    m = sim.simulate(sc['K_h'], sc['K_vs'], phase=sc['phase'],
                     fail_engines=sc['fail_engines'], fail_time=sc['fail_time'],
                     accident=sc['accident'], accident_side=sc['side'],
                     mass_kg=sc['weight'], wind=sc['wind'], metrics=True)
    return {
        'phase': sc['phase'],
        'failure': label(sc),
        'side': sc['side'] if (sc['accident'] in SIDED_ACCIDENTS or sc['fail_engines']) else '-',
        'fail_time': sc['fail_time'] if sc['fail_time'] is not None else '-',
        'weight': sc['weight'],
        'wind': sc['wind_label'],
        'K_h': sc['K_h'], 'K_vs': sc['K_vs'],
        'outcome': 'CRASH' if m['crashed'] else ('landed' if m['landed'] else 'survived'),
        't_end': m['t_end'], 'peak_alt': m['peak_alt'], 'min_alt': m['min_alt'],
        'min_speed': m['min_speed'], 'min_margin': m['min_margin'],
        'max_bank': m['max_bank'], 'max_aoa': m['max_aoa'],
    }


def parse_shard(text):
    k, n = text.split('/')
    return int(k), int(n)


def summarize(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print('no result files matched', pattern)
        return
    total = len(rows)
    crashed = sum(r['outcome'] == 'CRASH' for r in rows)
    print(f'{total} scenarios   survived {total - crashed}   crashed {crashed}\n')
    phases = sorted({r['phase'] for r in rows})
    failures = sorted({r['failure'] for r in rows})
    width = max(len(f) for f in failures) + 2
    print(' ' * width + ''.join(f'{p:>10}' for p in phases))
    for fl in failures:
        cells = []
        for p in phases:
            sub = [r for r in rows if r['failure'] == fl and r['phase'] == p]
            cell = '.' if not sub else f'{sum(r["outcome"] != "CRASH" for r in sub)}/{len(sub)}'
            cells.append(f'{cell:>10}')
        print(f'{fl:<{width}}' + ''.join(cells))
    print('\n(cells = survived-or-landed / total across timings and sides)')


def heatmap(pattern, out='results/heatmap.png'):
    # survival-rate heatmap: one panel per phase, failure type x failure time.
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print('no result files matched', pattern)
        return
    rows = [r for r in rows if r['failure'] != 'nominal']   # baseline has no fail_time axis
    phases = [p for p in ('takeoff', 'climb', 'cruise', 'approach')
              if any(r['phase'] == p for r in rows)]
    failures = sorted({r['failure'] for r in rows})

    # sequential blue ramp (light = low survival receding to the surface, dark = high)
    ramp = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
    cmap = LinearSegmentedColormap.from_list('survival', ramp)

    fig, axes = plt.subplots(1, len(phases), figsize=(3.4 * len(phases) + 2, 0.42 * len(failures) + 2),
                             layout='constrained', squeeze=False)
    fig.suptitle('survival rate by failure type and failure time', fontsize=12)
    for ax, phase in zip(axes[0], phases):
        sub = [r for r in rows if r['phase'] == phase]
        ftimes = sorted({float(r['fail_time']) for r in sub})
        grid = np.full((len(failures), len(ftimes)), np.nan)
        counts = {}
        for r in sub:
            i = failures.index(r['failure'])
            j = ftimes.index(float(r['fail_time']))
            n_ok, n = counts.get((i, j), (0, 0))
            counts[(i, j)] = (n_ok + (r['outcome'] != 'CRASH'), n + 1)
        for (i, j), (n_ok, n) in counts.items():
            grid[i, j] = n_ok / n
        im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')
        for (i, j), (n_ok, n) in counts.items():
            ink = '#ffffff' if grid[i, j] > 0.55 else '#0b0b0b'
            ax.text(j, i, f'{n_ok}/{n}', ha='center', va='center', fontsize=7, color=ink)
        ax.set_title(phase, fontsize=10)
        ax.set_xticks(range(len(ftimes)), [f'{ft:g}' for ft in ftimes], fontsize=8)
        ax.set_xlabel('failure time (s)', fontsize=8)
        if ax is axes[0][0]:
            ax.set_yticks(range(len(failures)), failures, fontsize=8)
        else:
            ax.set_yticks(range(len(failures)), [''] * len(failures))
    fig.colorbar(im, ax=axes[0][-1], shrink=0.8, label='survival rate')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'wrote {out}  ({len(rows)} scenario rows)')


def main():
    ap = argparse.ArgumentParser(description='aircraft failure sweep (SLURM-shardable)')
    ap.add_argument('--shard', default='0/1', help='this task as k/N (e.g. 3/8)')
    ap.add_argument('--workers', type=int, default=os.cpu_count(), help='parallel processes')
    ap.add_argument('--out', default='results/sweep.csv', help='output CSV for this shard')
    ap.add_argument('--summarize', metavar='GLOB', help='aggregate result CSVs into a matrix and exit')
    ap.add_argument('--heatmap', metavar='GLOB', help='render result CSVs as a survival heatmap and exit')
    ap.add_argument('--heatmap-out', default='results/heatmap.png', help='heatmap output image')
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if args.heatmap:
        heatmap(args.heatmap, args.heatmap_out)
        return

    scenarios = build_scenarios()
    k, n = parse_shard(args.shard)
    mine = scenarios[k::n]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    print(f'shard {k}/{n}: {len(mine)} of {len(scenarios)} scenarios on {args.workers} workers')

    with Pool(args.workers) as pool:
        results = pool.map(run_one, mine)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)
    crashed = sum(r['outcome'] == 'CRASH' for r in results)
    print(f'wrote {args.out}: {len(results)} runs, {crashed} crashed')


if __name__ == '__main__':
    main()
