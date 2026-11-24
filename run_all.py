"""Run the complete pipeline in order (about 10 min on two CPU cores).

    python run_all.py [--workers N] [--skip-simulation]

--skip-simulation reuses the archived results.json, models.pkl and results2.json.gz and only
regenerates the tables, the checks and the figures (about 1 min).
"""
import argparse, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=2)
ap.add_argument('--skip-simulation', action='store_true')
args = ap.parse_args()
steps = [] if args.skip_simulation else [['experiment.py', '--workers', str(args.workers)],
                                         ['evaluation.py', '--workers', str(args.workers)]]
steps += [['ni_drive.py'], ['tables.py'], ['checks.py'], ['figures.py']]
for step in steps:
    t0 = time.time()
    print(f'\n=== python {" ".join(step)}', flush=True)
    subprocess.run([sys.executable] + step, check=True)
    print(f'=== {step[0]} finished in {time.time() - t0:.0f} s', flush=True)
