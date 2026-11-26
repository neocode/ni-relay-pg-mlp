"""Step 4 - numbers quoted in the text of the paper that are not in the tables.

Writes checks.txt. Run after experiment.py and evaluation.py:  python checks.py
"""
import json
import numpy as np
import pandas as pd
from results_io import load_results2

R1 = json.load(open('results.json'))
R2 = load_results2()
d = pd.DataFrame(R2['rows'])
ex = d[d.est == 'exact']
lines = []
say = lambda *a: lines.append(' '.join(str(x) for x in a))

say('== Results (2500 test moves, exact inertia ratio)')
r = ex[ex.method == 'Rescheduled N-i']
say('Rescheduled N-i: max tau1 for rho >= 2:', round(r[r.rho >= 2].tau1.max(), 3),
    '| max tau1 for rho < 1:', round(r[r.rho < 1].tau1.max(), 2))
f = ex[ex.method == 'Fixed N-i']
say('Fixed N-i: max peak torque for kappa > 1:', round(f[f.kappa > 1].mpk.max(), 3), 'Mmax')
a = ex[ex.method == 'Adaptive N-i']
say('Amplitude-adaptive N-i: median tau1', round(a.tau1.median(), 3), '| for rho < 0.1: max tau1',
    round(a[a.rho < 0.1].tau1.max(), 3), ', max overshoot', round(100 * a[a.rho < 0.1].ovs.max(), 2), '%')
e40 = d[d.est == 'exact40'].pivot(index='case', columns='method', values='tau')
say('PG-MLP vs oracle on the 40 labelled moves: max |delta tau| =', round((e40['PG-MLP'] - e40['Oracle']).abs().max(), 4))
say('Peak speed over all evaluations: max', f"{d.wpk.max():.6f}", 'wmax')
say('Peak torque of PG-MLP and rescheduled N-i (exact):',
    round(ex[ex.method.isin(['PG-MLP', 'Rescheduled N-i'])].mpk.max(), 3), 'Mmax')

say('\n== Integration-step study (four moves)')
c = pd.DataFrame(R2['conv'])
for dt in sorted(c.dt.unique(), reverse=True):
    q = c[c.dt == dt]
    say(f'dt = {dt * 1e6:4.1f} us: torque excess {100 * (q.mpk.min() - 1):.2f}..{100 * (q.mpk.max() - 1):.2f} %')
ref = c[c.dt == 1e-6].set_index(['A', 'kappa', 'method']).tau
cur = c[c.dt == 1e-5].set_index(['A', 'kappa', 'method']).tau
allc = c.set_index(['A', 'kappa', 'method'])
say('max relative change of tau with respect to 1 us: at 10 us', f'{100 * ((cur - ref).abs() / ref).max():.2f} %,',
    'over all steps', f'{100 * ((allc.tau - ref.reindex(allc.index)).abs() / ref.reindex(allc.index)).max():.2f} %')

say('\n== Accuracy of the predicted corrections (40 labelled moves)')
for m in ('PG-MLP', 'LUT 2-D', 'MLP raw'):
    say(f'{m:10s} RMSE x12 {R2["rmse"][m][0]:.4f}, max relative error K12 {100 * R2["max_rel"][m][0]:.2f} %, '
        f'K13 {100 * R2["max_rel"][m][1]:.1f} %')
S = pd.DataFrame(R1['sens'])
say('Sensitivity (Fig. 2, b), median tau vs relative error of K12:',
    {f'{100 * (np.exp(k) - 1):+.0f} %': round(v, 3) for k, v in S.groupby('d').tau.median().items()})

say('\n== Seed groups (five disjoint groups of five seeds, first 500 moves; all 2500 moves: extensions.txt)')
s = pd.DataFrame(R2['seed_rows'])
g = s.groupby(['method', 'group']).tau1.max().groupby('method').agg(['mean', 'std'])
for m, row in g.iterrows():
    say(f'{m:8s} max tau1: {row["mean"]:.3f} +- {row["std"]:.3f} (mean +- standard deviation)')

say('\n== Robustness and the inertia margin')
for est in ('err10', 'under10', 'over10'):
    q0 = d[(d.est == est) & (d.method == 'PG-MLP')]; q1 = d[(d.est == est) & (d.method == 'PG-MLP margin')]
    say(f'{est:8s} median tau1 {q0.tau1.median():.3f} -> {q1.tau1.median():.3f} with margin '
        f'(+{100 * (q1.tau1.median() / q0.tau1.median() - 1):.1f} %), peak torque {q0.mpk.max():.3f} -> {q1.mpk.max():.3f}')
q0 = ex[ex.method == 'PG-MLP']; q1 = ex[ex.method == 'PG-MLP margin']
say(f'exact    median tau1 {q0.tau1.median():.3f} -> {q1.tau1.median():.3f} with margin '
    f'(+{100 * (q1.tau1.median() / q0.tau1.median() - 1):.1f} %)')
say('Scheduled controllers (all except the fixed design) with inertia error, no margin: max peak torque',
    round(d[d.est.isin(['err10', 'under10']) & ~d.method.isin(['Fixed N-i', 'PG-MLP margin'])].mpk.max(), 3), 'Mmax')

say('\n== Inputs outside the training range of kappa [0.5, 2] (exact, random and boundary errors)')
for m in ('PG-MLP', 'PG-MLP margin'):
    y = d[(d.est != 'exact40') & (d.method == m)]
    o = y[(y.kappa_hat > 2.0) | (y.kappa_hat < 0.5)]
    say(f'{m:14s} kappa_hat range {y.kappa_hat.min():.3f}..{y.kappa_hat.max():.3f}; outside: {len(o)} of {len(y)}, '
        f'max tau1 {o.tau1.max():.3f}, max peak torque {o.mpk.max():.3f} '
        f'(inside: max tau1 {y.drop(o.index).tau1.max():.3f})')

say('\n== Implementation')
say('Safeguard (12) active in', R2['safeguard_active'], 'of', R2['n_big'], 'moves')
say(f"PG-MLP: {R1['n_par']} parameters per network (inference timings: see deployment.txt)")
text = '\n'.join(lines)
open('checks.txt', 'w', encoding='utf-8').write(text + '\n')
print(text)
