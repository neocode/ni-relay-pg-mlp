"""Step 3 - Tables 1, 2, 4 (rows from evaluation.py) and 6 of the paper, printed and written to tables.md.
Tables 3, 5, 7 and the additional rows of Tables 4 and 6 are in extensions.txt.

Run after experiment.py and evaluation.py:  python tables.py
"""
import json
import numpy as np
import pandas as pd
from ni_drive import design_limits, ni_coefficients
from results_io import load_results2

R2 = load_results2()
df = pd.DataFrame(R2['rows'])
ex = df[df.est == 'exact']
out = []


def table(title, header, rows):
    out.append(f'### {title}\n')
    out.append('| ' + ' | '.join(header) + ' |')
    out.append('|' + '---|' * len(header))
    out.extend('| ' + ' | '.join(r) + ' |' for r in rows)
    out.append('')


# Table 1 (computed part): derivative limits (4) and N-i coefficients (5), (6) at kappa_hat = 1
V, A, J = design_limits(1.0)
K = ni_coefficients(V, A, J)
table('Table 1 (computed rows). Derivative limits (4) and N-i coefficients (5), (6), kappa_hat = 1',
      ['Quantity', 'Value'],
      [['E2max, E3max, E4max', f'{V:.2f} rad/s; {A:.3f} rad/s^2; {J:.0f} rad/s^3'],
       ['K12, K13, K23', f'{K[0]:.4f} s; {K[1]:.3g} s^2; {K[2]:.3g} s']])


def trow(lab, q):
    return [lab, f'{q.tau1.median():.3f}', f'{np.percentile(q.tau1, 99):.3f}', f'{q.tau1.max():.3f}',
            f'{100 * q.ovs.max():.2f}', f'{q.mpk.max():.3f}', f'{q.wpk.max():.3f}']


names = {'Fixed N-i': 'Fixed N-i design', 'Rescheduled N-i': 'Rescheduled N-i', 'Adaptive N-i': 'Amplitude-adaptive N-i',
         'Law (11)': 'Law (11)', 'LUT 2-D': 'LUT (raw inputs)', 'MLP raw': 'MLP (raw inputs)', 'PG-MLP': 'PG-MLP',
         'PG-MLP margin': 'PG-MLP, margin kappa_hat/0.9'}
o40 = df[(df.est == 'exact40') & (df.method == 'Oracle')]
table('Table 2. Closed-loop results on 2500 test moves, exact inertia ratio',
      ['Controller', 'tau1 median', 'tau1 99 %', 'tau1 max', 'overshoot max, %', 'Mpk/Mmax max', 'wpk/wmax max'],
      [trow(lab, ex[ex.method == m]) for m, lab in names.items()])

abl = {'MLP dimensionless': 'MLP on (log10 rho, kappa_hat), no law', 'LUT 1-D residual': 'Law + 1-D LUT of residual',
       'PCHIP residual': 'Law + PCHIP of residual', 'PG-MLP no kappa': 'PG-MLP without kappa_hat',
       'PG-MLP single': 'PG-MLP, single network', 'PG-MLP K12 only': 'PG-MLP, K12 only', 'PG-MLP': 'PG-MLP (reference implementation)'}
rows3 = []
for m, lab in abl.items():
    q = ex[ex.method == m]
    rows3.append([lab, f"{R2['rmse'][m][0]:.4f}", f'{q.tau1.median():.3f}', f'{np.percentile(q.tau1, 99):.3f}',
                  f'{q.tau1.max():.3f}', f'{100 * q.ovs.max():.2f}'])
table('Table 4. Ablation of the physics-guided scheduler (2500 test moves; further rows in extensions.txt)',
      ['Variant', 'RMSE x12 (40 moves)', 'tau1 median', 'tau1 99 %', 'tau1 max', 'overshoot max, %'], rows3)

rows4 = []
for est, lab, n in (('exact', 'Exact', 2500), ('err10', 'Random, |delta| <= 0.1', 2500),
                    ('under10', 'delta = -0.1', 500), ('over10', 'delta = +0.1', 500)):
    q0 = df[(df.est == est) & (df.method == 'PG-MLP')]; q1 = df[(df.est == est) & (df.method == 'PG-MLP margin')]
    assert len(q0) == len(q1) == n
    rows4.append([lab, str(n), f'{q0.tau1.median():.3f} / {q1.tau1.median():.3f}',
                  f'{q0.tau1.max():.3f} / {q1.tau1.max():.3f}',
                  f'{100 * q0.ovs.max():.2f} / {100 * q1.ovs.max():.2f}', f'{q0.mpk.max():.3f} / {q1.mpk.max():.3f}'])
table('Table 6. Robustness of PG-MLP under inertia estimation errors (without / with the inertia margin)',
      ['Inertia estimate', 'Moves', 'tau1 median', 'tau1 max', 'overshoot max, %', 'Mpk/Mmax max'], rows4)

text = '\n'.join(out)
open('tables.md', 'w', encoding='utf-8').write(text + '\n')
print(text)
