"""Step 2b - summary of the additional studies (results3.json) for the tables and the text -> extensions.txt.

Run after extensions.py:  python extensions_report.py
"""
import json
import numpy as np
import pandas as pd
from results_io import load_results2

R3 = json.load(open('results3.json'))
R2 = load_results2()
d2 = pd.DataFrame(R2['rows'])
lines = []
say = lambda *a: lines.append(' '.join(str(x) for x in a))
q = lambda s, p: float(np.percentile(s, p))

say('== Minimum 1 %-settling time T_opt of the drive model (1) under the limits (LP, numerical estimate)')
t = pd.DataFrame(R3['topt']['rows'])
t5 = t[t.group == 'fig5'].iloc[0]
MS = ('Oracle', 'PG-MLP', '2-D interpolation of residual', 'Law (11)', 'Adaptive N-i', 'Rescheduled N-i', 'Fixed N-i')
for grp in ('labelled', 'worst30', 'random30'):
    tg = t[t.group == grp]
    say(f'-- {grp} ({len(tg)} moves): T*_1% / T_opt median {np.median(tg.tref1 / tg.topt):.3f}, '
        f'range {np.min(tg.tref1 / tg.topt):.3f}..{np.max(tg.tref1 / tg.topt):.3f}')
    for m in MS:
        if m not in tg or tg[m].isna().all():
            continue
        r = tg[m] / tg.topt
        mp = f'; peak torque max {tg[m + " mpk"].max():.3f} Mmax' if m + ' mpk' in tg else ''
        say(f'   {m:30s} t_s / T_opt: median {r.median():.3f}, min {r.min():.3f}, max {r.max():.3f}{mp}')
allp = t[t.group != 'fig5']
r = allp['PG-MLP'] / allp.topt
say(f'PG-MLP over all {len(allp)} moves with T_opt: median {r.median():.3f}, max {r.max():.3f}')
say(f'Fig. 5 move (A = 0.05 rad, kappa = 1.3): T_opt = {t5.topt:.4f} s, T*_1% = {t5.tref1:.4f} s; t_s: ' +
    ', '.join(f'{m} {t5[m]:.4f} s' for m in MS[1:] if m in t5 and np.isfinite(t5[m])))
c = pd.DataFrame(R3['topt']['check'])
say(f'Discretisation check (n = 300 vs 600 intervals, {len(c)} moves): max relative difference '
    f'{100 * np.max(np.abs(c.t300 - c.t600) / c.t600):.2f} %')
if 'topt_sens' in R3:
    s_ = pd.DataFrame(R3['topt_sens']).merge(t[t.group == 'worst30'][['case', 'topt', 'PG-MLP']], on='case')
    say(f'Torque limit 1.016 Mmax ({len(s_)} worst moves): T_opt decreases by '
        f'{100 * (1 - (s_.topt_1016 / s_.topt)).min():.2f}..{100 * (1 - (s_.topt_1016 / s_.topt)).max():.2f} %; '
        f'PG-MLP t_s / T_opt(1.016) max {(s_["PG-MLP"] / s_.topt_1016).max():.3f}')

say('\n== Overshoot of PG-MLP and comparison with the oracle on the same 40 labelled moves')
ex = d2[d2.est == 'exact']; p = ex[ex.method == 'PG-MLP']
say(f'PG-MLP, 2500 moves: overshoot > 0.5 % in {int((p.ovs > 0.005).sum())} moves, > 0.75 % in {int((p.ovs > 0.0075).sum())}, max {100 * p.ovs.max():.2f} %')
e40 = d2[d2.est == 'exact40']
for m in ('Oracle', 'PG-MLP'):
    y = e40[e40.method == m]
    say(f'{m:7s} 40 labelled moves: tau1 median {y.tau1.median():.3f}, max {y.tau1.max():.3f}; overshoot max {100 * y.ovs.max():.2f} %')

say('Moves with overshoot above the 1 % band (PG-MLP / PG-MLP with margin): ' + '; '.join(
    f"{est}: {int((d2[(d2.est == est) & (d2.method == 'PG-MLP')].ovs > 0.01).sum())} / "
    f"{int((d2[(d2.est == est) & (d2.method == 'PG-MLP margin')].ovs > 0.01).sum())} "
    f"(max {100 * d2[(d2.est == est) & (d2.method == 'PG-MLP')].ovs.max():.2f} %)"
    for est in ('exact', 'err10', 'under10', 'over10')))
say('\n== Further residual baselines (2500 test moves, exact inertia ratio)')
b = pd.DataFrame(R3['base2d']['rows']); be = b[b.est == 'exact']
for m, y in be.groupby('method', sort=False):
    acc = R3['base2d']['acc'][m]
    say(f'{m:32s} tau1 median {y.tau1.median():.3f}, 99 % {q(y.tau1, 99):.3f}, max {y.tau1.max():.3f}; '
        f'overshoot max {100 * y.ovs.max():.2f} %; RMSE x12 (40 moves) {acc["rmse"][0]:.4f}; '
        f'max rel. error K12 {100 * acc["max_rel"][0]:.2f} %; safeguard active in {int(y.guard.sum())} moves')
say('GP kernel after fitting:', R3['base2d']['gp_kernel'])

say('\n== Damping ratio of the outer sliding equation, zeta = K12 / (2 sqrt(K13))')
for k, v in R3['zeta'].items():
    say(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}')

say('\n== Non-idealities (500 test moves; tau1 relative to the nominal kinematic reference)')
r = pd.DataFrame(R3['robust'])
for (sc, m), y in r.groupby(['scen', 'method'], sort=False):
    unset = int(np.isinf(y.tau1).sum())
    say(f'{sc:17s} {m:16s} tau1 median {y.tau1.replace(np.inf, np.nan).median():.3f}, max {y.tau1.max():.3f}, '
        f'unsettled {unset}; overshoot max {100 * y.ovs.max():.2f} % (> 1 % in {int((y.ovs > 0.01).sum())}); '
        f'Mpk max {y.mpk.max():.3f}; wpk max {y.wpk.max():.4f}')

say('\n== Seed groups on all 2500 test moves (five disjoint groups of five seeds)')
s = pd.DataFrame(R3['seeds'])
g = s.groupby(['method', 'group']).tau1.max().groupby('method').agg(['mean', 'std', 'max'])
for m, row in g.iterrows():
    say(f'{m:8s} max tau1: {row["mean"]:.3f} +- {row["std"]:.3f} (worst group {row["max"]:.3f})')

if 'irregular' in R3:
    say('\n== Regressors of the residual on different label sets (2500 test moves, exact inertia ratio)')
    ir = pd.DataFrame(R3['irregular']['rows'])
    for m, y in ir.groupby('method', sort=False):
        say(f'{m:50s} RMSE x12 (40 moves) {R3["irregular"]["rmse40"][m][0]:.4f}; tau1 median {y.tau1.median():.3f}, '
            f'99 % {q(y.tau1, 99):.3f}, max {y.tau1.max():.3f}; overshoot max {100 * y.ovs.max():.2f} %')
if 'designs' in R3:
    say('\n== Label designs (125 labels unless stated; 2500 test moves, exact inertia ratio); GP with separate '
        'hyper-parameters for x12 and x13')
    dz = pd.DataFrame(R3['designs']['rows'])
    for m, y in dz.groupby('method', sort=False):
        say(f'{m:50s} RMSE x12 / x13 (40 moves) {R3["designs"]["rmse40"][m][0]:.4f} / {R3["designs"]["rmse40"][m][1]:.4f}; '
            f'tau1 median {y.tau1.median():.3f}, 99 % {q(y.tau1, 99):.3f}, max {y.tau1.max():.3f}, '
            f'tau1 > 1.05 in {int((y.tau1 > 1.05).sum())} moves; overshoot max {100 * y.ovs.max():.2f} %')
if 'designs' in R3 and 'cases' in R3['designs']:
    tm = d2[(d2.est == 'exact') & (d2.method == 'PG-MLP')].sort_values('case')
    lo_, hi_ = np.log10(0.003), np.log10(3.0)
    unit = lambda c: np.c_[(np.log10(np.asarray(c)[:, 0]) - lo_) / (hi_ - lo_), (np.asarray(c)[:, 1] - 0.5) / 1.5]
    TU = unit(np.c_[tm.A.values, tm.kappa.values])
    say('Coverage: fill distance (largest distance from a test move to the nearest label, unit square of '
        '(log10 A, kappa)) and median distance of the moves with tau1 > 1.05 to the nearest label')
    for k, cs in R3['designs']['cases'].items():
        D = np.sqrt(((TU[:, None, :] - unit(cs)[None]) ** 2).sum(-1)).min(1)
        bad = sorted({int(c) for c in dz[dz.method.str.startswith(k + ':') & (dz.tau1 > 1.05)].case})
        say(f'{k:24s} {len(cs):3d} labels: fill distance {D.max():.3f}, median distance of all moves {np.median(D):.3f}, '
            f'of the {len(bad)} degraded moves ' + (f'{np.median(D[bad]):.3f}' if bad else '-'))
if 'robust3' in R3:
    say('\n== 2-D interpolant and Gaussian process of the residual: inertia margin (random delta, 2500 moves) and '
        'displacements below 0.003 rad (200 moves)')
    rb3 = pd.DataFrame(R3['robust3'])
    for (est, m), y in rb3.groupby(['est', 'method'], sort=False):
        say(f'{est:16s} {m:30s} tau1 median {y.tau1.median():.3f}, max {y.tau1.max():.3f}; overshoot max '
            f'{100 * y.ovs.max():.2f} %, above 1 % in {int((y.ovs > 0.01).sum())} moves; torque max {y.mpk.max():.3f}')
    oo = pd.DataFrame(R3['ood'])
    for est, y in (('err10 margin', d2[(d2.est == 'err10') & (d2.method == 'PG-MLP margin')]),
                   ('below 0.003 rad', oo[(oo.A < 0.003) & (oo.method == 'PG-MLP')])):
        say(f'{est:16s} PG-MLP (for comparison)        tau1 median {y.tau1.median():.3f}, max {y.tau1.max():.3f}; overshoot max '
            f'{100 * y.ovs.max():.2f} %, above 1 % in {int((y.ovs > 0.01).sum())} moves; torque max {y.mpk.max():.3f}')
if 'robust2' in R3:
    say('\n== 2-D interpolant and Gaussian process of the residual under inertia estimation errors')
    rb2 = pd.DataFrame(R3['robust2'])
    for (est, m), y in rb2.groupby(['est', 'method'], sort=False):
        say(f'{est:8s} {m:30s} tau1 median {y.tau1.median():.3f}, max {y.tau1.max():.3f}; overshoot max '
            f'{100 * y.ovs.max():.2f} %; Mpk max {y.mpk.max():.3f}')
    for est in ('err10', 'under10', 'over10'):
        y = d2[(d2.est == est) & (d2.method == 'PG-MLP')]
        say(f'{est:8s} PG-MLP (for comparison)          tau1 median {y.tau1.median():.3f}, max {y.tau1.max():.3f}; '
            f'overshoot max {100 * y.ovs.max():.2f} %; Mpk max {y.mpk.max():.3f}')

if 'numerics' in R3:
    say('\n== Integration step (20 labelled moves; PG-MLP trained on 10 us labels)')
    Ln = R3['numerics']['labels']
    for key in ('pg 1e-05', 'pg 5e-06', 'pg 1e-06', 'oracle10 1e-05', 'oracle10 1e-06', 'oracle1 1e-06'):
        t_ = np.array([l[key]['tau1'] for l in Ln]); o_ = np.array([l[key]['ovs'] for l in Ln]); m_ = np.array([l[key]['mpk'] for l in Ln])
        say(f'{key:16s} tau1 median {np.median(t_):.4f}, max {t_.max():.4f}; overshoot max {100 * o_.max():.2f} %; torque max {m_.max():.4f}')
    dx = np.array([l['label1'][0] - l['label10'][0] for l in Ln])
    say(f'Oracle labels at 1 us vs 10 us: |dx12| median {np.median(np.abs(dx)):.4f}, max {np.abs(dx).max():.4f} '
        f'(relative K12 {100 * (np.exp(np.abs(dx).max()) - 1):.2f} %)')
    rn = pd.DataFrame(R3['numerics']['rows']); t_ = rn[rn.tag == 'test']
    say(f'PG-MLP at 1 us, 2500 test moves: tau1 median {t_.tau1.median():.3f}, 99 % {q(t_.tau1, 99):.3f}, max {t_.tau1.max():.3f}; '
        f'overshoot above 1 % in {int((t_.ovs > 0.01).sum())} moves (max {100 * t_.ovs.max():.2f} %); torque max {t_.mpk.max():.4f}')
    o_ = rn[(rn.tag == 'topt') & (rn.group != 'fig5')]
    say(f'PG-MLP at 1 us, 100 moves with T_opt: t_s / T_opt median {(o_.ts / o_.topt).median():.3f}, max {(o_.ts / o_.topt).max():.3f} '
        f'(10 us: {(o_.ts10 / o_.topt).max():.3f}); torque max {o_.mpk.max():.4f}')
if 'step1us' in R3:
    s1 = R3['step1us']; r1 = pd.DataFrame(s1['rows'])
    say(f'Pipeline at 1 us (labels, training, evaluation): labels |dx12| vs 10 us median {s1["dx12_median"]:.4f}, max {s1["dx12_max"]:.4f}; '
        f'2500 moves: tau1 median {r1.tau1.median():.3f}, 99 % {q(r1.tau1, 99):.3f}, max {r1.tau1.max():.3f}; '
        f'overshoot above 1 % in {int((r1.ovs > 0.01).sum())} moves; torque max {r1.mpk.max():.4f}')
if 'topt_conv' in R3:
    tc = pd.DataFrame(R3['topt_conv']); r_ = tc.t600 / tc.t300 - 1
    say(f'T_opt with 600 vs 300 intervals ({len(tc)} moves): relative change median {100 * r_.median():.3f} %, '
        f'min {100 * r_.min():.3f} %, max {100 * r_.max():.3f} %')
if 'costmap' in R3:
    for c in R3['costmap']:
        F = np.array(c['F']); f0 = F.min(); d12 = np.array(c['d12']); d13 = np.array(c['d13'])
        best = [d12[row.argmin()] for row in F]; fixed = F[:, len(d12) // 2] / f0
        say(f'Cost map, rho = {c["rho"]:.3f}: along x13 offsets {d13[0]:+.1f}...{d13[-1]:+.1f} the best x12 offset is '
            f'{min(best):+.3f}...{max(best):+.3f} with cost <= {100 * (F.min(axis=1).max() / f0 - 1):.1f} % above the optimum; '
            f'with x12 fixed the cost rises by up to {100 * (fixed.max() - 1):.0f} %')

say('\n== Five-fold cross-validation on the 125 training labels (ensembles of five networks, residual target)')
for row in sorted(R3['cv'], key=lambda r: r['rmse_x12']):
    say(f'hidden {str(row["hidden"]):12s} alpha {row["alpha"]:.0e}: RMSE x12 {row["rmse_x12"]:.4f}, x13 {row["rmse_x13"]:.4f}')

say('\n== Displacements outside the training range [0.003, 3] rad (200 + 200 moves)')
o = pd.DataFrame(R3['ood'])
for rng_name, y0 in (('A < 0.003 rad', o[o.A < 0.003]), ('A > 3 rad', o[o.A > 3.0])):
    for m, y in y0.groupby('method', sort=False):
        say(f'{rng_name:13s} {m:16s} tau1 median {y.tau1.median():.3f}, max {y.tau1.max():.3f}; '
            f'overshoot max {100 * y.ovs.max():.2f} %; Mpk max {y.mpk.max():.3f}')

say('\n== Repeatability of the oracle labels (21 training cases, three restarts from perturbed starts each)')
L = R3['labels']
dx = np.array([[abs(rs['x'][0] - l['label'][0]), abs(rs['x'][1] - l['label'][1]), rs['f'] - l['label_cost']]
               for l in L for rs in l['restarts']])
say(f'|x12 - label|: median {np.median(dx[:, 0]):.4f}, max {dx[:, 0].max():.4f}; '
    f'|x13 - label|: median {np.median(dx[:, 1]):.3f}, max {dx[:, 1].max():.3f}; '
    f'cost of restarts minus label cost: median {np.median(dx[:, 2]):.4f}, min {dx[:, 2].min():.4f}')
say(f'Restarts (perturbed starts around the law, sigma = 0.3) with a lower cost than the label: '
    f'{int((dx[:, 2] < -1e-6).sum())} of {len(dx)}')
fl = np.array([[f['cost'] - l['flat'][2]['cost'] for f in l['flat']] for l in L])
say('Cost increase for x13 offsets (-0.5, -0.25, 0, +0.25, +0.5) around the label, median over cases: ' +
    ', '.join(f'{v:+.4f}' for v in np.median(fl, axis=0)) + '; max: ' + ', '.join(f'{v:+.3f}' for v in fl.max(axis=0)))
text = '\n'.join(lines)
open('extensions.txt', 'w', encoding='utf-8').write(text + '\n')
print(text)
