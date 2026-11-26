"""Step 5 - figures of the paper (300 dpi PNG).

fig1_structure.png           Fig. 1  relay cascade with scheduled switching coefficients
fig2_labels_sensitivity.png  Fig. 2  oracle corrections x12 and x13 vs rho (a), sensitivity to an error of K12 (b),
                             cost (10) around the oracle label of one move (c)
fig3_pipeline.png            Fig. 3  software pipeline: offline data generation and training, online scheduling
fig4_tau_vs_rho.png          Fig. 4  median and maximum tau1 in bins of rho on the 2500 test moves
fig5_transients.png          Fig. 5  transients for A = 0.05 rad, kappa = 1.3, four controllers (numbers in fig5_numbers.json)

Run after experiment.py and evaluation.py:  python figures.py
"""
import json, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from experiment import K_of, rho, law
from ni_drive import run, ni_coefficients, design_limits, t_ref, MMAX
from evaluation import adaptive_ni

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'font.size': 9, 'axes.linewidth': 0.6,
                     'axes.edgecolor': '#555', 'xtick.color': '#333', 'ytick.color': '#333',
                     'axes.grid': True, 'grid.color': '#e4e4e4', 'grid.linewidth': 0.5,
                     'legend.frameon': False, 'mathtext.fontset': 'stix'})
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#6b6b6b'
R = json.load(open('results.json'))
M = pickle.load(open('models.pkl', 'rb'))


def pg_predict(A, kh):
    P = np.array([[np.log10(rho(A, kh)), kh]])
    return law(A, kh) + np.mean([m.predict(M['sp'].transform(P)) for m in M['m_pg']], axis=0)[0]


# ---------- Fig. 1: structure ----------
NL = '\n'
fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.set_axis_off(); ax.set_xlim(-10, 100); ax.set_ylim(-3, 34)
def box(x, y, w, h, t, fc='white', ec='#333', fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8', fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs)
def arr(x1, y1, x2, y2, t=None, dy=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, lw=0.8, color='#333',
                                 shrinkA=0, shrinkB=0, zorder=3))
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, t, ha='center', va='bottom', fontsize=8)
box(0.5, 23, 18, 9, r'Move command $A$,' + NL + r'inertia estimate $\hat\kappa$', fc='#f4f4f2')
box(23, 23, 20, 9, r'Limits $E_{\max}(\hat\kappa)$,' + NL + r'analytic N-i vector $K^0$', fc='#f4f4f2')
box(47.5, 23, 25, 9, r'Scheduler: $\rho(A,\hat\kappa)$, law +' + NL + 'residual regressor' + NL + '(PG-MLP in this study)', fc='#e6effa', ec=BLUE)
box(77, 23, 22, 9, r'$K = K^0\odot\exp(x)$,' + NL + 'held during the move', fc='#f4f4f2')
P = 0.25  # box padding: arrows start and end exactly on the drawn box outlines
arr(18.5 + P, 27.5, 23 - P, 27.5); arr(43 + P, 27.5, 47.5 - P, 27.5); arr(72.5 + P, 27.5, 77 - P, 27.5)
box(4, 4, 13, 9, r'$R_1$: $s_1$' + NL + r'$\pm E_{2\max}$')
box(24, 4, 13, 9, r'$R_2$: $s_2$' + NL + r'$\pm E_{3\max}$')
box(44, 4, 13, 9, r'$R_3$: $s_3$' + NL + r'$\pm U_{\max}$')
box(64, 4, 15, 9, 'DC drive' + NL + r'$\theta,\ \omega,\ M$', fc='#f4f4f2')
box(84, 4, 15, 9, 'derivative' + NL + 'measurement', fc='#f4f4f2')
arr(17 + P, 8.5, 24 - P, 8.5, r'$E_2^*$'); arr(37 + P, 8.5, 44 - P, 8.5, r'$E_3^*$'); arr(57 + P, 8.5, 64 - P, 8.5, r'$U$')
arr(79 + P, 8.5, 84 - P, 8.5)
LK = dict(color='#333', lw=0.8, solid_capstyle='projecting', solid_joinstyle='miter', zorder=3)
ax.plot([88, 88, 10.5], [23 - P, 17.5, 17.5], **LK)
for xx in (10.5, 30.5):
    arr(xx, 17.5, xx, 13 + P)
ax.text(59, 18.3, r'$K_{12}, K_{13}$ to $R_1$;  $K_{23}$ to $R_2$', fontsize=8, ha='center')
box(-9.5, 4, 8, 9, 'errors' + NL + '(2)', fc='#f4f4f2')
arr(-5.5, 21.5, -5.5, 13 + P); ax.text(-5.5, 22.0, r'$\theta^*$', fontsize=9, ha='center', va='bottom')
ax.plot([91.5, 91.5, -5.5], [4 - P, -1.5, -1.5], **LK); arr(-5.5, -1.5, -5.5, 4 - P)
arr(-1.5 + P, 8.5, 4 - P, 8.5, r'$E_{1..3}$', dy=0.8)
ax.text(43, -1.0, r'measured $\theta,\ \omega,\ M$; errors $E_1=\theta^*-\theta,\ E_2,\ E_3$ (2) fed to all relays',
        fontsize=8, ha='center', va='bottom')
fig.savefig('fig1_structure.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------- Fig. 2: labels collapse + sensitivity ----------
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
tr = np.array(R['train']); lab = np.array([l[0] for l in R['lab_tr']])
lr = np.array([np.log10(rho(a, k)) for a, k in tr])
mk = {0.5: 'o', 0.75: 's', 1.0: '^', 1.5: 'D', 2.0: 'v'}
for k, m in mk.items():
    s = tr[:, 1] == k
    a1.plot(lr[s], lab[s, 0], m, ms=3.6, mfc='white', mec=GRAY, mew=0.7, ls='none', label=f'$\\kappa$={k}')
a1.plot(lr, lab[:, 1], '.', ms=3.2, color='#8b5cc6', ls='none', label='$x_{13}$, all $\\kappa$')
g = np.linspace(-2.6, 1.4, 300)
a1.plot(g, 0.5 * np.log(np.minimum(10 ** g, 1)), color=ORANGE, lw=1.4, ls='--', label='law (11)')
kh = 1.0
V, Am, J = design_limits(kh); Asat = V * V / Am + V * Am / J
gg = g[(g >= lr.min()) & (g <= lr.max())]
a1.plot(gg, [pg_predict(10 ** v * Asat, kh)[0] for v in gg], color=BLUE, lw=1.4, label='PG-MLP, $\\hat\\kappa$=1')
a1.set_xlabel('$\\log_{10}\\rho$'); a1.set_ylabel('$x_{12}$ (open), $x_{13}$ (dots)')
a1.legend(fontsize=6.3, ncol=2, loc='lower right', handlelength=1.4, columnspacing=0.6)
a1.set_title('(a) oracle corrections vs. $\\log_{10}\\rho$', fontsize=8.2)
S = R['sens']; ds = sorted(set(x['d'] for x in S))
med = [np.median([x['tau'] for x in S if x['d'] == d]) for d in ds]
lo = [np.min([x['tau'] for x in S if x['d'] == d]) for d in ds]
hi = [np.max([x['tau'] for x in S if x['d'] == d]) for d in ds]
pc = 100 * (np.exp(ds) - 1)
a2.fill_between(pc, lo, hi, color=BLUE, alpha=0.15, lw=0, label='range, 10 cases')
a2.plot(pc, med, 'o-', color=BLUE, ms=3.5, lw=1.4, label='median')
a2.axvline(0, color='#999', lw=0.6)
a2.set_xlabel('relative error of $K_{12}$, %'); a2.set_ylabel('$\\tau=t_s/T^*$')
a2.text(-7.0, 1.44, 'overshoot', fontsize=8, color='#333'); a2.text(3.0, 1.44, 'creep', fontsize=8, color='#333')
a2.legend(fontsize=6.5, loc='lower left'); a2.set_title('(b) sensitivity to $K_{12}$', fontsize=8.2)
import os
if os.path.exists('results3.json') and 'costmap' in json.load(open('results3.json')):   # extensions.py (costmap)
    cmap_ = json.load(open('results3.json'))['costmap'][0]                              # move of small rho
    F = np.array(cmap_['F']); Fmin = F.min()
    X12, X13 = np.meshgrid(cmap_['d12'], cmap_['d13'])
    Z = 100 * (F / Fmin - 1)
    lev = [0, 1, 2, 5, 10, 50, 1000]
    cf = a3.contourf(X12, X13, Z, levels=lev, colors=['#08306b', '#2171b5', '#6baed6', '#a9cce3', '#d6e6f2', '#f7fbff'])
    a3.contour(X12, X13, Z, levels=[2, 10], colors='#333', linewidths=0.5)
    cb = fig.colorbar(cf, ax=a3, fraction=0.06, pad=0.03, ticks=lev[:-1], spacing='uniform')
    cb.ax.set_yticklabels(['0', '1', '2', '5', '10', '50']); cb.ax.tick_params(labelsize=6.5)
    cb.set_label('cost above optimum, %', fontsize=7)
    a3.plot(0, 0, 'k+', ms=7, mew=1.2)
    a3.set_xlim(-0.04, 0.10)
    a3.set_xlabel('$x_{12}-x_{12}^{\\mathrm{o}}$'); a3.set_ylabel('$x_{13}-x_{13}^{\\mathrm{o}}$')
    a3.set_title(f'(c) cost (10), $\\rho$ = {cmap_["rho"]:.2f}', fontsize=8.2)
fig.tight_layout(w_pad=0.8); fig.savefig('fig2_labels_sensitivity.png', dpi=300); plt.close(fig)

# ---------- Fig. 5: transients for a small displacement ----------
A, kap = 0.05, 1.3
cases = [('Fixed N-i design', ni_coefficients(*design_limits(1.0)), 1.0, GRAY, '-'),
         ('Rescheduled N-i', ni_coefficients(*design_limits(kap)), kap, ORANGE, '--'),
         ('PG-MLP', K_of(pg_predict(A, kap), kap), kap, BLUE, '-'),
         ('Amplitude-adaptive N-i', adaptive_ni(A, kap), kap, '#8b5cc6', (0, (4, 2, 1, 2)))]
V, Am, J = design_limits(kap); Ts = t_ref(A, V, Am, J)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.7, 2.45))
out = {}
for name, K, kh, c, ls in cases:
    r = run(A, K, kappa=kap, kappa_hat=kh, T=1.0, rec_every=20)
    rec = r['rec']; out[name] = (r['ts'], r['Mpeak'] / MMAX, r['ovs'])
    b1.plot(rec[:, 0], 1000 * (A - rec[:, 1]), color=c, ls=ls, lw=1.3, label=name)
    Mf = np.convolve(rec[:, 3], np.ones(10) / 10, mode='same')
    b2.plot(rec[:, 0], rec[:, 3], color=c, lw=0.3, alpha=0.35)
    b2.plot(rec[:, 0], Mf, color=c, ls=ls, lw=1.1, label=name)
b1.axvline(Ts, color='k', lw=0.7, ls=':'); b1.text(Ts + 0.01, 42, '$T^*$', fontsize=8)
import os
if os.path.exists('results3.json'):   # minimum 1 %-settling time of the model (extensions.py)
    Topt = [r for r in json.load(open('results3.json'))['topt']['rows'] if r['case'] < 0][0]['topt']
    b1.axvline(Topt, color='k', lw=0.7, ls='--'); b1.text(Topt - 0.008, 30, '$T_{\\mathrm{opt}}$', fontsize=8, ha='right')
    out_topt = float(Topt)
b1.set_xlim(0, 0.9); b1.set_xlabel('t, s'); b1.set_ylabel('$E_1$, mrad'); b1.legend(fontsize=7)
b1.set_title(f'(a) position error, A = {A} rad, $\\kappa$ = {kap}', fontsize=8.5)
b2.axhline(MMAX, color='k', lw=0.6, ls=':'); b2.axhline(-MMAX, color='k', lw=0.6, ls=':')
b2.text(0.62, MMAX + 1.5, '$\\pm M_{\\max}$', fontsize=8)
b2.set_xlim(0, 0.9); b2.set_xlabel('t, s'); b2.set_ylabel('M, N·m')
b2.set_title('(b) torque: 2 ms moving average (thick), raw (thin)', fontsize=8.5)
fig.tight_layout(); fig.savefig('fig5_transients.png', dpi=300); plt.close(fig)
json.dump({k: dict(zip(('ts', 'Mpk_over_Mmax', 'overshoot'), map(float, vv))) for k, vv in out.items()} | {'Tstar': Ts}
          | ({'Topt': out_topt} if 'out_topt' in globals() else {}),
          open('fig5_numbers.json', 'w'), indent=1)

# ---------- Fig. 4: median and worst-case tau1 per bin of rho (2500 test moves) ----------
import pandas as pd
from results_io import load_results2
R2 = load_results2()
d = pd.DataFrame(R2['rows']); d = d[d.est == 'exact']
import os
if os.path.exists('results3.json'):   # 2-D interpolant and Gaussian process of the residual (extensions.py)
    b3 = pd.DataFrame(json.load(open('results3.json'))['base2d']['rows'])
    d = pd.concat([d, b3[b3.est == 'exact'].replace({'method': {'2-D interpolation of residual': 'Interp2D',
                                                                'GP residual': 'GP'}})])
edges = np.logspace(np.log10(d.rho.min()) - 1e-9, np.log10(d.rho.max()) + 1e-9, 15)
d = d.assign(bin=np.digitize(d.rho, edges) - 1)
mid = np.sqrt(edges[:-1] * edges[1:])
PURPLE, BROWN, OLIVE, PINK = '#8b5cc6', '#9c6b3c', '#9a9a2a', '#d4559a'
# (method, label, colour, marker, line style, panels): Fixed and rescheduled N-i are off scale in panel (b)
spec = [('Fixed N-i', 'fixed N-i design', BROWN, 'x', '-', 'a'),
        ('Rescheduled N-i', 'rescheduled N-i', ORANGE, '^', '-', 'a'),
        ('Adaptive N-i', 'amplitude-adaptive N-i', PURPLE, 's', '-', 'ab'),
        ('Law (11)', 'law (11)', OLIVE, 'P', '-', 'ab'),
        ('LUT 2-D', 'LUT (raw inputs)', PINK, '*', '-', 'ab'),
        ('MLP raw', 'MLP (raw inputs)', AQUA, 'D', '-', 'ab'),
        ('PG-MLP', 'PG-MLP', BLUE, 'o', '-', 'ab')]
if (d.method == 'Interp2D').any():
    spec += [('Interp2D', 'law + 2-D interpolant of residual', '#333333', 'v', '--', 'ab'),
             ('GP', 'law + Gaussian process of residual', '#999999', '<', ':', 'ab')]
fig, (c1, c2, ins) = plt.subplots(1, 3, figsize=(6.9, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
for m, lab, c, mk_, ls, panels in spec:
    q = d[d.method == m].groupby('bin').tau1
    med, mx = q.median(), q.max()
    c1.plot(mid[med.index], med.values, marker=mk_, ms=3.4, lw=1.2, color=c, ls=ls, label=lab)
    if 'b' in panels:
        c2.plot(mid[mx.index], mx.values, marker=mk_, ms=3.4, lw=1.2, color=c, ls=ls, label=lab)
    if m in ('PG-MLP', 'Interp2D', 'GP'):
        ins.plot(mid[mx.index], mx.values, marker=mk_, ms=3.0, lw=1.1, color=c, ls=ls)
for a in (c1, c2, ins):
    a.set_xscale('log'); a.axvline(1.0, color='#999', lw=0.7, ls='--')
    a.set_xlabel(r'$\rho$')
c1.set_yscale('log'); c1.set_ylim(0.9, 15); c1.set_yticks([1, 2, 5, 10]); c1.set_yticklabels(['1', '2', '5', '10'])
c1.set_ylabel(r'$\tau_1=t_s/T^*_{1\%}$'); c1.set_title('(a) median in each bin', fontsize=8.0)
c2.set_ylim(0.95, 1.6); c2.axhline(1.0, color='#999', lw=0.6)
c2.set_title('(b) maximum (fixed, rescheduled\noff scale)', fontsize=8.0)
ins.set_ylim(0.985, 1.010); ins.axhline(1.0, color='#999', lw=0.6)
ins.set_title('(c) maximum, residual\nschedulers (zoom)', fontsize=8.0)
c1.text(1.12, 9.0, r'$\rho=1$', fontsize=7.5, color='#555')
h_, l_ = c1.get_legend_handles_labels()
fig.legend(h_, l_, loc='upper center', ncol=3, fontsize=7.0, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.83), w_pad=0.6)
fig.savefig('fig4_tau_vs_rho.png', dpi=300); plt.close(fig)

# ---------- Fig. 3: software pipeline ----------
fig, ax = plt.subplots(figsize=(6.7, 3.15)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 47)
PAD = 0.2
def box3(x, y, w, h, t, fc='white', ec='#333', fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={PAD},rounding_size=0.7', fc=fc, ec=ec, lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, linespacing=1.15, zorder=4)
def arr3(x1, y1, x2, y2, c='#333'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, lw=0.9, color=c,
                                 shrinkA=0, shrinkB=0, zorder=3))
ax.add_patch(FancyBboxPatch((0.3, 19.3), 99.4, 27.2, boxstyle='round,pad=0,rounding_size=1.2', fc='none', ec='#999', lw=0.7, ls='--'))
ax.text(1.5, 44.6, 'Offline: workstation (Python, Numba, scikit-learn)', fontsize=7.6, va='center', color='#444')
ax.add_patch(FancyBboxPatch((0.3, 0.3), 99.4, 16.2, boxstyle='round,pad=0,rounding_size=1.2', fc='none', ec='#999', lw=0.7, ls='--'))
ax.text(98.5, 14.6, 'Online: drive controller (C, once per positioning command)', fontsize=7.6, va='center', ha='right', color='#444')
y, h, w, gap = 22.8, 17, 12.6, 4.38
xs = [1.5 + k * (w + gap) for k in range(6)]
texts = ['Drive model' + NL + '(1), relay' + NL + 'cascade (3),' + NL + 'compiled' + NL + 'simulator',
         'Oracle labels:' + NL + 'Nelder' + chr(8211) + 'Mead' + NL + 'search (10),' + NL + 'parallel' + NL + 'processes',
         'Dataset:' + NL + r'$(\log_{10}\rho,\ \hat\kappa)$' + NL + r'$\rightarrow x^* - x^{\mathrm{law}}$',
         'Training:' + NL + 'residual' + NL + 'regressor' + NL + '(MLP' + NL + 'ensemble)',
         'Evaluation:' + NL + 'baselines,' + NL + 'ablation,' + NL + 'inertia' + NL + 'errors',
         'Export:' + NL + 'weights' + NL + '(JSON) and' + NL + 'C header']
for k, (x, t) in enumerate(zip(xs, texts)):
    box3(x, y, w, h, t, fc='#e6effa' if k == 3 else '#f4f4f2', ec=BLUE if k == 3 else '#333')
for k in range(5):
    arr3(xs[k] + w + PAD, y + h / 2, xs[k + 1] - PAD, y + h / 2)
y2, h2 = 2.0, 10.0
bx = [(2.0, 17.0), (25.0, 30.0), (60.5, 17.5), (83.0, 15.5)]
box3(bx[0][0], y2, bx[0][1], h2, r'Command $A$,' + NL + r'estimate $\hat\kappa$', fc='#f4f4f2')
box3(bx[1][0], y2, bx[1][1], h2, r'Scheduler: limits (4), $K^0$ (5), (6),' + NL + r'$\rho$, law (11), regressor, safeguard (12)', fc='#e6effa', ec=BLUE)
box3(bx[2][0], y2, bx[2][1], h2, 'Relay cascade' + NL + r'$R_1$' + chr(8211) + r'$R_3$ (Fig. 1)')
box3(bx[3][0], y2, bx[3][1], h2, 'DC drive', fc='#f4f4f2')
for k in range(3):
    arr3(bx[k][0] + bx[k][1] + PAD, y2 + h2 / 2, bx[k + 1][0] - PAD, y2 + h2 / 2)
ax.text((bx[1][0] + bx[1][1] + bx[2][0]) / 2, y2 + h2 / 2 + 0.8, '$K$', fontsize=7.4, ha='center', va='bottom')
xe = xs[5] + w / 2; xd = 40.0
ax.plot([xe, xe, xd], [y - PAD, 17.9, 17.9], color=BLUE, lw=0.9, solid_capstyle='butt', zorder=3)
arr3(xd, 17.9, xd, y2 + h2 + PAD, c=BLUE)
ax.text(xd - 1.2, 14.6, 'deployed parameters', fontsize=7.2, ha='right', va='center', color=BLUE)
fig.savefig('fig3_pipeline.png', dpi=300, bbox_inches='tight'); plt.close(fig)
