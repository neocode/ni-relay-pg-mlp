"""Step 5 - figures of the paper (300 dpi PNG).

fig1_structure.png           Fig. 1  relay cascade with scheduled switching coefficients
fig2_labels_sensitivity.png  Fig. 2  oracle corrections vs rho (a), sensitivity to an error of K12 (b)
fig3_transients.png          Fig. 3  transients for A = 0.05 rad, kappa = 1.3 (numbers in fig3_numbers.json)
figS1_tau_vs_rho.png         supplementary: tau1 vs rho on the 2500 test moves (not printed in the paper)

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
fig, ax = plt.subplots(figsize=(6.7, 2.5)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(-3, 34)
def box(x, y, w, h, t, fc='white', ec='#333', fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8', fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs)
def arr(x1, y1, x2, y2, t=None, dy=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, lw=0.8, color='#333'))
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, t, ha='center', va='bottom', fontsize=8)
box(0.5, 23, 18, 9, r'Move command $A$,' + NL + r'inertia estimate $\hat\kappa$', fc='#f4f4f2')
box(23, 23, 20, 9, r'Limits $E_{\max}(\hat\kappa)$,' + NL + r'analytic N-i vector $K^0$', fc='#f4f4f2')
box(47.5, 23, 25, 9, r'Scheduler: $\rho(A,\hat\kappa)$,' + NL + 'scaling law + PG-MLP', fc='#e6effa', ec=BLUE)
box(77, 23, 22, 9, r'$K = K^0\odot\exp(x)$,' + NL + 'held during the move', fc='#f4f4f2')
arr(19, 27.5, 22.5, 27.5); arr(43.5, 27.5, 47, 27.5); arr(73, 27.5, 76.5, 27.5)
box(4, 4, 13, 9, r'$R_1$: $s_1$' + NL + r'$\pm E_{2\max}$')
box(24, 4, 13, 9, r'$R_2$: $s_2$' + NL + r'$\pm E_{3\max}$')
box(44, 4, 13, 9, r'$R_3$: $s_3$' + NL + r'$\pm U_{\max}$')
box(64, 4, 15, 9, 'DC drive' + NL + r'$\theta,\ \omega,\ M$', fc='#f4f4f2')
box(84, 4, 15, 9, 'derivative' + NL + 'measurement', fc='#f4f4f2')
arr(17.5, 8.5, 23.5, 8.5, r'$E_2^*$'); arr(37.5, 8.5, 43.5, 8.5, r'$E_3^*$'); arr(57.5, 8.5, 63.5, 8.5, r'$U$')
arr(79.5, 8.5, 83.5, 8.5)
ax.plot([88, 88, 10.5], [22.6, 17.5, 17.5], color='#333', lw=0.8)
for xx in (10.5, 30.5):
    arr(xx, 17.5, xx, 13.6)
ax.text(59, 18.3, r'$K_{12}, K_{13}$ to $R_1$;  $K_{23}$ to $R_2$', fontsize=8, ha='center')
ax.plot([91.5, 91.5, 1.5, 1.5], [3.5, -1.5, -1.5, 8.5], color='#333', lw=0.8); arr(1.5, 8.5, 3.5, 8.5)
ax.text(46, -1.0, r'$E_1=\theta^*-\theta,\ E_2,\ E_3$ fed to all relays', fontsize=8, ha='center', va='bottom')
fig.savefig('fig1_structure.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------- Fig. 2: labels collapse + sensitivity ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.55))
tr = np.array(R['train']); lab = np.array([l[0] for l in R['lab_tr']])
lr = np.array([np.log10(rho(a, k)) for a, k in tr])
mk = {0.5: 'o', 0.75: 's', 1.0: '^', 1.5: 'D', 2.0: 'v'}
for k, m in mk.items():
    s = tr[:, 1] == k
    a1.plot(lr[s], lab[s, 0], m, ms=3.6, mfc='white', mec=GRAY, mew=0.7, ls='none', label=f'$\\kappa$={k}')
g = np.linspace(-2.6, 1.4, 300)
a1.plot(g, 0.5 * np.log(np.minimum(10 ** g, 1)), color=ORANGE, lw=1.4, ls='--', label='law (11)')
kh = 1.0
V, Am, J = design_limits(kh); Asat = V * V / Am + V * Am / J
gg = g[(g >= lr.min()) & (g <= lr.max())]
a1.plot(gg, [pg_predict(10 ** v * Asat, kh)[0] for v in gg], color=BLUE, lw=1.4, label='PG-MLP, $\\hat\\kappa$=1')
a1.set_xlabel('$\\log_{10}\\rho$'); a1.set_ylabel('$x_{12}=\\ln(K_{12}/K_{12}^0)$')
a1.legend(fontsize=7, ncol=2, loc='lower right', handlelength=1.6, columnspacing=0.8)
a1.set_title('(a) oracle corrections vs. dimensionless displacement', fontsize=8.5)
S = R['sens']; ds = sorted(set(x['d'] for x in S))
med = [np.median([x['tau'] for x in S if x['d'] == d]) for d in ds]
lo = [np.min([x['tau'] for x in S if x['d'] == d]) for d in ds]
hi = [np.max([x['tau'] for x in S if x['d'] == d]) for d in ds]
pc = 100 * (np.exp(ds) - 1)
a2.fill_between(pc, lo, hi, color=BLUE, alpha=0.15, lw=0, label='range, 10 cases')
a2.plot(pc, med, 'o-', color=BLUE, ms=3.5, lw=1.4, label='median')
a2.axvline(0, color='#999', lw=0.6)
a2.set_xlabel('relative error of $K_{12}$, %'); a2.set_ylabel('$\\tau=t_s/T^*$')
a2.text(-7.6, 1.36, 'overshoot', fontsize=8, color='#333'); a2.text(3.0, 1.36, 'creep', fontsize=8, color='#333')
a2.legend(fontsize=7, loc='lower left'); a2.set_title('(b) sensitivity around the optimum', fontsize=8.5)
fig.tight_layout(); fig.savefig('fig2_labels_sensitivity.png', dpi=300); plt.close(fig)

# ---------- Fig. 3: transients for a small displacement ----------
A, kap = 0.05, 1.3
cases = [('Fixed N-i (monograph)', ni_coefficients(*design_limits(1.0)), 1.0, GRAY, '-'),
         ('Rescheduled N-i', ni_coefficients(*design_limits(kap)), kap, ORANGE, '--'),
         ('PG-MLP scheduled', K_of(pg_predict(A, kap), kap), kap, BLUE, '-')]
V, Am, J = design_limits(kap); Ts = t_ref(A, V, Am, J)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.7, 2.45))
out = {}
for name, K, kh, c, ls in cases:
    r = run(A, K, kappa=kap, kappa_hat=kh, T=0.8, rec_every=20)
    rec = r['rec']; out[name] = (r['ts'], r['Mpeak'] / MMAX, r['ovs'])
    b1.plot(rec[:, 0], 1000 * (A - rec[:, 1]), color=c, ls=ls, lw=1.3, label=name)
    Mf = np.convolve(rec[:, 3], np.ones(10) / 10, mode='same')
    b2.plot(rec[:, 0], rec[:, 3], color=c, lw=0.3, alpha=0.35)
    b2.plot(rec[:, 0], Mf, color=c, ls=ls, lw=1.1, label=name)
b1.axvline(Ts, color='k', lw=0.7, ls=':'); b1.text(Ts + 0.01, 42, '$T^*$', fontsize=8)
b1.set_xlim(0, 0.8); b1.set_xlabel('t, s'); b1.set_ylabel('$E_1$, mrad'); b1.legend(fontsize=7)
b1.set_title(f'(a) position error, A = {A} rad, $\\kappa$ = {kap}', fontsize=8.5)
b2.axhline(MMAX, color='k', lw=0.6, ls=':'); b2.axhline(-MMAX, color='k', lw=0.6, ls=':')
b2.text(0.55, MMAX + 1.5, '$\\pm M_{\\max}$', fontsize=8)
b2.set_xlim(0, 0.8); b2.set_xlabel('t, s'); b2.set_ylabel('M, N·m')
b2.set_title('(b) torque: 2 ms moving average (thick), raw (thin)', fontsize=8.5)
fig.tight_layout(); fig.savefig('fig3_transients.png', dpi=300); plt.close(fig)
json.dump({k: dict(zip(('ts', 'Mpk_over_Mmax', 'overshoot'), map(float, vv))) for k, vv in out.items()} | {'Tstar': Ts},
          open('fig3_numbers.json', 'w'), indent=1)

# ---------- Fig. S1 (supplementary): tau1 vs rho on the 2500-move test set ----------
import pandas as pd
from results_io import load_results2
R2 = load_results2()
d = pd.DataFrame(R2['rows']); d = d[d.est == 'exact']
fig, ax = plt.subplots(figsize=(6.7, 2.5))
spec = [('Rescheduled N-i', 'rescheduled N-i', ORANGE, '^', 0.35),
        ('Adaptive N-i', 'amplitude-adaptive N-i', GRAY, 's', 0.6),
        ('MLP raw', 'data-driven MLP', AQUA, 'D', 0.6), ('PG-MLP', 'PG-MLP', BLUE, 'o', 0.9)]
for m, lab, c, mk_, al in spec:
    q = d[d.method == m]
    ax.plot(q.rho, q.tau1, mk_, ms=2.2, mfc='none', mec=c, mew=0.6, alpha=al, ls='none', label=lab)
ax.axvline(1.0, color='#999', lw=0.7, ls='--'); ax.text(1.08, 9.0, r'$\rho=1$', fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.85, 16)
ax.set_yticks([1, 2, 5, 10]); ax.set_yticklabels(['1', '2', '5', '10'])
ax.set_xlabel(r'dimensionless displacement $\rho$'); ax.set_ylabel(r'$\tau_1=t_s/T^*_{1\%}$')
leg = ax.legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2), handlelength=1.2, columnspacing=1.0, markerscale=2)
for lh in leg.legend_handles: lh.set_alpha(1)
fig.tight_layout(); fig.savefig('figS1_tau_vs_rho.png', dpi=300); plt.close(fig)
