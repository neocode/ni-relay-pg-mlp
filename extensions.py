"""Step 2a - additional studies (minimum settling time, further regressors, non-idealities and robustness checks;
see the mapping of tables and numbers in README.md).

* topt   : minimum 1 %-settling time T_opt of the drive model (1) under the same limits (|U| <= Umax,
           |omega| <= omega_max, |M| <= M_max, checked at the grid points and at three points inside every
           interval), computed by bisection on the settling time with a linear-programming feasibility problem
           (exact zero-order-hold discretisation, HiGHS solver); a numerical estimate, compared with the settling
           times of the controllers on the 40 labelled moves, the Fig. 5 move, the 30 worst moves of PG-MLP and
           30 random moves of the 2500-move test set.
* topt_sens : T_opt with the torque limit raised to 1.016 M_max (chattering level of the relay model), ten moves.
* base2d : two further residual baselines on the same 125 labels - 2-D interpolation of the residual
           on (log10 rho, kappa_hat) and a Gaussian-process regressor; a variant of the law with
           x13 = ln min(rho, 1); evaluated on the 2500 test moves (exact inertia ratio).
* robust : non-idealities not covered by the main study (load torque, controller sampling period,
           supply-voltage error), 500 test moves.
* seeds  : five disjoint seed groups of PG-MLP and the raw-input MLP on all 2500 test moves.
* cv     : five-fold cross-validation of the network hyper-parameters on the 125 training labels.
* ood    : displacements outside the training range, with and without clipping of the inputs.
* labels : repeatability of the oracle labels (random restarts) and flatness of the cost in x13.
* zeta   : minimum damping ratio of the outer sliding equation for every scheduler.
* irregular : PG-MLP, a Gaussian process and an interpolant of the residual trained on 125 or 60 irregular labels
           and on a coarser grid of 65 labels; a Gaussian process on raw inputs without the law.
* robust2: the 2-D interpolant and the Gaussian process of the residual under inertia estimation errors.
* designs: label designs (grid of 125 and of 65, three random samples, three Latin hypercubes, three scrambled
           Sobol samples of 125, 60 random labels) with three regressors each (PG-MLP, Gaussian process per output,
           piecewise-linear interpolant of scattered labels).
* robust3: the 2-D interpolant and the Gaussian process with the inertia margin and below 0.003 rad.
* numerics : integration step 10, 5 and 1 us: PG-MLP and the oracle labels on 20 labelled moves, oracle labels
           recomputed at 1 us; PG-MLP at 1 us on all 2500 moves and on the 101 moves with T_opt (same torque limit).
* topt_conv: T_opt with 600 instead of 300 intervals for 12 moves.
* costmap: labelling cost (10) around the oracle label for three moves (Fig. 2, c).
* step1us: labels, training and evaluation of PG-MLP with the integration step 1 us.

Output: results3.json (sections are merged, so they can be run separately).
Run:    python extensions.py [--workers N] [--only topt,base2d,...]     (about 2.5 hours on two CPU cores, mostly the linear programs of T_opt; finished T_opt cases are cached)
"""
import argparse, warnings, json, os, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.linalg import expm, toeplitz
from scipy.optimize import linprog, minimize
from scipy.interpolate import PchipInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import ni_drive as nd
from ni_drive import run, run_ext, ni_coefficients, design_limits, t_ref, t_ref_band, MMAX, WMAX
from experiment import K_of, rho, law, cost
import evaluation as ev

OUT = 'results3.json'
R = json.load(open('results.json'))
train, Ytr, test40, Y40 = ev.train, ev.Ytr, ev.test40, ev.Y40


def save(key, obj):
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data[key] = obj
    json.dump(data, open(OUT, 'w'), indent=1)


def big_set():
    """The 2500 test moves of evaluation.py (same seed and sampling)."""
    rng = np.random.default_rng(ev.SEED_BIG)
    big = [(float(10 ** rng.uniform(np.log10(0.004), np.log10(2.5))), float(rng.uniform(0.55, 1.9))) for _ in range(2000)]
    while len(big) < 2500:
        a, k = float(10 ** rng.uniform(np.log10(0.004), np.log10(2.5))), float(rng.uniform(0.55, 1.9))
        if 0.5 <= rho(a, k) <= 2.0: big.append((a, k))
    return big


# ---------------------------------------------------------------- minimum settling time (LP)
UNDECIDED = []


def _lp_feasible(A, kappa, ts, n1=300, tail=None, sub=4, mscale=1.0):
    """Is there an input |U| <= Umax that keeps |omega| <= wmax and |M| <= Mmax (checked at the grid points and
    at `sub` - 1 equally spaced points inside every interval), enters the band |theta - A| <= 0.01 A at ts,
    stays in it at the grid points and ends at rest (omega = M = 0) after the tail?"""
    b = nd.B23 / kappa
    Ac = np.array([[0, 1, 0], [0, 0, b], [0, nd.A32, nd.A33]], float)
    Bc = np.array([0, 0, nd.M3], float)
    h = ts / n1
    if tail is None:
        tail = max(0.5 * ts, 0.03)
    N = n1 + int(np.ceil(tail / h))

    def zoh(dt):
        E = expm(np.block([[Ac, Bc[:, None]], [np.zeros((1, 4))]]) * dt)
        return E[:3, :3], E[:3, 3]
    Phi, Gam = zoh(h)
    P = np.empty((N, 3)); v = Gam.copy()
    for m in range(N):
        P[m] = v; v = Phi @ v
    L = [np.tril(toeplitz(P[:, c], np.zeros(N))) for c in range(3)]   # x_k (k = 1..N) = L u
    th, w, M = L
    rows_w, rows_M = [w], [M]
    Xprev = np.stack([np.vstack([np.zeros((1, N)), Lc[:-1]]) for Lc in L])   # x_{k-1}, k = 1..N
    for j in range(1, sub):                     # states inside the intervals
        Pj, Gj = zoh(j * h / sub)
        Xj = np.einsum('ij,jkn->ikn', Pj, Xprev) + Gj[:, None, None] * np.eye(N)[None]
        rows_w.append(Xj[1]); rows_M.append(Xj[2])
    # scaled problem: variables u / Umax, rows divided by their bounds (omega / wmax, M / Mmax, theta / A)
    Wr = np.vstack(rows_w) * (nd.UMAX / nd.WMAX); Mr = np.vstack(rows_M) * (nd.UMAX / (mscale * MMAX))
    Th = th[n1 - 1:] * (nd.UMAX / A)
    nW = Wr.shape[0]
    A_ub = np.vstack([Wr, -Wr, Mr, -Mr, Th, -Th])
    b_ub = np.concatenate([np.ones(4 * nW), np.full(len(Th), 1.01), np.full(len(Th), -0.99)])
    A_eq = np.vstack([w[-1] * (nd.UMAX / nd.WMAX), M[-1] * (nd.UMAX / MMAX)])
    # interior point first (fast and robust on these dense problems), simplex variants as fall-backs
    for method in ('highs-ipm', 'highs', 'highs-ds'):
        res = linprog(np.zeros(N), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[0.0, 0.0], bounds=[(-1.0, 1.0)] * N,
                      method=method, options=dict(time_limit=60.0))
        if res.status in (0, 2):
            return res.status == 0
    UNDECIDED.append((A, kappa, ts))      # counted as feasible: T_opt can only be underestimated (conservative)
    return True


def t_opt(case, n1=300, iters=11, mscale=1.0):
    """Bisection on the settling time between an infeasible lower and a feasible upper bound."""
    A, kappa = case
    V, Am, J = design_limits(kappa)
    lo, hi = 0.85 * t_ref_band(A, V, Am, J), 1.1 * t_ref(A, V, Am, J)
    while _lp_feasible(A, kappa, lo, n1, mscale=mscale):
        lo *= 0.85
    while not _lp_feasible(A, kappa, hi, n1, mscale=mscale):
        lo, hi = hi, 1.5 * hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if _lp_feasible(A, kappa, mid, n1, mscale=mscale) else (mid, hi)
    return hi


def topt_case(args):
    i, A, kap, M, group = args
    UNDECIDED.clear()
    T = t_opt((A, kap))
    row = dict(case=i, group=group, A=A, kappa=kap, rho=rho(A, kap), topt=T, lp_undecided=len(UNDECIDED))
    V, Am, J = design_limits(kap)
    row['tref1'] = t_ref_band(A, V, Am, J); row['tref'] = t_ref(A, V, Am, J)
    ctr = ev.controllers(A, kap, M)
    for name in ('Fixed N-i', 'Rescheduled N-i', 'Adaptive N-i', 'Law (11)', 'PG-MLP'):
        K, kh = ctr[name]
        r = run(A, K, kappa=kap, kappa_hat=kh, T=4.0)
        row[name] = r['ts']; row[name + ' mpk'] = r['Mpeak'] / MMAX
    x2 = interp2d_residual([(A, kap)])[0]
    row['2-D interpolation of residual'] = run(A, ev.safeguard(K_of(x2, kap)), kappa=kap, kappa_hat=kap, T=4.0)['ts']
    if group == 'labelled':
        row['Oracle'] = run(A, K_of(Y40[i], kap), kappa=kap, kappa_hat=kap, T=4.0)['ts']
    return row


def sec_topt(workers):
    """T_opt for the 40 labelled moves, the Fig. 5 move, the 30 moves of the 2500-move set with the largest
    tau1 of PG-MLP (the worst cases) and 30 further moves of that set drawn at random (seed 2029)."""
    from results_io import load_results2
    M = ev.build(); big = big_set()
    d = pd.DataFrame(load_results2()['rows']); p = d[(d.est == 'exact') & (d.method == 'PG-MLP')]
    worst = list(p.sort_values('tau1', ascending=False).case.values[:30])
    rnd = [int(c) for c in np.random.default_rng(2029).choice([c for c in range(len(big)) if c not in worst], 30, replace=False)]
    jobs = [(i, a, k, M, 'labelled') for i, (a, k) in enumerate(test40)] + [(-1, 0.05, 1.3, M, 'fig5')]
    jobs += [(int(c), *big[c], M, 'worst30') for c in worst] + [(c, *big[c], M, 'random30') for c in rnd]
    cache = 'topt_cache.jsonl'           # finished cases survive an interruption
    done = {}
    if os.path.exists(cache):
        for line in open(cache):
            r = json.loads(line); done[(r['group'], r['case'])] = r
    todo = [j for j in jobs if (j[4], j[0]) not in done]
    with Pool(workers) as p_, open(cache, 'a') as f:
        for r in p_.imap_unordered(topt_case, todo, chunksize=1):
            f.write(json.dumps(r) + '\n'); f.flush(); done[(r['group'], r['case'])] = r
    rows = [done[(j[4], j[0])] for j in jobs]
    chk = [dict(case=i, t300=rows[i]['topt'], t600=t_opt(test40[i], n1=600)) for i in (0, 16, 32)]
    save('topt', dict(rows=rows, check=chk))


# ---------------------------------------------------------------- further residual baselines
Rtr = ev.Rtr
LR = ev.phy(train)[:, 0].reshape(5, 25)            # log10 rho of the training grid, one row per kappa
KG = ev.k_grid
_rows = [[PchipInterpolator(LR[r], Rtr.reshape(5, 25, 2)[r, :, j], extrapolate=False) for j in range(2)]
         for r in range(5)]
_gp = None


def interp2d_residual(cs):
    """PCHIP of the residual along log10(rho) inside each kappa row of the training grid, then linear
    interpolation between the two neighbouring rows (inputs clipped to the grid)."""
    out = []
    for a, k in cs:
        k = float(np.clip(k, KG[0], KG[-1])); r = min(int(np.searchsorted(KG, k, side='right')) - 1, 3)
        w = (k - KG[r]) / (KG[r + 1] - KG[r]); lr = np.log10(rho(a, k))
        vals = []
        for rr in (r, r + 1):
            x = float(np.clip(lr, LR[rr, 0], LR[rr, -1]))
            vals.append([float(_rows[rr][j](x)) for j in range(2)])
        out.append((1 - w) * np.array(vals[0]) + w * np.array(vals[1]))
    return ev.lawv(cs) + np.array(out)


def gp_residual(cs):
    global _gp
    if _gp is None:
        sp = StandardScaler().fit(ev.phy(train))
        kern = ConstantKernel(1.0) * RBF([1.0, 1.0]) + WhiteKernel(1e-4)
        _gp = (sp, GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=3, random_state=0)
               .fit(sp.transform(ev.phy(train)), Rtr))
    sp, g = _gp
    return ev.lawv(cs) + g.predict(sp.transform(ev.phy(cs)))


def law_x13(cs):
    return np.array([[0.5 * np.log(min(rho(a, k), 1.0)), np.log(min(rho(a, k), 1.0))] for a, k in cs])


EXTRA = {'2-D interpolation of residual': interp2d_residual, 'GP residual': gp_residual,
         'Law, x13 = ln min(rho,1)': law_x13}


def base_case(args):
    i, A, kap, tag = args
    rows = []
    for name, f in EXTRA.items():
        x = f([(A, kap)])[0]
        K0 = K_of(x, kap); K = ev.safeguard(K0)
        rows.append(dict(case=i, rho=rho(A, kap), est=tag, method=name, guard=int(np.any(K != K0)),
                         **ev.evaluate(A, kap, kap, K)))
    return rows


def sec_base2d(workers):
    gp_residual(train[:1])        # fit once before forking
    big = big_set()
    jobs = [(i, a, k, 'exact') for i, (a, k) in enumerate(big)] + [(i, a, k, 'exact40') for i, (a, k) in enumerate(test40)]
    with Pool(workers) as p:
        rows = sum(p.map(base_case, jobs, chunksize=20), [])
    acc = {}
    for name, f in EXTRA.items():
        P = f(test40)
        acc[name] = dict(rmse=[float(v) for v in np.sqrt(np.mean((P - Y40) ** 2, axis=0))],
                         max_rel=[float(v) for v in (np.exp(np.abs(P - Y40)) - 1).max(axis=0)])
    save('base2d', dict(rows=rows, acc=acc, gp_kernel=str(_gp[1].kernel_)))


# ---------------------------------------------------------------- non-idealities
SCEN = [('nominal', dict()),
        ('load +0.25 Mmax', dict(Mc=0.25 * MMAX)), ('load -0.25 Mmax', dict(Mc=-0.25 * MMAX)),
        ('sampling 50 us', dict(hold=5)), ('sampling 100 us', dict(hold=10)), ('sampling 200 us', dict(hold=20)),
        ('supply -10 %', dict(u_scale=0.9)), ('supply +10 %', dict(u_scale=1.1))]


def robust_case(args):
    i, A, kap, M = args
    rows = []
    ctr = ev.controllers(A, kap, M)
    V, Am, J = design_limits(kap); T1 = t_ref_band(A, V, Am, J)
    for name in ('Fixed N-i', 'Rescheduled N-i', 'Adaptive N-i', 'PG-MLP'):
        K, kh = ctr[name]
        for sc, kw in SCEN:
            r = run_ext(A, K, kappa=kap, kappa_hat=kh, **kw)
            rows.append(dict(case=i, rho=rho(A, kap), method=name, scen=sc, tau1=r['ts'] / T1, ovs=r['ovs'],
                             mpk=r['Mpeak'] / MMAX, wpk=r['wpk'] / WMAX))
    return rows


def sec_robust(workers):
    M = ev.build()
    big = big_set()[:500]
    with Pool(workers) as p:
        rows = sum(p.map(robust_case, [(i, a, k, M) for i, (a, k) in enumerate(big)], chunksize=10), [])
    save('robust', rows)


# ---------------------------------------------------------------- seed groups on all 2500 moves
def seed_case(args):
    g, i, a, k, Mg = args
    out = []
    for m in ('PG-MLP', 'MLP raw'):
        K = ev.safeguard(K_of(ev.predict(m, [(a, k)], Mg)[0], k))
        out.append(dict(group=g, case=i, method=m, **ev.evaluate(a, k, k, K)))
    return out


def sec_seeds(workers):
    big = big_set(); rows = []
    for g in range(5):
        Mg = ev.build(range(5 * g, 5 * g + 5))
        with Pool(workers) as p:
            rows += sum(p.map(seed_case, [(g, i, a, k, Mg) for i, (a, k) in enumerate(big)], chunksize=50), [])
    save('seeds', rows)


# ---------------------------------------------------------------- cross-validation of hyper-parameters
def sec_cv(workers):
    X = ev.phy(train); out = []
    for hid in ((8, 8), (16, 16), (32, 32), (16,), (16, 16, 16)):
        for alpha in (1e-4, 1e-3, 1e-2):
            err = []
            for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
                sc = StandardScaler().fit(X[tr])
                ms = [MLPRegressor(hidden_layer_sizes=hid, activation='tanh', solver='lbfgs', alpha=alpha,
                                   max_iter=5000, random_state=s).fit(sc.transform(X[tr]), Rtr[tr]) for s in range(5)]
                P = np.mean([m.predict(sc.transform(X[va])) for m in ms], axis=0)
                err.append((P - Rtr[va]) ** 2)
            e = np.vstack(err)
            out.append(dict(hidden=list(hid), alpha=alpha, rmse_x12=float(np.sqrt(e[:, 0].mean())),
                            rmse_x13=float(np.sqrt(e[:, 1].mean()))))
    save('cv', out)


# ---------------------------------------------------------------- displacements outside the training range
def ood_case(args):
    i, A, kap, M = args
    rows = []
    for clip in (False, True):
        kh = kap
        if clip:
            Ac = float(np.clip(A, 0.003, 3.0)); kc = float(np.clip(kh, 0.5, 2.0))
            x = ev.predict('PG-MLP', [(Ac, kc)], M)[0]
            x = np.array([0.5 * np.log(min(rho(A, kh), 1.0)), 0.0]) + (x - law(Ac, kc))
        else:
            x = ev.predict('PG-MLP', [(A, kh)], M)[0]
        K = ev.safeguard(K_of(x, kh))
        rows.append(dict(case=i, A=A, kappa=kap, rho=rho(A, kap), method='PG-MLP clipped' if clip else 'PG-MLP',
                         **ood_eval(A, kap, kh, K)))
    for name in ('Rescheduled N-i', 'Adaptive N-i'):
        K, kh = ev.controllers(A, kap, M)[name]
        rows.append(dict(case=i, A=A, kappa=kap, rho=rho(A, kap), method=name, **ood_eval(A, kap, kh, K)))
    return rows


def ood_eval(A, kap, kh, K):
    """As evaluation.evaluate, with a horizon long enough for moves above 3 rad (max(4 s, 1.6 T* + 0.4 s))."""
    V, Am, J = design_limits(kap)
    r = run(A, K, kappa=kap, kappa_hat=kh, T=max(4.0, 1.6 * t_ref(A, V, Am, J) + 0.4))
    return dict(tau=r['ts'] / t_ref(A, V, Am, J), tau1=r['ts'] / t_ref_band(A, V, Am, J),
                ovs=r['ovs'], mpk=r['Mpeak'] / MMAX, wpk=r['wpk'] / WMAX)


def sec_ood(workers):
    M = ev.build(); rng = np.random.default_rng(2028)
    cases = [(float(10 ** rng.uniform(-3.5, np.log10(0.003))), float(rng.uniform(0.55, 1.9))) for _ in range(200)]
    cases += [(float(10 ** rng.uniform(np.log10(3.0), np.log10(10.0))), float(rng.uniform(0.55, 1.9))) for _ in range(200)]
    with Pool(workers) as p:
        rows = sum(p.map(ood_case, [(i, a, k, M) for i, (a, k) in enumerate(cases)]), [])
    save('ood', rows)


# ---------------------------------------------------------------- repeatability of the labels, flatness in x13
def restart_case(args):
    j, A, kap = args
    rng = np.random.default_rng(100 + j); out = []
    for _ in range(3):
        x0 = law(A, kap) + rng.normal(0, 0.3, 2)
        r = minimize(cost, x0, args=(A, kap), method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-4, maxfev=200))
        out.append(dict(x=[float(v) for v in r.x], f=float(r.fun)))
    flat = []
    for dx in (-0.5, -0.25, 0.0, 0.25, 0.5):
        x = np.array(Ytr[j]) + np.array([0.0, dx])
        flat.append(dict(dx13=dx, cost=float(cost(x, A, kap))))
    return dict(j=j, A=A, kappa=kap, rho=rho(A, kap), label=[float(v) for v in Ytr[j]],
                label_cost=float(R['lab_tr'][j][1]), restarts=out, flat=flat)


def sec_labels(workers):
    idx = list(range(0, 125, 6))
    with Pool(workers) as p:
        rows = p.map(restart_case, [(j, *train[j]) for j in idx])
    save('labels', rows)


# ---------------------------------------------------------------- damping ratio of the outer sliding equation
def sec_zeta(workers):
    M = ev.build(); big = big_set(); out = {}
    z = lambda K: K[0] / (2 * np.sqrt(K[1]))
    out['oracle labels (train + test)'] = float(min(z(K_of(y, c[1])) for y, c in
                                                   zip(list(Ytr) + list(Y40), train + test40)))
    for name in ('Law (11)', 'PG-MLP'):
        raw_z = [z(K_of(ev.predict(name, [c], M)[0], c[1])) for c in big]
        out[name + ' before safeguard'] = float(min(raw_z))
        out[name + ' safeguard active'] = int(sum(v < ev.ZETA_MIN for v in raw_z))
    out['PG-MLP after safeguard, all evaluation conditions'] = float(min(
        z(ev.safeguard(K_of(ev.predict('PG-MLP', [(a, kh)], M)[0], kh)))
        for a, k in big for kh in (k, 0.9 * k, 1.1 * k)))
    save('zeta', out)


# ---------------------------------------------------------------- sensitivity of T_opt to the torque limit
def _topt_sens_case(args):
    i, A, kap = args
    return dict(case=i, A=A, kappa=kap, topt_1016=t_opt((A, kap), mscale=1.016))


def sec_topt_sens(workers):
    """T_opt with the torque limit raised to 1.016 M_max (the chattering level of the relay model) for the ten
    worst moves of PG-MLP; compare with the T_opt rows of sec_topt."""
    rows = json.load(open(OUT))['topt']['rows']
    worst = [r for r in rows if r['group'] == 'worst30'][:10]
    with Pool(workers) as p:
        out = p.map(_topt_sens_case, [(r['case'], r['A'], r['kappa']) for r in worst], chunksize=1)
    save('topt_sens', out)


# ---------------------------------------------------------------- regressors on irregular and fewer labels
def _grid_interp(lr_rows, res_rows, kg):
    """2-D interpolant of the residual on a regular grid: PCHIP along log10(rho) in each kappa row, linear in kappa."""
    pch = [[PchipInterpolator(lr_rows[r], res_rows[r, :, j], extrapolate=False) for j in range(2)] for r in range(len(kg))]

    def f(cs):
        out = []
        for a, k in cs:
            k = float(np.clip(k, kg[0], kg[-1])); r = min(int(np.searchsorted(kg, k, side='right')) - 1, len(kg) - 2)
            w = (k - kg[r]) / (kg[r + 1] - kg[r]); lr = np.log10(rho(a, k)); v = []
            for rr in (r, r + 1):
                x = float(np.clip(lr, lr_rows[rr, 0], lr_rows[rr, -1]))
                v.append([float(pch[rr][j](x)) for j in range(2)])
            out.append((1 - w) * np.array(v[0]) + w * np.array(v[1]))
        return ev.lawv(cs) + np.array(out)
    return f


def _scattered_interp(cases, res):
    """Piecewise-linear interpolant of the residual on scattered (irregular) labels in standardised
    (log10 rho, kappa_hat), nearest-neighbour value outside the convex hull."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    sc = StandardScaler().fit(ev.phy(cases)); X = sc.transform(ev.phy(cases))
    lin, near = LinearNDInterpolator(X, res), NearestNDInterpolator(X, res)

    def f(cs):
        Z = sc.transform(ev.phy(cs)); y = lin(Z); m = np.isnan(y[:, 0]); y[m] = near(Z[m])
        return ev.lawv(cs) + y
    return f


def _fit_gp(cases, res, raw=False):
    feat = ev.raw if raw else ev.phy
    sc = StandardScaler().fit(feat(cases))
    kern = ConstantKernel(1.0) * RBF([1.0, 1.0]) + WhiteKernel(1e-4)
    g = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(sc.transform(feat(cases)), res)
    return (lambda cs: g.predict(sc.transform(feat(cs)))) if raw else (lambda cs: ev.lawv(cs) + g.predict(sc.transform(feat(cs))))


def _mlp(cases, res):
    sc = StandardScaler().fit(ev.phy(cases)); ms = ev.mlp(sc.transform(ev.phy(cases)), res, range(5))
    return lambda cs: ev.lawv(cs) + np.mean([m.predict(sc.transform(ev.phy(cs))) for m in ms], axis=0)


def _sim_rows(args):
    i, A, kap, kh, xs, tag = args
    out = []
    for name, x in xs.items():
        out.append(dict(case=i, rho=rho(A, kap), est=tag, method=name, **ev.evaluate(A, kap, kh, ev.safeguard(K_of(x, kh)))))
    return out


def _evaluate_schedulers(fs, cases_est, workers):
    """Evaluate the schedulers fs {name: f(cs) -> x} on cases_est = [(i, A, kappa, kappa_hat, tag)]."""
    cs = [(a, kh) for _, a, _, kh, _ in cases_est]
    X = {n: f(cs) for n, f in fs.items()}
    jobs = [(i, a, k, kh, {n: X[n][j] for n in fs}, tag) for j, (i, a, k, kh, tag) in enumerate(cases_est)]
    with Pool(workers) as p:
        return sum(p.map(_sim_rows, jobs, chunksize=50), [])


def sec_irregular(workers):
    """Is the network needed? PG-MLP, a Gaussian process and an interpolant of the same residual trained on
    (a) 125 irregular labels (random moves, seed 2030), (b) the first 60 of them, (c) a coarser regular grid of
    65 labels (every second displacement of the training grid); plus a Gaussian process on raw inputs without the
    law (all 125 grid labels). Evaluated on the 2500 test moves with the exact inertia ratio."""
    from experiment import oracle
    rng = np.random.default_rng(2030)
    irr = [(float(10 ** rng.uniform(np.log10(0.003), np.log10(3.0))), float(rng.uniform(0.5, 2.0))) for _ in range(125)]
    with Pool(workers) as p:
        lab = p.map(oracle, irr)
    Yi = np.array([l[0] for l in lab]); Ri = Yi - ev.lawv(irr)
    sel = np.arange(0, 25, 2)                                  # 13 of the 25 displacements
    g65 = [train[r * 25 + c] for r in range(5) for c in sel]
    R65 = ev.Rtr.reshape(5, 25, 2)[:, sel]; LR65 = LR[:, sel]
    fs = {'irregular 125: PG-MLP': _mlp(irr, Ri), 'irregular 125: Gaussian process': _fit_gp(irr, Ri),
          'irregular 125: linear interpolant': _scattered_interp(irr, Ri),
          'irregular 60: PG-MLP': _mlp(irr[:60], Ri[:60]), 'irregular 60: Gaussian process': _fit_gp(irr[:60], Ri[:60]),
          'irregular 60: linear interpolant': _scattered_interp(irr[:60], Ri[:60]),
          'grid 65: PG-MLP': _mlp(g65, R65.reshape(-1, 2)), 'grid 65: Gaussian process': _fit_gp(g65, R65.reshape(-1, 2)),
          'grid 65: 2-D interpolant': _grid_interp(LR65, R65, KG),
          'grid 125: Gaussian process on raw inputs, no law': _fit_gp(train, Ytr, raw=True)}
    big = big_set()
    rows = _evaluate_schedulers(fs, [(i, a, k, k, 'exact') for i, (a, k) in enumerate(big)], workers)
    acc = {n: [float(v) for v in np.sqrt(np.mean((f(test40) - Y40) ** 2, axis=0))] for n, f in fs.items()}
    save('irregular', dict(rows=rows, rmse40=acc, irr=irr, lab_irr=lab))


def sec_robust2(workers):
    """The 2-D interpolant and the Gaussian process of the residual (125 grid labels) under inertia estimation
    errors: delta = -0.1 and +0.1 (first 500 moves) and random delta (all 2500 moves, as evaluation.py)."""
    gp_residual(train[:1])
    rng = np.random.default_rng(ev.SEED_BIG)        # same draws as evaluation.py: moves, then the errors
    big = [(float(10 ** rng.uniform(np.log10(0.004), np.log10(2.5))), float(rng.uniform(0.55, 1.9))) for _ in range(2000)]
    while len(big) < 2500:
        a, k = float(10 ** rng.uniform(np.log10(0.004), np.log10(2.5))), float(rng.uniform(0.55, 1.9))
        if 0.5 <= rho(a, k) <= 2.0: big.append((a, k))
    assert big == big_set()
    deltas = rng.uniform(-0.1, 0.1, len(big))
    cases = [(i, a, k, k * (1 + deltas[i]), 'err10') for i, (a, k) in enumerate(big)]
    cases += [(i, a, k, 0.9 * k, 'under10') for i, (a, k) in enumerate(big[:500])]
    cases += [(i, a, k, 1.1 * k, 'over10') for i, (a, k) in enumerate(big[:500])]
    fs = {'2-D interpolation of residual': interp2d_residual, 'GP residual': gp_residual}
    save('robust2', _evaluate_schedulers(fs, cases, workers))


# ---------------------------------------------------------------- designs of the labels (several samples)
def _fit_gp_sep(cases, res):
    """Gaussian process with separate hyper-parameters for x12 and x13 (residual on (log10 rho, kappa_hat))."""
    sc = StandardScaler().fit(ev.phy(cases)); X = sc.transform(ev.phy(cases)); gs = []
    for j in range(2):
        kern = ConstantKernel(1.0) * RBF([1.0, 1.0]) + WhiteKernel(1e-4)
        gs.append(GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(X, res[:, j]))
    return lambda cs: ev.lawv(cs) + np.c_[[g.predict(sc.transform(ev.phy(cs))) for g in gs]].T


def _design(kind, n, seed):
    """n training cases in (log10 A, kappa): uniform random, Latin hypercube or scrambled Sobol."""
    from scipy.stats import qmc
    lo, hi = np.array([np.log10(0.003), 0.5]), np.array([np.log10(3.0), 2.0])
    if kind == 'random':
        u = np.random.default_rng(seed).uniform(size=(n, 2))
    elif kind == 'lhs':
        u = qmc.LatinHypercube(d=2, seed=seed).random(n)
    else:
        with warnings.catch_warnings():          # 125 is not a power of 2; kept equal to the size of the grid
            warnings.simplefilter('ignore', UserWarning)
            u = qmc.Sobol(d=2, scramble=True, seed=seed).random(n)
    x = lo + u * (hi - lo)
    return [(float(10 ** a), float(k)) for a, k in x]


def sec_designs(workers):
    """Label designs with 125 labels each: the regular grid, three uniform random samples (seeds 2030-2032), a Latin
    hypercube and a scrambled Sobol sample (two further independent Latin hypercubes and Sobol samples), plus two smaller sets (every second displacement of the grid, 65 labels,
    and the first 60 random labels of seed 2030); regressors PG-MLP, Gaussian process (separate per output) and a
    piecewise-linear interpolant of the scattered labels (also applied to the grid for a like-for-like comparison).
    Evaluated on the 2500 test moves with the exact inertia ratio."""
    from experiment import oracle
    old = json.load(open(OUT))
    prev, done = old.get('irregular', {}), old.get('designs', {}).get('labels', {})   # labels computed before
    sel = np.arange(0, 25, 2)                                  # coarser grid: 13 of the 25 displacements
    sets = {'grid': (list(train), np.array(Ytr)),
            'grid 65': ([train[r * 25 + c] for r in range(5) for c in sel], np.array(Ytr).reshape(5, 25, 2)[:, sel].reshape(-1, 2))}
    if prev:
        sets['random 2030'] = ([tuple(c) for c in prev['irr']], np.array([l[0] for l in prev['lab_irr']]))
        sets['random 2030, first 60'] = (sets['random 2030'][0][:60], sets['random 2030'][1][:60])
    designs = [('random 2031', 'random', 2031), ('random 2032', 'random', 2032), ('Latin hypercube', 'lhs', 2034),
               ('Sobol', 'sobol', 2035), ('Latin hypercube 2036', 'lhs', 2036), ('Latin hypercube 2037', 'lhs', 2037),
               ('Sobol 2038', 'sobol', 2038), ('Sobol 2039', 'sobol', 2039)]
    if not prev:
        designs.insert(0, ('random 2030', 'random', 2030))
    labels_out = {}
    for name, kind, seed in designs:
        cs = _design(kind, 125, seed)
        if name in done and np.allclose(done[name]['cases'], cs):
            lab = done[name]['labels']
        else:
            with Pool(workers) as p:
                lab = p.map(oracle, cs)
        sets[name] = (cs, np.array([l[0] for l in lab])); labels_out[name] = dict(cases=cs, labels=lab)
    fs = {}
    for name, (cs, Y) in sets.items():
        Rres = Y - ev.lawv(cs)
        fs[name + ': PG-MLP'] = _mlp(cs, Rres)
        fs[name + ': Gaussian process'] = _fit_gp_sep(cs, Rres)
        fs[name + ': linear interpolant'] = _scattered_interp(cs, Rres)
    big = big_set()
    rows = _evaluate_schedulers(fs, [(i, a, k, k, 'exact') for i, (a, k) in enumerate(big)], workers)
    acc = {n: [float(v) for v in np.sqrt(np.mean((f(test40) - Y40) ** 2, axis=0))] for n, f in fs.items()}
    save('designs', dict(rows=rows, rmse40=acc, labels=labels_out,
                         cases={n: [list(c) for c in cs] for n, (cs, Y) in sets.items()}))


def sec_robust3(workers):
    """The 2-D grid interpolant and the Gaussian process of the residual with the inertia margin kappa_hat / 0.9
    (random errors, 2500 moves) and below the training range of displacements (200 moves of ood)."""
    gp_residual(train[:1])
    rng = np.random.default_rng(ev.SEED_BIG)
    big = [(float(10 ** rng.uniform(np.log10(0.004), np.log10(2.5))), float(rng.uniform(0.55, 1.9))) for _ in range(2000)]
    while len(big) < 2500:
        a, k = float(10 ** rng.uniform(np.log10(0.004), np.log10(2.5))), float(rng.uniform(0.55, 1.9))
        if 0.5 <= rho(a, k) <= 2.0: big.append((a, k))
    deltas = rng.uniform(-0.1, 0.1, len(big))
    cases = [(i, a, k, k * (1 + deltas[i]) / 0.9, 'err10 margin') for i, (a, k) in enumerate(big)]
    rng2 = np.random.default_rng(2028)
    small = [(float(10 ** rng2.uniform(-3.5, np.log10(0.003))), float(rng2.uniform(0.55, 1.9))) for _ in range(200)]
    cases += [(i, a, k, k, 'below 0.003 rad') for i, (a, k) in enumerate(small)]
    fs = {'2-D interpolation of residual': interp2d_residual, 'GP residual': gp_residual}
    save('robust3', _evaluate_schedulers(fs, cases, workers))


# ---------------------------------------------------------------- integration step, equal torque limits, T_opt grid
def _cost_dt(x, A, kap, dt):
    """Labelling cost (10) with the integration step dt."""
    r = run(A, K_of(x, kap), kappa=kap, kappa_hat=kap, dt=dt)
    if not np.isfinite(r['ts']):
        return 20.0
    return (r['ts'] / r['Tref'] + 50 * max(0.0, r['ovs'] - 0.005) + 5 * max(0.0, r['Mpeak'] / MMAX - 1.05)
            + 1e-3 * float(np.dot(x, x)))


def _numerics_case(args):
    j, A, kap, M = args
    out = dict(j=j, A=A, kappa=kap, rho=rho(A, kap), label10=[float(v) for v in Y40[j]])
    Kpg = ev.safeguard(K_of(ev.predict('PG-MLP', [(A, kap)], M)[0], kap))
    for dt in (1e-5, 5e-6, 1e-6):
        out[f'pg {dt:.0e}'] = ev.evaluate(A, kap, kap, Kpg, dt=dt)
        out[f'oracle10 {dt:.0e}'] = ev.evaluate(A, kap, kap, K_of(Y40[j], kap), dt=dt)
    best = None                              # oracle at 1 us: the same four starts as experiment.oracle
    for x0 in ([0, 0], law(A, kap), [-1, 0], [-2, -1]):
        r = minimize(_cost_dt, np.array(x0, float), args=(A, kap, 1e-6), method='Nelder-Mead',
                     options=dict(xatol=1e-3, fatol=1e-4, maxfev=200))
        if best is None or r.fun < best.fun:
            best = r
    out['label1'] = [float(v) for v in best.x]; out['cost1'] = float(best.fun)
    out['cost10 at 1e-06'] = float(_cost_dt(np.array(Y40[j]), A, kap, 1e-6))
    out['oracle1 1e-06'] = ev.evaluate(A, kap, kap, K_of(best.x, kap), dt=1e-6)
    return out


def _pg_dt_case(args):
    tag, i, A, kap, M = args
    Kpg = ev.safeguard(K_of(ev.predict('PG-MLP', [(A, kap)], M)[0], kap))
    return dict(tag=tag, case=i, A=A, kappa=kap, rho=rho(A, kap), **ev.evaluate(A, kap, kap, Kpg, dt=1e-6))


def sec_numerics(workers):
    """Integration step: PG-MLP and the oracle labels at 10, 5 and 1 us on 20 of the 40 labelled moves, new oracle
    labels computed at 1 us for these moves; PG-MLP (trained on 10 us labels) at 1 us on all 2500 test moves and
    on the 101 moves with T_opt, where the chattering torque excess vanishes, so that t_s and T_opt refer to the
    same torque limit."""
    M = ev.build()
    with Pool(workers) as p:
        lab = p.map(_numerics_case, [(j, *test40[j], M) for j in range(0, 40, 2)], chunksize=1)
    big = big_set()
    trows = json.load(open(OUT))['topt']['rows']
    jobs = [('test', i, a, k, M) for i, (a, k) in enumerate(big)]
    jobs += [('topt', n, r['A'], r['kappa'], M) for n, r in enumerate(trows)]
    with Pool(workers) as p:
        rows = p.map(_pg_dt_case, jobs, chunksize=20)
    for r in rows:
        if r['tag'] == 'topt':
            tr = trows[r['case']]
            r.update(group=tr['group'], topt=tr['topt'], ts=r['tau1'] * tr['tref1'], ts10=tr['PG-MLP'])
    save('numerics', dict(labels=lab, rows=rows))


def _topt_conv_case(args):
    n, A, kap, t300 = args
    return dict(row=n, A=A, kappa=kap, rho=rho(A, kap), t300=t300, t600=t_opt((A, kap), n1=600))


def sec_topt_conv(workers):
    """Discretisation of the LP: T_opt with 600 instead of 300 intervals for 12 moves: small, medium and large
    rho at low and high inertia ratio among the labelled moves, and the 6 moves with the largest t_s / T_opt of
    PG-MLP."""
    rows = json.load(open(OUT))['topt']['rows']
    lab = [(n, r) for n, r in enumerate(rows) if r['group'] == 'labelled']
    pick = []
    for lo, hi in ((0, 0.05), (0.05, 1.0), (1.0, 1e9)):
        cand = [(n, r) for n, r in lab if lo <= r['rho'] < hi]
        pick += [min(cand, key=lambda t: t[1]['kappa']), max(cand, key=lambda t: t[1]['kappa'])]
    other = sorted([(n, r) for n, r in enumerate(rows) if r['group'] != 'fig5' and (n, r) not in pick],
                   key=lambda t: -t[1]['PG-MLP'] / t[1]['topt'])[:6]
    pick += other
    with Pool(workers) as p:
        out = p.map(_topt_conv_case, [(n, r['A'], r['kappa'], r['topt']) for n, r in pick], chunksize=1)
    save('topt_conv', out)


def _costmap_case(args):
    j, A, kap = args
    d12 = np.linspace(-0.15, 0.15, 31); d13 = np.linspace(-2.0, 1.5, 36)
    F = [[float(cost(np.array(Y40[j]) + np.array([a, b]), A, kap)) for a in d12] for b in d13]
    return dict(j=j, A=A, kappa=kap, rho=rho(A, kap), label=[float(v) for v in Y40[j]], d12=d12.tolist(),
                d13=d13.tolist(), F=F)


def sec_costmap(workers):
    """Labelling cost (10) around the oracle label (x12, x13) for three labelled moves: small, medium and large
    rho (the form of the region of almost optimal coefficients)."""
    order = sorted(range(40), key=lambda j: rho(*test40[j]))
    idx = [order[3], order[20], order[36]]
    with Pool(workers) as p:
        out = p.map(_costmap_case, [(j, *test40[j]) for j in idx], chunksize=1)
    save('costmap', out)


def _oracle_1us(case):
    """Oracle label (best of the four Nelder-Mead searches of experiment.oracle) with the integration step 1 us."""
    A, kap = case
    best = None
    for x0 in ([0, 0], law(A, kap), [-1, 0], [-2, -1]):
        r = minimize(_cost_dt, np.array(x0, float), args=(A, kap, 1e-6), method='Nelder-Mead',
                     options=dict(xatol=1e-3, fatol=1e-4, maxfev=200))
        if best is None or r.fun < best.fun:
            best = r
    return [float(v) for v in best.x], float(best.fun)


def _sim1_case(args):
    i, A, kap, x = args
    return dict(case=i, A=A, kappa=kap, rho=rho(A, kap), **ev.evaluate(A, kap, kap, ev.safeguard(K_of(x, kap)), dt=1e-6))


def sec_step1us(workers):
    """The whole pipeline with the integration step 1 us instead of 10 us: oracle labels for the 125 training
    cases, the PG-MLP ensemble trained on them, evaluation on the 2500 test moves at 1 us."""
    with Pool(workers) as p:
        lab = p.map(_oracle_1us, train, chunksize=2)
    Y1 = np.array([l[0] for l in lab])
    f = _mlp(train, Y1 - ev.lawv(train))
    big = big_set(); X = f(big)
    with Pool(workers) as p:
        rows = p.map(_sim1_case, [(i, a, k, X[i]) for i, (a, k) in enumerate(big)], chunksize=20)
    d = Y1 - np.array(Ytr)
    save('step1us', dict(labels=lab, rows=rows, dx12_median=float(np.median(np.abs(d[:, 0]))),
                         dx12_max=float(np.abs(d[:, 0]).max())))


SECTIONS = dict(topt=sec_topt, base2d=sec_base2d, robust=sec_robust, seeds=sec_seeds, cv=sec_cv, ood=sec_ood,
                labels=sec_labels, zeta=sec_zeta, topt_sens=sec_topt_sens, irregular=sec_irregular, robust2=sec_robust2,
                designs=sec_designs, robust3=sec_robust3,
                numerics=sec_numerics, topt_conv=sec_topt_conv, costmap=sec_costmap,
                step1us=sec_step1us)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--only', default=','.join(SECTIONS))
    args = ap.parse_args()
    for key in args.only.split(','):
        t0 = time.time(); SECTIONS[key](args.workers)
        print(f'{key}: {time.time() - t0:.0f} s', flush=True)
