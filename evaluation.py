"""Step 2 - closed-loop evaluation of all controllers (results tables of the paper and numbers quoted in the text;
see the mapping in README.md).

Uses the oracle labels of experiment.py (results.json) and:
* builds the schedulers: law (11), 2-D look-up table on raw inputs, MLP on raw inputs, the physics-guided
  PG-MLP (law + residual ensemble, Eq. (12)) and the ablation variants;
* adds the analytic baselines: fixed N-i design, rescheduled N-i design, amplitude-adaptive N-i design;
* evaluates every controller on 2500 test moves (seed 2027: 2000 moves log-uniform in A and uniform in
  kappa, plus 500 moves from the same distribution retained only if 0.5 <= rho <= 2), with the exact
  inertia ratio, a random estimation error delta ~ U[-0.1, 0.1], and the boundary errors delta = -0.1 and
  +0.1 (first 500 moves);
* evaluates the conservative margin kappa_safe = kappa_hat / (1 - delta_max), delta_max = 0.1;
* evaluates the oracle on the 40 labelled moves;
* retrains PG-MLP and the raw-input MLP with five disjoint seed groups (seeds 0-24, 500 moves);
* computes the RMSE of the predicted corrections on the 40 labelled moves;
* repeats four moves with integration steps of 20, 10, 5, 2.5 and 1 us.

Output: results2.json.gz.
Run:    python evaluation.py [--workers N]      (about 7 min on two CPU cores)
"""
import argparse, json, time
import numpy as np
from multiprocessing import Pool
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from ni_drive import run, ni_coefficients, design_limits, t_ref, t_ref_band, MMAX, WMAX
from experiment import K_of, rho, law
from results_io import save_results2

ZETA_MIN = 1.0               # safeguard of Eq. (12)
MARGIN = 1.0 / 0.9           # kappa_safe = kappa_hat / (1 - delta_max), delta_max = 0.1
SEED_BIG = 2027              # 2500-move test set and the random inertia errors

R = json.load(open('results.json'))
train = [tuple(c) for c in R['train']]; Ytr = np.array([l[0] for l in R['lab_tr']])
test40 = [tuple(c) for c in R['test']]; Y40 = np.array([l[0] for l in R['lab_te']])
A_grid = np.logspace(np.log10(0.003), np.log10(3.0), 25); k_grid = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
raw = lambda cs: np.array([[np.log10(a), k] for a, k in cs])
phy = lambda cs: np.array([[np.log10(rho(a, k)), k] for a, k in cs])
lawv = lambda cs: np.array([law(a, k) for a, k in cs])
Rtr = Ytr - lawv(train)      # residual of the law (11)


def mlp(X, Y, seeds):
    return [MLPRegressor(hidden_layer_sizes=(16, 16), activation='tanh', solver='lbfgs', alpha=1e-3,
                         max_iter=5000, random_state=s).fit(X, Y) for s in seeds]


def build(seeds=range(5)):
    """Train all network-based schedulers with the given seeds."""
    sx = StandardScaler().fit(raw(train)); sp = StandardScaler().fit(phy(train))
    sr = StandardScaler().fit(phy(train)[:, :1])
    return dict(
        raw=(sx, mlp(sx.transform(raw(train)), Ytr, seeds)),                 # MLP on raw inputs
        dimless=(sp, mlp(sp.transform(phy(train)), Ytr, seeds)),             # (log rho, kappa), no law
        pg=(sp, mlp(sp.transform(phy(train)), Rtr, seeds)),                  # physics-guided PG-MLP
        pg_nok=(sr, mlp(sr.transform(phy(train)[:, :1]), Rtr, seeds)),       # without kappa_hat
        pg_single=(sp, mlp(sp.transform(phy(train)), Rtr, [list(seeds)[0]])),  # single network
        pg_k12=(sp, mlp(sp.transform(phy(train)), Rtr[:, 0], seeds)))         # K12 only


# 1-D smoothers of the residual on log10(rho): 25 knots, each the mean of five consecutive sorted labels
lr = phy(train)[:, 0]; o = np.argsort(lr)
knots_x = lr[o].reshape(25, 5).mean(1); knots_y = Rtr[o].reshape(25, 5, 2).mean(1)
pchip = [PchipInterpolator(knots_x, knots_y[:, j], extrapolate=True) for j in range(2)]
lut2d = [RegularGridInterpolator((k_grid, np.log10(A_grid)), Ytr[:, j].reshape(5, 25)) for j in range(2)]


def predict(name, cs, M):
    """Predicted corrections x = (x12, x13) of a scheduler for the cases cs = [(A, kappa_hat), ...]."""
    if name == 'Law (11)':
        return lawv(cs)
    if name == 'LUT 2-D':
        X = raw(cs); pts = np.c_[np.clip(X[:, 1], .5, 2), np.clip(X[:, 0], np.log10(.003), np.log10(3))]
        return np.c_[lut2d[0](pts), lut2d[1](pts)]
    if name in ('LUT 1-D residual', 'PCHIP residual'):
        x = np.clip(phy(cs)[:, 0], knots_x[0], knots_x[-1])
        if name == 'PCHIP residual':
            return lawv(cs) + np.c_[pchip[0](x), pchip[1](x)]
        return lawv(cs) + np.c_[np.interp(x, knots_x, knots_y[:, 0]), np.interp(x, knots_x, knots_y[:, 1])]
    key = {'MLP raw': 'raw', 'MLP dimensionless': 'dimless', 'PG-MLP': 'pg', 'PG-MLP no kappa': 'pg_nok',
           'PG-MLP single': 'pg_single', 'PG-MLP K12 only': 'pg_k12'}[name]
    sc, ms = M[key]
    X = phy(cs) if key != 'raw' else raw(cs)
    if key == 'pg_nok': X = X[:, :1]
    y = np.mean([m.predict(sc.transform(X)).reshape(len(cs), -1) for m in ms], axis=0)
    if key == 'pg_k12': y = np.c_[y[:, 0], np.zeros(len(cs))]
    return y if key in ('raw', 'dimless') else lawv(cs) + y


def safeguard(K):
    """Eq. (12): K13 <= K12^2 / (4 zeta_min^2) keeps real negative roots of K13 p^2 + K12 p + 1 = 0."""
    K = K.copy(); K[1] = min(K[1], K[0] ** 2 / (4 * ZETA_MIN ** 2)); return K


def vpeak(A, V, Am, J):
    """Peak speed of the kinematic reference for the displacement A."""
    dist = lambda vp: vp * (vp / Am + Am / J if vp >= Am * Am / J else 2 * np.sqrt(vp / J))
    if A >= dist(V): return V
    lo, hi = 0.0, V
    for _ in range(80):
        m = .5 * (lo + hi); lo, hi = (m, hi) if dist(m) < A else (lo, m)
    return .5 * (lo + hi)


def adaptive_ni(A, kh):
    """Amplitude-adaptive analytic N-i synthesis: E2max in (5), (6) replaced by the reference peak speed."""
    V, Am, J = design_limits(kh)
    return ni_coefficients(vpeak(A, V, Am, J), Am, J)


def evaluate(A, kap, kh, K, dt=1e-5):
    """tau1 = ts / T*_1%, tau = ts / T* (references of the TRUE drive), overshoot, peak torque and speed."""
    r = run(A, K, kappa=kap, kappa_hat=kh, T=4.0, dt=dt)
    V, Am, J = design_limits(kap)
    return dict(tau=r['ts'] / t_ref(A, V, Am, J), tau1=r['ts'] / t_ref_band(A, V, Am, J),
                ovs=r['ovs'], mpk=r['Mpeak'] / MMAX, wpk=r['wpk'] / WMAX)


METHODS = ['Fixed N-i', 'Rescheduled N-i', 'Adaptive N-i', 'Law (11)', 'LUT 2-D', 'MLP raw',
           'MLP dimensionless', 'LUT 1-D residual', 'PCHIP residual', 'PG-MLP no kappa', 'PG-MLP single',
           'PG-MLP K12 only', 'PG-MLP', 'PG-MLP margin']


def controllers(A, kh, M):
    """Coefficients and the inertia ratio assumed by each controller for one move."""
    out = {'Fixed N-i': (ni_coefficients(*design_limits(1.0)), 1.0),
           'Rescheduled N-i': (ni_coefficients(*design_limits(kh)), kh),
           'Adaptive N-i': (adaptive_ni(A, kh), kh)}
    for m in METHODS[3:-1]:
        out[m] = (safeguard(K_of(predict(m, [(A, kh)], M)[0], kh)), kh)
    ks = MARGIN * kh
    out['PG-MLP margin'] = (safeguard(K_of(predict('PG-MLP', [(A, ks)], M)[0], ks)), ks)
    return out


def eval_case(args):
    i, A, kap, kh, M, tag = args
    rows = []
    for name, (K, khh) in controllers(A, kh, M).items():
        rows.append(dict(case=i, A=A, kappa=kap, kappa_hat=khh, rho=rho(A, kap), est=tag, method=name,
                         **evaluate(A, kap, khh, K)))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    M = build()
    rng = np.random.default_rng(SEED_BIG)
    big = [(float(10 ** rng.uniform(np.log10(0.004), np.log10(2.5))), float(rng.uniform(0.55, 1.9))) for _ in range(2000)]
    while len(big) < 2500:   # 500 extra moves with 0.5 <= rho <= 2 (rejection sampling)
        a, k = float(10 ** rng.uniform(np.log10(0.004), np.log10(2.5))), float(rng.uniform(0.55, 1.9))
        if 0.5 <= rho(a, k) <= 2.0: big.append((a, k))
    deltas = rng.uniform(-0.1, 0.1, len(big))
    jobs = [(i, a, k, k, M, 'exact') for i, (a, k) in enumerate(big)]
    jobs += [(i, a, k, k * (1 + deltas[i]), M, 'err10') for i, (a, k) in enumerate(big)]
    jobs += [(i, a, k, k * 0.9, M, 'under10') for i, (a, k) in enumerate(big[:500])]
    jobs += [(i, a, k, k * 1.1, M, 'over10') for i, (a, k) in enumerate(big[:500])]
    jobs40 = [(i, a, k, k, M, 'exact40') for i, (a, k) in enumerate(test40)]
    with Pool(args.workers) as p:
        rows = sum(p.map(eval_case, jobs + jobs40, chunksize=20), [])
    for i, (a, k) in enumerate(test40):     # oracle on the 40 labelled moves
        rows.append(dict(case=i, A=a, kappa=k, kappa_hat=k, rho=rho(a, k), est='exact40', method='Oracle',
                         **evaluate(a, k, k, K_of(Y40[i], k))))
    t1 = time.time()
    # five disjoint seed groups for PG-MLP and the raw-input MLP, evaluated on the first 500 moves
    seed_rows = []
    for g in range(5):
        Mg = build(range(5 * g, 5 * g + 5))
        for i, (a, k) in enumerate(big[:500]):
            for m in ('PG-MLP', 'MLP raw'):
                K = safeguard(K_of(predict(m, [(a, k)], Mg)[0], k))
                seed_rows.append(dict(group=g, case=i, method=m, rho=rho(a, k), **evaluate(a, k, k, K)))
    # accuracy of the predicted corrections on the 40 labelled moves
    rmse, max_rel = {}, {}
    for m in METHODS[3:-1]:
        P = predict(m, test40, M)
        rmse[m] = [float(v) for v in np.sqrt(np.mean((P - Y40) ** 2, axis=0))]
        max_rel[m] = [float(v) for v in (np.exp(np.abs(P - Y40)) - 1).max(axis=0)]
    # integration-step study: the Fig. 5 move and three labelled moves
    conv = []
    for (a, k) in [(0.05, 1.3)] + test40[:3]:
        for name in ('Rescheduled N-i', 'PG-MLP'):
            K, khh = controllers(a, k, M)[name]
            for dt in (2e-5, 1e-5, 5e-6, 2.5e-6, 1e-6):
                conv.append(dict(A=a, kappa=k, method=name, dt=dt, **evaluate(a, k, khh, K, dt=dt)))
    # how often the safeguard (12) changes the PG-MLP coefficients
    active = sum(int(np.any(np.abs(safeguard(K_of(predict('PG-MLP', [c], M)[0], c[1])) -
                                   K_of(predict('PG-MLP', [c], M)[0], c[1])) > 0)) for c in big)
    save_results2(dict(rows=rows, seed_rows=seed_rows, rmse=rmse, max_rel=max_rel, conv=conv,
                       safeguard_active=active, n_big=len(big), t_main=t1 - t0, t_total=time.time() - t0))
    print(f'done in {time.time() - t0:.0f} s, safeguard active in {active} of {len(big)} moves')
