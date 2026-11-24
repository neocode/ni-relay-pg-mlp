"""Step 1 - oracle labels, scheduling models and sensitivity study.

* Oracle labels x* = (x12, x13) of Eq. (10) for 125 training cases (25 displacements x 5 inertia ratios)
  and 40 labelled test moves (seed 2026), Nelder-Mead from four starting points.
* Residual PG-MLP ensemble (Eq. (12)) and a raw-input MLP ensemble, saved to models.pkl.
* Sensitivity of the settling time to a relative error of K12 around the oracle optimum
  (ten labelled moves, Fig. 2, b).
* Inference time of one PG-MLP prediction.

Output: results.json, models.pkl.
Run:    python experiment.py [--workers N]      (about 1-2 min on two CPU cores)
"""
import argparse, json, time, pickle
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from ni_drive import run, ni_coefficients, design_limits, t_ref, MMAX

OVS_MAX = 5e-3       # overshoot tolerated without penalty in Eq. (10), fraction of the displacement
EVAL_T = 4.0         # simulation horizon T_sim, s
SEED_TEST = 2026     # labelled test moves


def K_of(x, kappa_hat):
    """Scheduled coefficients, Eq. (9): K12 = K12^0 exp(x12), K13 = K13^0 exp(x13), K23 = K23^0."""
    K = ni_coefficients(*design_limits(kappa_hat)).copy()
    K[0] *= np.exp(x[0]); K[1] *= np.exp(x[1])
    return K


def rho(A, kappa_hat):
    """Dimensionless displacement, Eq. (7): A / A_sat, A_sat = E2max^2/E3max + E2max E3max/E4max."""
    V, Am, J = design_limits(kappa_hat)
    return A / (V * V / Am + V * Am / J)


def law(A, kappa_hat):
    """Square-root scaling law, Eq. (11): x12 = 0.5 ln min(rho, 1), x13 = 0."""
    return np.array([0.5 * np.log(min(rho(A, kappa_hat), 1.0)), 0.0])


def cost(x, A, kap):
    """Penalised labelling cost, Eq. (10)."""
    r = run(A, K_of(x, kap), kappa=kap, kappa_hat=kap)
    if not np.isfinite(r['ts']):
        return 20.0
    return (r['ts'] / r['Tref'] + 50 * max(0.0, r['ovs'] - OVS_MAX)
            + 5 * max(0.0, r['Mpeak'] / MMAX - 1.05) + 1e-3 * float(np.dot(x, x)))


def oracle(case):
    """Best of four Nelder-Mead searches (the 'oracle'; not a guaranteed global optimum)."""
    A, kap = case
    best = None
    for x0 in ([0, 0], law(A, kap), [-1, 0], [-2, -1]):
        res = minimize(cost, np.array(x0, float), args=(A, kap), method='Nelder-Mead',
                       options=dict(xatol=1e-3, fatol=1e-4, maxfev=200))
        if best is None or res.fun < best.fun:
            best = res
    return [float(v) for v in best.x], float(best.fun)


def evaluate(A, kap, kap_hat, K):
    """Normalised settling time tau = ts / T* (T* of the TRUE drive), overshoot and peak torque."""
    r = run(A, K, kappa=kap, kappa_hat=kap_hat, T=EVAL_T)
    V, Am, J = design_limits(kap)
    return dict(tau=r['ts'] / t_ref(A, V, Am, J), ovs=r['ovs'], mpk=r['Mpeak'] / MMAX)


def mlp_ensemble(X, Y):
    """Five perceptrons, two hidden layers of 16 tanh units, L-BFGS, seeds 0-4."""
    return [MLPRegressor(hidden_layer_sizes=(16, 16), activation='tanh', solver='lbfgs',
                         alpha=1e-3, max_iter=5000, random_state=s).fit(X, Y) for s in range(5)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    A_grid = np.logspace(np.log10(0.003), np.log10(3.0), 25)
    k_grid = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
    train = [(float(a), float(k)) for k in k_grid for a in A_grid]
    rng = np.random.default_rng(SEED_TEST)
    test = [(float(10 ** rng.uniform(np.log10(0.004), np.log10(2.5))), float(rng.uniform(0.55, 1.9)))
            for _ in range(40)]
    with Pool(args.workers) as p:
        lab_tr = p.map(oracle, train)
        lab_te = p.map(oracle, test)
    t_oracle = time.time() - t0
    Ytr = np.array([l[0] for l in lab_tr]); Yte = np.array([l[0] for l in lab_te])
    raw = lambda cases: np.array([[np.log10(a), k] for a, k in cases])
    phy = lambda cases: np.array([[np.log10(rho(a, k)), k] for a, k in cases])
    lawv = lambda cases: np.array([law(a, k) for a, k in cases])

    sx = StandardScaler().fit(raw(train)); sp = StandardScaler().fit(phy(train))
    m_raw = mlp_ensemble(sx.transform(raw(train)), Ytr)
    m_pg = mlp_ensemble(sp.transform(phy(train)), Ytr - lawv(train))      # residual over the law (11)

    def pg_predict(cases):
        return lawv(cases) + np.mean([m.predict(sp.transform(phy(cases))) for m in m_pg], axis=0)

    t1 = time.perf_counter()
    for _ in range(200):
        pg_predict(test[:1])
    t_inf = (time.perf_counter() - t1) / 200

    # sensitivity of tau to a relative error of K12 around the oracle optimum (every fourth labelled move)
    sens = []
    for i in range(0, 40, 4):
        A, kap = test[i]
        for d in (-0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08):
            e = evaluate(A, kap, kap, K_of(Yte[i] + np.array([d, 0.0]), kap))
            sens.append(dict(case=i, rho=rho(A, kap), d=d, **e))
    n_par = int(sum(c.size for c in m_pg[0].coefs_) + sum(c.size for c in m_pg[0].intercepts_))
    json.dump(dict(train=train, lab_tr=lab_tr, test=test, lab_te=lab_te, sens=sens,
                   t_oracle=t_oracle, t_inf=t_inf, n_par=n_par),
              open('results.json', 'w'), indent=1)
    pickle.dump(dict(m_raw=m_raw, m_pg=m_pg, sx=sx, sp=sp, A_grid=A_grid, k_grid=k_grid, Ytr=Ytr),
                open('models.pkl', 'wb'))
    print(f"labels {t_oracle:.0f} s, total {time.time() - t0:.0f} s, "
          f"PG-MLP parameters per network {n_par}, one prediction {t_inf * 1e3:.2f} ms")
