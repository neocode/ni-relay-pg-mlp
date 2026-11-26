"""Step 7 - computational cost of the pipeline (software implementation section of the paper).

* Speed-up of the Numba-compiled simulator over the same code interpreted by Python.
* Cost of one oracle label: number of simulations and wall time per label (all 165 labels).
* Training time of the PG-MLP ensemble and throughput of the evaluation (from results2.json.gz).

Output: profiling.txt.   Run:  python profiling.py [--workers N]      (about 1 min)
"""
import argparse, json, time
import numpy as np
from multiprocessing import Pool
from ni_drive import simulate, design_limits, ni_coefficients, t_ref
import experiment
from results_io import load_results2

lines = []
say = lambda *a: (lines.append(' '.join(str(x) for x in a)), print(*a))


ORIG_COST = experiment.cost


def counted_oracle(case):
    n = [0]
    cost0 = ORIG_COST

    def cost(x, A, kap):
        n[0] += 1
        return cost0(x, A, kap)
    experiment.cost = cost
    t0 = time.perf_counter()
    experiment.oracle(case)
    return n[0], time.perf_counter() - t0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    # simulator: one move of 0.05 rad (Fig. 5), horizon 0.5 s, 10 us step = 50 000 steps
    V, A, J = design_limits(1.0); K = ni_coefficients(V, A, J)
    argv = (0.05, K[0], K[1], K[2], V, A, J, 1.0, 0.0, 0.5, 1e-5, 100, 0.01)
    simulate(*argv)                                   # compile
    t0 = time.perf_counter(); n = 50
    for _ in range(n):
        simulate(*argv)
    t_jit = (time.perf_counter() - t0) / n
    t0 = time.perf_counter(); simulate.py_func(*argv); t_py = time.perf_counter() - t0
    say(f'Simulation of 50 000 Euler steps: Numba {1e3 * t_jit:.2f} ms, interpreted Python {1e3 * t_py:.0f} ms, '
        f'speed-up {t_py / t_jit:.0f}x')
    R = json.load(open('results.json'))
    cases = [tuple(c) for c in R['train']] + [tuple(c) for c in R['test']]
    with Pool(args.workers) as p:
        res = p.map(counted_oracle, cases)
    nf = np.array([r[0] for r in res]); tt = np.array([r[1] for r in res])
    say(f'Oracle labels ({len(cases)}): simulations per label median {np.median(nf):.0f} '
        f'(range {nf.min()}-{nf.max()}), total {nf.sum()}; time per label median {np.median(tt):.2f} s '
        f'(one core), total {tt.sum():.0f} core-seconds')
    # training of the PG-MLP ensemble (five networks, L-BFGS) on the 125 labels
    Ytr = np.array([l[0] for l in R['lab_tr']]); tr = [tuple(c) for c in R['train']]
    X = np.array([[np.log10(experiment.rho(a, k)), k] for a, k in tr])
    Res = Ytr - np.array([experiment.law(a, k) for a, k in tr])
    from sklearn.preprocessing import StandardScaler
    t0 = time.perf_counter(); experiment.mlp_ensemble(StandardScaler().fit_transform(X), Res)
    say(f'Training of the PG-MLP ensemble (5 networks, 125 labels): {time.perf_counter() - t0:.1f} s')
    R2 = load_results2()
    say(f"Evaluation (evaluation.py): {len(R2['rows'])} closed-loop simulations of 4 s in {R2['t_main']:.0f} s on "
        f"{args.workers} cores ({1e3 * R2['t_main'] * args.workers / len(R2['rows']):.1f} ms per simulation and core); "
        f"total run {R2['t_total']:.0f} s")
    open('profiling.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
