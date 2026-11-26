# Physics-guided data-driven scheduler of switching coefficients for near-time-optimal relay control of a positioning electric drive

Code and data that reproduce all numerical results, tables and figures of the article

> Sheremet O. I., Sadovoi O. V., Sokhina Yu. V., Kiriienko T. V. “Physics-guided data-driven scheduler of switching coefficients for
> near-time-optimal relay control of a positioning electric drive”.
> Submitted, 2026.

The repository contains a simulation model of a DC positioning drive with a three-stage relay cascade
synthesised by the N-i switching method (method of successive switchings), the computation of reference
("oracle") switching coefficients by simulation-based optimisation, a physics-guided scheduler of the
switching coefficients (a square-root scaling law plus a small residual multilayer-perceptron ensemble,
PG-MLP), all baselines and ablation variants, and the evaluation on 2500 test moves.

## Contents

| File | Purpose |
|---|---|
| `ni_drive.py` | Drive model (1), relay cascade (3), design limits (4), analytic N-i coefficients (5)–(6), kinematic references T* and T*₁% |
| `experiment.py` | Step 1: oracle labels (125 training + 40 test moves), PG-MLP and MLP ensembles, sensitivity study for Fig. 2 b |
| `evaluation.py` | Step 2: all controllers on 2500 test moves, inertia-estimation errors, inertia margin, ablation, seed groups (500 moves), integration-step study |
| `extensions.py` | Step 2a: additional studies — minimum 1 %-settling time T_opt of the model by linear programming, 2-D interpolant and Gaussian process of the residual, law with x13 = ln min(ρ, 1), non-idealities (load torque, controller sampling, supply error), seed groups on all 2500 moves, cross-validation, displacements outside the training range, repeatability of the labels, damping ratio, label designs (regular grids, three random samples, three Latin hypercubes, three Sobol samples) with three regressors of the residual and their fill distances, interpolant and Gaussian process under inertia errors, with the inertia margin and below the training range, integration step 10/5/1 µs and the whole pipeline at 1 µs, discretisation check of T_opt (600 intervals), labelling-cost maps → `results3.json` |
| `extensions_report.py` | Step 2b: summary of step 2a → `extensions.txt` |
| `tables.py` | Step 3: Tables 1, 2, 4 and 6 of the paper → `tables.md` |
| `checks.py` | Step 4: numbers quoted in the text but not shown in the tables → `checks.txt` |
| `figures.py` | Step 5: Figs. 1–5 of the paper |
| `deployment.py` | Step 6: export of the PG-MLP scheduler to JSON and to a self-contained C header, accuracy and timing of the NumPy and C implementations → `deployment.txt` |
| `profiling.py` | Step 7: computational cost (Numba speed-up, simulations and time per oracle label) → `profiling.txt` |
| `run_all.py` | Runs all steps in order |
| `results_io.py` | Reading/writing of the compressed evaluation results |
| `results.json` | Archived output of step 1: oracle labels, sensitivity data, inference time |
| `models.pkl` | Archived output of step 1: trained scikit-learn ensembles (pickle) |
| `results2.json.gz` | Archived output of step 2: every simulated move of every controller (gzip-compressed JSON) |
| `results3.json`, `extensions.txt` | Archived outputs of step 2a and its summary |
| `tables.md`, `checks.txt`, `fig*.png`, `fig5_numbers.json` | Archived outputs of steps 3–5 |
| `pgmlp_scheduler.h`, `pgmlp_weights.json`, `deployment.txt`, `profiling.txt` | Archived outputs of steps 6–7 |
| `requirements.txt` | Exact package versions used for the paper |

## Requirements

* Python 3.13 (the results were produced with Python 3.13.18; Python 3.11–3.13 should work).
* The packages in `requirements.txt` (NumPy, SciPy, Numba, scikit-learn, pandas, Matplotlib).
* About 2.5–3 hours on two CPU cores for the complete pipeline, most of it for the additional studies of
  `extensions.py` (the linear programs of T_opt); about 10 minutes without them; less than 2 GB of memory.

No system-wide installation is needed: everything is installed into a virtual environment inside the
repository folder.

## Installation

Linux / macOS:

```bash
cd ni-relay-pg-mlp            # the repository folder
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows (PowerShell or Command Prompt):

```bat
cd ni-relay-pg-mlp
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Below, `python` means `.venv/bin/python` (Linux / macOS) or `.venv\Scripts\python` (Windows).

## Reproducing the paper

Complete pipeline (simulation from scratch, about 2.5–3 hours):

```bash
python run_all.py --workers 2
```

Tables, checks, summaries and figures from the archived simulation results only (about 2 minutes):

```bash
python run_all.py --skip-simulation
```

The steps can also be run one by one:

```bash
python ni_drive.py      # check of the analytic synthesis against the published coefficients
python experiment.py    # step 1 -> results.json, models.pkl
python evaluation.py    # step 2 -> results2.json.gz
python extensions.py    # step 2a -> results3.json (about 2.5 hours; finished T_opt cases are cached in topt_cache.jsonl)
python extensions_report.py   # step 2b -> extensions.txt
python tables.py        # step 3 -> tables.md
python checks.py        # step 4 -> checks.txt
python figures.py       # step 5 -> fig1_structure.png ... fig5_transients.png
python deployment.py    # step 6 -> pgmlp_scheduler.h, pgmlp_weights.json, deployment.txt
python profiling.py     # step 7 -> profiling.txt
```

`deployment.py` compiles a small C test program if `gcc`, `cc` or `clang` is available on the PATH; otherwise the
C check is skipped and everything else still runs.

`--workers N` sets the number of parallel processes in steps 1, 2 and 2a; the results do not depend on it
(re-running step 2 on another computer reproduced all 84 600 records of `results2.json.gz` exactly).

### Where each result of the paper comes from

| Paper | Produced by | Output |
|---|---|---|
| Coefficients K12, K13, K23 and the published roots (text after Eq. (6), Table 1) | `ni_drive.py`, `tables.py` | console, `tables.md` |
| Software implementation section: computational cost, footprint, inference time, C verification | `profiling.py`, `deployment.py` | `profiling.txt`, `deployment.txt` |
| Table 2 (closed-loop results, 2500 moves) | `evaluation.py` → `tables.py` | `tables.md` |
| Table 3 (settling time relative to the minimum settling time T_opt: 40 labelled, 30 worst and 30 random moves) | `extensions.py` → `extensions_report.py` | `extensions.txt`, `results3.json` |
| Table 4 (ablation, including the x13 law) | `evaluation.py`, `extensions.py` | `tables.md`, `extensions.txt` |
| Table 5 (PG-MLP, Gaussian process and linear interpolant of the residual on eight label designs) | `extensions.py` (`designs`) → `extensions_report.py` | `extensions.txt`, `results3.json` |
| Table 6 (inertia estimation errors, inertia margin) | `evaluation.py` → `tables.py`; interpolant and Gaussian process: `extensions.py` | `tables.md`, `extensions.txt` |
| Table 7 (load torque from the static state M(0) = Mc, controller sampling, supply error) | `extensions.py` → `extensions_report.py` | `extensions.txt` |
| Peak speed, integration-step study, coefficient errors, extrapolation in the inertia ratio, sensitivity | `evaluation.py` → `checks.py` | `checks.txt` |
| T_opt of the Fig. 5 move, overshoot statistics, seed groups on 2500 moves, cross-validation, out-of-range displacements, repeatability of the labels, damping ratio | `extensions.py` → `extensions_report.py` | `extensions.txt` |
| Fig. 1 (relay cascade with the scheduler) | `figures.py` | `fig1_structure.png` |
| Fig. 2 (oracle corrections and sensitivity) | `experiment.py` → `figures.py` | `fig2_labels_sensitivity.png` |
| Fig. 3 (software pipeline) | `figures.py` | `fig3_pipeline.png` |
| Fig. 4 (median and maximum τ₁ in bins of ρ, 2500 moves) | `evaluation.py`, `extensions.py` (`base2d`) → `figures.py` | `fig4_tau_vs_rho.png` |
| Fig. 5 (transients for A = 0.05 rad, κ = 1.3, with T* and T_opt) | `figures.py` | `fig5_transients.png`, `fig5_numbers.json` |

## Using the scheduler in a controller

`pgmlp_scheduler.h` is generated from the trained ensemble and has no dependencies other than `<math.h>`:

```c
#include "pgmlp_scheduler.h"
float K[3];
int status = pgmlp_coefficients(0.05f, 1.3f, K);  /* A = 0.05 rad, inertia estimate 1.3 -> K12, K13, K23 */
if (status & PGMLP_A_BELOW) { /* below the training range: accuracy degraded, extend the training range */ }
```

The function evaluates Eqs. (4)–(12) of the paper (design limits, analytic N-i coefficients, scaling law,
residual ensemble and safeguard) once per positioning command in single precision; the relay cascade itself
is not part of the header. Contract (status bit flags):

* `PGMLP_OK` (0) — inputs inside the training domain, |A| in [0.003, 3] rad and κ̂ in [0.5, 2];
* `PGMLP_A_BELOW`, `PGMLP_A_ABOVE`, `PGMLP_K_OUTSIDE` — outside this domain but inside the tested range
  (|A| in [3e-4, 10] rad, κ̂ in [0.4, 2.5]); the prediction is extrapolated and returned. In the tests, for
  3–10 rad τ₁ ≤ 1.01 with an overshoot of about 0.5 % of A (as inside the domain for ρ ≥ 1), for κ̂ up to 2.32
  (which occurs with the inertia margin κ̂/0.9) τ₁ ≤ 1.12, and below 0.003 rad the accuracy degraded (τ₁ up to
  1.69, still far better than the analytic coefficients, τ₁ up to 51), see `extensions.txt`;
* `PGMLP_FALLBACK` — input outside the tested range or invalid (non-finite, κ̂ outside [0.4, 2.5], |A| outside
  [3e-4, 10] rad) or a non-finite intermediate result; the analytic coefficients are returned (for κ̂ = 1 if κ̂
  is invalid). Widen the hard limits only after testing.

The sign of A is ignored. For every input the returned coefficients are finite, positive and satisfy the
safeguard K13 ≤ K12²/4; `deployment.py` checks this on 10 514 calls, including 5000 random inputs over the whole
float range, and checks the agreement of x and K with the scikit-learn reference and the status flags on
in-domain, out-of-domain and special inputs. The functions are `static inline` (C99), so unused ones cause no
warnings. The header defines its arrays as `static const`, so every translation unit that includes it gets
its own copy of the 7 kB of weights; include it in one source file only. The same weights are available in
`pgmlp_weights.json` for other languages.

For this two-input problem a two-dimensional interpolant of the residual on the regular training grid was as
accurate as the network ensemble inside the training domain (Table 5 of the paper); it is implemented in
`extensions.py` (`interp2d_residual`) but not exported to C. The ensemble is the reference implementation; below
the training range it extrapolated better than the interpolant and the Gaussian process (`extensions.txt`).

## Method in brief

* **Plant.** 30 W, 27 V DC motor with a 360:1 gear, third-order output-shaft model (1) with the data of
  Table 1 of the paper. Inertia variation κ = J/J_nom, true κ in
  [0.55, 1.9] for the test moves. Explicit Euler integration with a 10 µs step, zero load torque, ideal
  derivative measurements.
* **Controller.** Relay cascade (3) with linear switching functions; relay amplitudes are the design limits
  (4) computed from the inertia estimate κ̂; nominal coefficients from the characteristic points of the
  jerk-limited design trajectory (5)–(6).
* **Performance measure.** Settling time t_s into the 1 % band (8), normalised by the time T*₁% at which the
  jerk-limited kinematic S-curve of the true drive enters the same band: τ₁ = t_s / T*₁%.
* **Oracle labels.** Corrections x = (ln(K12/K12⁰), ln(K13/K13⁰)) minimising the penalised cost (10) by
  Nelder–Mead from four starting points (best result of this procedure, not a guaranteed global optimum).
* **Scheduler.** x̂ = law(ρ̂) + f(log₁₀ ρ̂, κ̂), where ρ is the dimensionless displacement (7), the law is
  x12 = ½ ln min(ρ, 1), x13 = 0 (11), and f is the mean of five MLPs (2 × 16 tanh units, L-BFGS, ℓ2
  penalty 10⁻³, seeds 0–4) trained on the residual; a safeguard K13 ≤ K12²/4 keeps real negative roots of
  the outer sliding equation (12).
* **Robustness.** κ̂ = κ(1 + δ) with δ ~ U[−0.1, 0.1] and δ = ±0.1; conservative margin
  κ_safe = κ̂ / (1 − δ_max), δ_max = 0.1.
* **Additional studies (`extensions.py`).** Minimum 1 %-settling time T_opt of model (1) under |U| ≤ U_max,
  |ω| ≤ ω_max, |M| ≤ M_max: bisection on t_s with a linear-programming feasibility problem for the exactly
  discretised model (300 intervals up to t_s, a tail ending at rest inside the band). Further regressors of
  the same residual, non-idealities, seed groups on all moves, cross-validation, out-of-range displacements
  and repeatability of the labels.

## Data files

`results.json` (step 1):

* `train`, `test` – lists of moves `[A (rad), κ]` (125 training, 40 labelled test moves);
* `lab_tr`, `lab_te` – oracle labels `[[x12, x13], cost]` for these moves;
* `sens` – sensitivity study: `case`, `rho`, `d` (added error of x12), `tau`, `ovs`, `mpk`;
* `t_oracle` (s), `t_inf` (s per PG-MLP prediction), `n_par` (parameters per network).

`results2.json.gz` (step 2), keys:

* `rows` – one record per move and controller: `case`, `A` (rad), `kappa` (true), `kappa_hat` (assumed by
  the controller, including the margin), `rho`, `est` (`exact`, `err10` random error, `under10` δ = −0.1,
  `over10` δ = +0.1, `exact40` labelled moves), `method`, `tau` = t_s/T*, `tau1` = t_s/T*₁%, `ovs`
  (overshoot, fraction of A), `mpk` (peak torque / M_max), `wpk` (peak speed / ω_max);
* `seed_rows` – the same measures for five seed groups (`group` 0–4);
* `rmse`, `max_rel` – RMSE of (x12, x13) and maximum relative errors of (K12, K13) on the 40 labelled moves;
* `conv` – integration-step study (`dt` in s);
* `safeguard_active`, `n_big`, `t_main`, `t_total` – bookkeeping.

To read it in Python: `from results_io import load_results2; R2 = load_results2()`.

`results3.json` (step 2a), keys `topt` (rows with `topt`, `tref1`, `tref` and the settling times of the
controllers, plus a discretisation check), `topt_sens` (torque limit 1.016 M_max), `base2d`, `robust`
(`scen` = condition), `seeds`, `cv`, `ood`, `labels`, `zeta`, `irregular` (first irregular-label study),
`robust2` (interpolant and Gaussian process under inertia errors), `designs` (twelve label designs × three
regressors, with the new labels and the cases of every design), `robust3` (interpolant and Gaussian process with
the inertia margin and below 0.003 rad), `numerics` (integration step 10/5/1 µs; PG-MLP at 1 µs on all test moves
and on the moves with T_opt), `step1us` (labels, training and evaluation at 1 µs), `topt_conv` (T_opt with 600
intervals) and `costmap` (labelling cost around the oracle label, Fig. 2, c); `extensions.txt` summarises them.

## Reproducibility notes

* Random seeds: labelled test moves 2026; 2500-move test set and inertia errors 2027; MLP ensemble 0–4;
  seed groups 0–24; displacements outside the training range 2028; 30 random moves for T_opt 2029; random label
  designs 2030–2032, Latin hypercubes 2034, 2036, 2037, Sobol sequences 2035, 2038, 2039; restarts of the labels 100 + j; tests of
  `deployment.py` 7. The simulations are deterministic.
* Numba compiles the simulator on the first call and caches it in `__pycache__`.
* `models.pkl` is a scikit-learn pickle; load it only with the package versions of `requirements.txt`.
  It is regenerated by `experiment.py`.
* Re-running `evaluation.py` on a different computer reproduced all records of `results2.json.gz` exactly;
  small differences in the last digit are nevertheless possible on other CPUs or package versions because of
  floating-point summation order in the network training. Timings in `profiling.txt` and `deployment.txt`
  depend on the computer.
