# Physics-guided learning of switching coefficients for near-time-optimal relay control of a DC positioning drive

Code and data that reproduce all numerical results, tables and figures of the article

> Sheremet O. I., Sadovoi O. V., Sokhina Yu. V., Kiriienko T. V. “Physics-guided learning of switching
> coefficients for near-time-optimal relay control of a direct-current positioning drive”.
> *Informatics. Culture. Technology*, 2026 (submitted).

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
| `evaluation.py` | Step 2: all controllers on 2500 test moves, inertia-estimation errors, inertia margin, ablation, seed groups, integration-step study |
| `tables.py` | Step 3: Tables 1–4 of the paper → `tables.md` |
| `checks.py` | Step 4: numbers quoted in the text but not shown in the tables → `checks.txt` |
| `figures.py` | Step 5: Figs. 1–3 of the paper and the supplementary Fig. S1 |
| `run_all.py` | Runs steps 1–5 in order |
| `results_io.py` | Reading/writing of the compressed evaluation results |
| `results.json` | Archived output of step 1: oracle labels, sensitivity data, inference time |
| `models.pkl` | Archived output of step 1: trained scikit-learn ensembles (pickle) |
| `results2.json.gz` | Archived output of step 2: every simulated move of every controller (gzip-compressed JSON) |
| `tables.md`, `checks.txt`, `fig*.png`, `fig3_numbers.json` | Archived outputs of steps 3–5 |
| `requirements.txt` | Exact package versions used for the paper |

## Requirements

* Python 3.13 (the results were produced with Python 3.13.18; Python 3.11–3.13 should work).
* The packages in `requirements.txt` (NumPy, SciPy, Numba, scikit-learn, pandas, Matplotlib).
* About 10 minutes on two CPU cores for the complete pipeline; less than 1 GB of memory.

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

Complete pipeline (simulation from scratch, about 10 minutes):

```bash
python run_all.py --workers 2
```

Tables, checks and figures from the archived simulation results only (about 1 minute):

```bash
python run_all.py --skip-simulation
```

The steps can also be run one by one:

```bash
python ni_drive.py      # check of the analytic synthesis against the published coefficients
python experiment.py    # step 1 -> results.json, models.pkl
python evaluation.py    # step 2 -> results2.json.gz
python tables.py        # step 3 -> tables.md
python checks.py        # step 4 -> checks.txt
python figures.py       # step 5 -> fig1_structure.png, fig2_labels_sensitivity.png, fig3_transients.png, figS1_tau_vs_rho.png
```

`--workers N` sets the number of parallel processes in steps 1 and 2; the results do not depend on it.

### Where each result of the paper comes from

| Paper | Produced by | Output |
|---|---|---|
| Coefficients K12, K13, K23 and the published roots (text after Eq. (6), Table 1) | `ni_drive.py`, `tables.py` | console, `tables.md` |
| Table 2 (closed-loop results, 2500 moves) | `evaluation.py` → `tables.py` | `tables.md` |
| Table 3 (ablation) | `evaluation.py` → `tables.py` | `tables.md` |
| Table 4 (inertia estimation errors, inertia margin) | `evaluation.py` → `tables.py` | `tables.md` |
| Oracle comparison, peak speed, integration-step study, seed groups, coefficient errors, extrapolation in the inertia ratio, safeguard activity, inference time | `evaluation.py` → `checks.py` | `checks.txt` |
| Fig. 1 (structure) | `figures.py` | `fig1_structure.png` |
| Fig. 2 (oracle corrections and sensitivity) | `experiment.py` → `figures.py` | `fig2_labels_sensitivity.png` |
| Fig. 3 (transients for A = 0.05 rad, κ = 1.3) | `figures.py` | `fig3_transients.png`, `fig3_numbers.json` |
| Supplementary Fig. S1 (τ₁ vs ρ for 2500 moves) | `evaluation.py` → `figures.py` | `figS1_tau_vs_rho.png` |

## Method in brief

* **Plant.** 30 W, 27 V DC motor with a 360:1 gear, third-order output-shaft model (1) with the data of
  the source monograph (Sadovoi et al., 2011, Table 7.1). Inertia variation κ = J/J_nom, true κ in
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

## Reproducibility notes

* Random seeds: labelled test moves 2026; 2500-move test set and inertia errors 2027; MLP ensemble 0–4;
  seed groups 0–24. The simulations are deterministic.
* Numba compiles the simulator on the first call and caches it in `__pycache__`.
* `models.pkl` is a scikit-learn pickle; load it only with the package versions of `requirements.txt`.
  It is regenerated by `experiment.py`.
* Small differences in the last digit are possible on other CPUs or package versions because of
  floating-point summation order in the network training.

## License

Code: MIT License (see `LICENSE`). Data files (`results.json`, `results2.json.gz`, `models.pkl`, tables and
figures): Creative Commons Attribution 4.0 International (CC BY 4.0).

## How to cite

Please cite the article above and this archive (see `CITATION.cff`; the DOI is assigned by Zenodo).

## Contact

Oleksii I. Sheremet, Donbas State Engineering Academy, Kramatorsk, Ukraine – oleksii.sheremet@ddma.edu.ua
