### Table 1 (computed rows). Derivative limits (4) and N-i coefficients (5), (6), kappa_hat = 1

| Quantity | Value |
|---|---|
| E2max, E3max, E4max | 1.19 rad/s; 4.586 rad/s^2; 4568 rad/s^3 |
| K12, K13, K23 | 0.1302 s; 6.52e-05 s^2; 0.000502 s |

### Table 2. Closed-loop results on 2500 test moves, exact inertia ratio

| Controller | tau1 median | tau1 99 % | tau1 max | overshoot max, % | Mpk/Mmax max | wpk/wmax max |
|---|---|---|---|---|---|---|
| Fixed N-i [1] | 1.866 | 10.694 | 13.628 | 0.17 | 1.915 | 1.000 |
| Rescheduled N-i | 1.948 | 12.723 | 14.648 | 0.03 | 1.016 | 1.000 |
| Amplitude-adaptive N-i | 1.004 | 1.456 | 1.531 | 4.76 | 1.016 | 1.000 |
| Law (11) | 1.002 | 1.415 | 1.416 | 3.04 | 1.016 | 1.000 |
| LUT (raw inputs) | 0.994 | 1.332 | 1.396 | 3.07 | 1.016 | 1.000 |
| MLP (raw inputs) | 0.993 | 1.368 | 1.400 | 3.12 | 1.016 | 1.000 |
| PG-MLP | 0.987 | 0.996 | 1.006 | 0.93 | 1.016 | 1.000 |
| PG-MLP, margin kappa_hat/0.9 | 1.040 | 1.054 | 1.066 | 0.92 | 0.917 | 1.000 |
| Oracle (40 moves only) | 0.987 | 0.992 | 0.993 | 0.50 | 1.016 | 1.000 |

### Table 3. Ablation of the physics-guided scheduler (2500 test moves)

| Variant | RMSE x12 (40 moves) | tau1 median | tau1 99 % | tau1 max | overshoot max, % |
|---|---|---|---|---|---|
| MLP on (log10 rho, kappa_hat), no law | 0.0107 | 0.992 | 1.362 | 1.398 | 3.09 |
| Law + 1-D LUT of residual | 0.0051 | 0.989 | 1.017 | 1.182 | 1.20 |
| Law + PCHIP of residual | 0.0051 | 0.989 | 1.016 | 1.186 | 1.21 |
| PG-MLP without kappa_hat | 0.0061 | 0.991 | 1.012 | 1.203 | 1.32 |
| PG-MLP, single network | 0.0029 | 0.988 | 1.120 | 1.160 | 1.29 |
| PG-MLP, K12 only | 0.0041 | 0.990 | 1.346 | 1.352 | 1.99 |
| PG-MLP (proposed) | 0.0023 | 0.987 | 0.996 | 1.006 | 0.93 |

### Table 4. Robustness of PG-MLP under inertia estimation errors (without / with the inertia margin)

| Inertia estimate | Moves | tau1 median | tau1 max | overshoot max, % | Mpk/Mmax max |
|---|---|---|---|---|---|
| Exact | 2500 | 0.987 / 1.040 | 1.006 / 1.066 | 0.93 / 0.92 | 1.016 / 0.917 |
| Random, |delta| <= 0.1 | 2500 | 0.988 / 1.030 | 1.055 / 1.117 | 1.01 / 0.86 | 1.127 / 1.016 |
| delta = -0.1 | 500 | 0.935 / 0.988 | 1.165 / 1.001 | 1.58 / 0.92 | 1.127 / 1.016 |
| delta = +0.1 | 500 | 1.035 / 1.092 | 1.060 / 1.124 | 0.72 / 0.76 | 0.926 / 0.835 |

