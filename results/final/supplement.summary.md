# Supplementary experiments (PROTOCOL 9), final seeds 0-29

## 9.1 Preference noise (flip probability p)

Benchmark (36 instances, final error; ratio = geometric mean of median error vs p = 0, <1 is better).

| method | p | ratio vs p=0 | Wilcoxon p vs p=0 | W/T/L vs SW-UCB | one-sided p vs SW-UCB (x10 Bonferroni bound) | W/T/L vs FSS-4 |
|---|---|---|---|---|---|---|
| UPHODS-v2 | 0.0 | 1.000 | — | 12/23/1 | 1.85e-03 (x10: 1.85e-02) | 31/5/0 |
| UPHODS-v2 | 0.1 | 0.992 | 4.14e-01 | 11/24/1 | 2.07e-04 (x10: 2.07e-03) | 30/6/0 |
| UPHODS-v2 | 0.2 | 1.029 | 1.20e-01 | 13/23/0 | 1.17e-03 (x10: 1.17e-02) | 30/6/0 |
| UPHODS-v2 | 0.3 | 0.997 | 7.08e-01 | 13/23/0 | 3.33e-03 (x10: 3.33e-02) | 30/6/0 |
| GP-PO-v2 | 0.0 | 1.000 | — | 3/5/28 | 1.00e+00 (x10: 1.00e+00) | 12/19/5 |
| GP-PO-v2 | 0.1 | 0.777 | 3.79e-01 | 3/7/26 | 1.00e+00 (x10: 1.00e+00) | 12/18/6 |
| GP-PO-v2 | 0.2 | 0.907 | 2.09e-01 | 3/4/29 | 1.00e+00 (x10: 1.00e+00) | 12/18/6 |
| GP-PO-v2 | 0.3 | 0.878 | 2.95e-03 | 2/5/29 | 1.00e+00 (x10: 1.00e+00) | 12/20/4 |

Misaligned scenario (12 instances, utility of the DM's pick; higher is better).

| method | p | mean dm_pick_u | Wilcoxon p vs p=0 | W/T/L vs SW-UCB | one-sided p vs SW-UCB |
|---|---|---|---|---|---|
| UPHODS-v2 | 0.0 | -3.054 | — | 2/10/0 | 4.61e-02 |
| UPHODS-v2 | 0.1 | -3.054 | 3.80e-01 | 2/10/0 | 4.61e-02 |
| UPHODS-v2 | 0.2 | -3.045 | 1.00e+00 | 3/9/0 | 5.49e-02 |
| UPHODS-v2 | 0.3 | -3.038 | 7.91e-01 | 2/10/0 | 2.61e-02 |
| GP-PO-v2 | 0.0 | -6.076 | — | 0/0/12 | 1.00e+00 |
| GP-PO-v2 | 0.1 | -6.278 | 9.77e-04 | 0/0/12 | 1.00e+00 |
| GP-PO-v2 | 0.2 | -6.353 | 4.88e-04 | 0/0/12 | 1.00e+00 |
| GP-PO-v2 | 0.3 | -6.729 | 4.88e-04 | 0/0/12 | 1.00e+00 |

## 9.2 One-at-a-time sensitivity of UPHODS-v2 (18 instances: 6 functions x D in {10, 30, 100})

Default: W/T/L vs SW-UCB = 6/11/1

| parameter | value | ratio vs default | W/T/L vs default | Wilcoxon p | Holm | W/T/L vs SW-UCB | mean NS |
|---|---|---|---|---|---|---|---|
| archive_delta | 0.02 | 1.038 | 0/18/0 | 8.14e-01 | 1.00e+00 | 6/11/1 | 15.6 |
| archive_delta | 0.1 | 1.056 | 0/18/0 | 4.33e-01 | 1.00e+00 | 7/10/1 | 15.5 |
| cooldown_evals | 5 | 1.000 | 0/18/0 | 1.00e+00 | 1.00e+00 | 6/11/1 | 15.4 |
| cooldown_evals | 20 | 0.989 | 0/18/0 | 1.19e-01 | 1.00e+00 | 6/12/0 | 14.4 |
| cooldown_evals | 40 | 1.360 | 0/14/4 | 5.99e-02 | 1.00e+00 | 4/11/3 | 11.5 |
| eps | 0.1 | 0.993 | 0/18/0 | 4.63e-01 | 1.00e+00 | 7/11/0 | 17.2 |
| eps | 0.4 | 1.050 | 0/18/0 | 5.51e-01 | 1.00e+00 | 6/11/1 | 12.7 |
| inject_ratio | 0.2 | 1.026 | 0/18/0 | 2.85e-01 | 1.00e+00 | 6/11/1 | 15.5 |
| inject_ratio | 0.7 | 1.052 | 0/18/0 | 1.61e-01 | 1.00e+00 | 6/11/1 | 15.5 |
| kappa | 0.5 | 1.013 | 0/18/0 | 1.54e-01 | 1.00e+00 | 6/11/1 | 12.8 |
| kappa | 2 | 1.057 | 0/17/1 | 5.51e-01 | 1.00e+00 | 7/10/1 | 19.8 |
| lam_m | 0.25 | 0.989 | 0/18/0 | 7.47e-02 | 1.00e+00 | 6/12/0 | 15.5 |
| lam_m | 1 | 0.996 | 0/18/0 | 7.22e-01 | 1.00e+00 | 6/12/0 | 15.4 |
| lam_p | 0.25 | 0.958 | 0/18/0 | 6.15e-02 | 1.00e+00 | 6/12/0 | 15.6 |
| lam_p | 1 | 0.994 | 0/17/1 | 3.25e-01 | 1.00e+00 | 6/11/1 | 15.2 |
| lengthscale | 0.25 | 0.960 | 0/18/0 | 6.09e-01 | 1.00e+00 | 6/12/0 | 15.5 |
| lengthscale | 1 | 1.010 | 0/18/0 | 5.75e-01 | 1.00e+00 | 6/11/1 | 15.4 |
| phase_half_life | 100 | 1.017 | 0/18/0 | 5.86e-01 | 1.00e+00 | 7/11/0 | 18.0 |
| phase_half_life | 400 | 0.988 | 0/18/0 | 9.06e-01 | 1.00e+00 | 7/10/1 | 14.0 |
| slate_size | 3 | 1.004 | 0/18/0 | 9.29e-01 | 1.00e+00 | 6/11/1 | 15.4 |
| slate_size | 8 | 1.007 | 0/18/0 | 8.75e-01 | 1.00e+00 | 6/12/0 | 15.5 |
| weight_window | 15 | 1.000 | 0/18/0 | 7.15e-01 | 1.00e+00 | 6/12/0 | 15.4 |
| weight_window | 60 | 1.000 | 0/18/0 | 1.00e+00 | 1.00e+00 | 6/11/1 | 15.5 |

(default mean NS on these instances: 15.4; W/T/L vs default counts the variant's wins)

## 9.4 Budget 5000 (12 instances: 6 functions x D in {30, 100})

Mean ranks: UPHODS-v2 1.92, SW-UCB 3.08, FSS-3 4.75, FSS-1 4.92, Single-PSO 5.08, FSS-2 6.25, Single-GD 6.50, FSS-4 7.25, Single-SA 7.42, Single-GA 7.83

Friedman p = 1.5e-06

| vs | W/T/L | Wilcoxon p (one-sided) | Holm |
|---|---|---|---|
| FSS-1 | 10/2/0 | 2.44e-04 | 2.20e-03 |
| FSS-2 | 11/1/0 | 2.44e-04 | 2.20e-03 |
| FSS-4 | 11/1/0 | 2.44e-04 | 2.20e-03 |
| Single-SA | 12/0/0 | 2.44e-04 | 2.20e-03 |
| Single-GA | 10/1/1 | 2.44e-03 | 1.22e-02 |
| FSS-3 | 10/2/0 | 4.64e-03 | 1.86e-02 |
| Single-GD | 9/3/0 | 1.05e-02 | 3.15e-02 |
| Single-PSO | 7/1/4 | 6.47e-02 | 1.29e-01 |
| SW-UCB | 8/2/2 | 8.81e-02 | 1.29e-01 |

UPHODS-v2 vs SW-UCB by dimension (B = 5000):
- 30D: W/T/L 4/0/2
- 100D: W/T/L 4/2/0

GP cost at B = 5000 (UPHODS-v2, mean per run):
      n_pairs  n_train  gp_seconds  gp_max_seconds  wall_seconds      ns
dim                                                                     
30   1010.611  443.117      57.389           0.834       102.950  54.989
100   450.833  196.900       3.949           0.101        16.599  39.011

## 9.3 GP fit time vs number of pairs (median of 5, single-threaded BLAS)

Hardware: {'machine': 'AMD64', 'processor': 'AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD', 'system': 'Windows-11-10.0.26200-SP0', 'python': '3.14.3', 'numpy': '2.5.4', 'logical_cpus': 32, 'blas_threads': '1'}

| D | pairs | designs | cold fit (s) | warm-started fit (s) |
|---|---|---|---|---|
| 10 | 100 | 43 | 0.004 | 0.003 |
| 10 | 250 | 108 | 0.013 | 0.011 |
| 10 | 500 | 215 | 0.067 | 0.057 |
| 10 | 1000 | 430 | 0.442 | 0.394 |
| 10 | 2000 | 860 | 2.495 | 2.119 |
| 10 | 4000 | 1720 | 16.187 | 14.175 |
| 100 | 100 | 43 | 0.005 | 0.004 |
| 100 | 250 | 108 | 0.015 | 0.015 |
| 100 | 500 | 215 | 0.070 | 0.066 |
| 100 | 1000 | 430 | 0.394 | 0.371 |
| 100 | 2000 | 860 | 2.297 | 2.209 |
| 100 | 4000 | 1720 | 13.443 | 12.540 |
