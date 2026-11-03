
## penalty level 1000  (30 seeds, 12 instances)

Share of runs whose best design is feasible: Single-GD 42%, FSS-2 58%, Single-GA 59%, FSS-4 61%, FSS-1 63%, Single-SA 64%, FSS-3 67%, UPHODS-v2-noHys 75%, UPHODS-v2-fixedW 76%, SW-UCB 76%, UPHODS-v2 76%, GP-PO-v2 76%, UPHODS-v2-noRN 76%, Single-PSO 77%, UPHODS-v2-noET 78%
All methods together: 68%

UPHODS-v2 vs ablations (two-sided; W/T/L from UPHODS-v2's side)

| variant | W/T/L | UPHODS-v2 better on | Wilcoxon p | Holm | mean NS (variant / v2) |
|---|---|---|---|---|---|
| UPHODS-v2-fixedW | 0/12/0 | 17% | 5.47e-02 | 2.19e-01 | 20.8 / 20.7 |
| UPHODS-v2-noET | 7/4/1 | 67% | 3.39e-01 | 6.79e-01 | 15.4 / 20.7 |
| UPHODS-v2-noHys | 0/12/0 | 58% | 9.70e-01 | 9.70e-01 | 26.9 / 20.7 |
| UPHODS-v2-noRN | 0/12/0 | 33% | 1.51e-01 | 4.54e-01 | 20.6 / 20.7 |

Mean ranks (non-ablation methods): UPHODS-v2 1.50, SW-UCB 2.75, Single-PSO 5.00, FSS-3 5.08, GP-PO-v2 5.42, Single-GD 5.92, FSS-1 6.42, FSS-4 6.92, FSS-2 8.50, Single-SA 8.75, Single-GA 9.75
Friedman p = 1.3e-10

| vs | W/T/L | Wilcoxon p (one-sided) | Holm |
|---|---|---|---|
| FSS-1 | 10/2/0 | 2.44e-04 | 2.44e-03 |
| FSS-2 | 12/0/0 | 2.44e-04 | 2.44e-03 |
| FSS-3 | 9/3/0 | 2.44e-04 | 2.44e-03 |
| FSS-4 | 11/1/0 | 2.44e-04 | 2.44e-03 |
| GP-PO-v2 | 11/1/0 | 2.44e-04 | 2.44e-03 |
| SW-UCB | 3/9/0 | 2.44e-04 | 2.44e-03 |
| Single-GA | 12/0/0 | 2.44e-04 | 2.44e-03 |
| Single-SA | 11/1/0 | 2.44e-04 | 2.44e-03 |
| Single-PSO | 8/3/1 | 8.06e-03 | 1.61e-02 |
| Single-GD | 6/6/0 | 2.61e-02 | 2.61e-02 |

Per-instance medians (UPHODS-v2, -noRN, -fixedW, SW-UCB):

method                UPHODS-v2  UPHODS-v2-noRN  UPHODS-v2-fixedW  SW-UCB
function         dim                                                     
Ackley           10         3.2           2.726             2.925   3.242
                 30       11.73           10.81             11.73   16.31
Griewank         10     0.03695         0.03326           0.03695 0.04183
                 30    0.007648        0.006406          0.007648 0.01909
Rastrigin        10       16.91           15.42             16.42   19.81
                 30       87.56           87.56             87.56   108.6
Rosenbrock       10        1044            1045              1044    1049
                 30       49.09           49.09             49.09   94.62
ShiftedRastrigin 10        40.3           39.38              40.3   41.29
                 30       173.6           173.6             173.6   187.6
ShiftedSphere    10        1051            1051              1051    1051
                 30        1053            1053              1053    1053

## penalty level 1000000  (30 seeds, 12 instances)

Share of runs whose best design is feasible: Single-GA 93%, FSS-2 96%, FSS-3 100%, FSS-1 100%, FSS-4 100%, GP-PO-v2 100%, SW-UCB 100%, Single-GD 100%, Single-PSO 100%, Single-SA 100%, UPHODS-v2 100%, UPHODS-v2-fixedW 100%, UPHODS-v2-noET 100%, UPHODS-v2-noHys 100%, UPHODS-v2-noRN 100%
All methods together: 99%

UPHODS-v2 vs ablations (two-sided; W/T/L from UPHODS-v2's side)

| variant | W/T/L | UPHODS-v2 better on | Wilcoxon p | Holm | mean NS (variant / v2) |
|---|---|---|---|---|---|
| UPHODS-v2-fixedW | 0/12/0 | 8% | 2.19e-01 | 8.75e-01 | 19.4 / 19.4 |
| UPHODS-v2-noET | 5/7/0 | 50% | 8.50e-01 | 1.00e+00 | 14.7 / 19.4 |
| UPHODS-v2-noHys | 0/12/0 | 42% | 8.50e-01 | 1.00e+00 | 24.8 / 19.4 |
| UPHODS-v2-noRN | 0/12/0 | 67% | 3.39e-01 | 1.00e+00 | 19.0 / 19.4 |

Mean ranks (non-ablation methods): SW-UCB 2.58, UPHODS-v2 2.67, Single-GD 3.08, FSS-3 5.25, Single-PSO 5.25, GP-PO-v2 5.50, FSS-1 5.75, FSS-4 6.83, FSS-2 8.50, Single-GA 9.92, Single-SA 10.67
Friedman p = 9.1e-14

| vs | W/T/L | Wilcoxon p (one-sided) | Holm |
|---|---|---|---|
| FSS-2 | 11/1/0 | 2.44e-04 | 2.44e-03 |
| Single-GA | 11/1/0 | 2.44e-04 | 2.44e-03 |
| Single-SA | 12/0/0 | 2.44e-04 | 2.44e-03 |
| Single-PSO | 6/4/2 | 1.05e-02 | 7.35e-02 |
| FSS-1 | 9/2/1 | 1.71e-02 | 1.03e-01 |
| FSS-4 | 10/1/1 | 1.71e-02 | 1.03e-01 |
| GP-PO-v2 | 7/5/0 | 1.71e-02 | 1.03e-01 |
| FSS-3 | 8/3/1 | 2.12e-02 | 1.03e-01 |
| SW-UCB | 2/9/1 | 3.96e-01 | 7.91e-01 |
| Single-GD | 3/8/1 | 5.75e-01 | 7.91e-01 |

Per-instance medians (UPHODS-v2, -noRN, -fixedW, SW-UCB):

method                UPHODS-v2  UPHODS-v2-noRN  UPHODS-v2-fixedW  SW-UCB
function         dim                                                     
Ackley           10        3.03           3.174              3.03   2.822
                 30       11.39           9.401             11.39    16.3
Griewank         10     0.08124         0.05789           0.08124 0.06658
                 30    0.009922         0.01208          0.009922 0.01988
Rastrigin        10        19.9              19              19.5   18.95
                 30       106.5             109             106.5   109.9
Rosenbrock       10       22.79           33.86             22.79   3.542
                 30       49.05           49.06             49.05   124.9
ShiftedRastrigin 10       46.91           49.75             45.45   42.62
                 30       162.2           167.6             162.2   188.8
ShiftedSphere    10        1126            1136              1126    1152
                 30        7963            7470              7963    5948

## discrete level 10  (30 seeds, 12 instances)

UPHODS-v2 vs ablations (two-sided; W/T/L from UPHODS-v2's side)

| variant | W/T/L | UPHODS-v2 better on | Wilcoxon p | Holm | mean NS (variant / v2) |
|---|---|---|---|---|---|
| UPHODS-v2-fixedW | 0/12/0 | 8% | 1.00e+00 | 1.00e+00 | 19.6 / 19.7 |
| UPHODS-v2-noET | 3/8/1 | 58% | 3.91e-02 | 1.25e-01 | 17.7 / 19.7 |
| UPHODS-v2-noHys | 1/11/0 | 50% | 3.12e-02 | 1.25e-01 | 31.7 / 19.7 |
| UPHODS-v2-noRN | 0/12/0 | 17% | 6.25e-01 | 1.00e+00 | 19.6 / 19.7 |

Mean ranks (non-ablation methods): Single-PSO 1.71, UPHODS-v2 2.50, Single-SA 4.00, GP-PO-v2 4.29, SW-UCB 4.46, FSS-2 6.96, FSS-3 6.96, Single-GA 7.46, FSS-1 8.29, FSS-4 8.38, Single-GD 11.00
Friedman p = 8.8e-16

| vs | W/T/L | Wilcoxon p (one-sided) | Holm |
|---|---|---|---|
| Single-GD | 12/0/0 | 2.44e-04 | 2.44e-03 |
| FSS-1 | 11/1/0 | 4.88e-04 | 4.39e-03 |
| FSS-2 | 11/1/0 | 4.88e-04 | 4.39e-03 |
| FSS-3 | 11/1/0 | 4.88e-04 | 4.39e-03 |
| FSS-4 | 11/0/1 | 4.88e-04 | 4.39e-03 |
| SW-UCB | 8/4/0 | 9.77e-04 | 4.88e-03 |
| Single-GA | 10/0/2 | 9.77e-04 | 4.88e-03 |
| GP-PO-v2 | 8/4/0 | 1.95e-03 | 5.86e-03 |
| Single-SA | 6/5/1 | 6.54e-02 | 1.31e-01 |
| Single-PSO | 0/5/7 | 1.00e+00 | 1.00e+00 |

Per-instance medians (UPHODS-v2, -noRN, -fixedW, SW-UCB):

method                UPHODS-v2  UPHODS-v2-noRN  UPHODS-v2-fixedW    SW-UCB
function         dim                                                       
Ackley           10       11.49           11.49             11.49     11.49
                 30       13.51           13.51             13.51     17.99
Griewank         10          10              10                10        10
                 30        42.4            42.4              42.4      85.6
Rastrigin        10       202.3           202.3             202.3     204.2
                 30       612.6           612.6             612.6     658.5
Rosenbrock       10         210           348.3               210     378.7
                 30   1.697e+04       1.617e+04         1.697e+04 4.779e+04
ShiftedRastrigin 10       129.8           129.8             129.8     130.8
                 30       322.9           323.4             322.9     353.3
ShiftedSphere    10         338             338               338       338
                 30        7776            7618              7709      8351

## discrete level 30  (30 seeds, 12 instances)

UPHODS-v2 vs ablations (two-sided; W/T/L from UPHODS-v2's side)

| variant | W/T/L | UPHODS-v2 better on | Wilcoxon p | Holm | mean NS (variant / v2) |
|---|---|---|---|---|---|
| UPHODS-v2-fixedW | 0/12/0 | 25% | 8.44e-01 | 8.63e-01 | 19.6 / 19.6 |
| UPHODS-v2-noET | 2/10/0 | 67% | 2.06e-01 | 6.41e-01 | 17.3 / 19.6 |
| UPHODS-v2-noHys | 1/11/0 | 58% | 1.60e-01 | 6.41e-01 | 27.0 / 19.6 |
| UPHODS-v2-noRN | 0/12/0 | 50% | 4.32e-01 | 8.63e-01 | 19.7 / 19.6 |

Mean ranks (non-ablation methods): Single-PSO 1.71, UPHODS-v2 2.12, SW-UCB 3.33, GP-PO-v2 5.17, Single-SA 5.83, FSS-3 6.33, Single-GA 7.25, FSS-1 7.42, FSS-4 7.83, FSS-2 8.00, Single-GD 11.00
Friedman p = 5.0e-14

| vs | W/T/L | Wilcoxon p (one-sided) | Holm |
|---|---|---|---|
| FSS-1 | 12/0/0 | 2.44e-04 | 2.44e-03 |
| FSS-2 | 12/0/0 | 2.44e-04 | 2.44e-03 |
| FSS-3 | 12/0/0 | 2.44e-04 | 2.44e-03 |
| FSS-4 | 12/0/0 | 2.44e-04 | 2.44e-03 |
| Single-GA | 11/1/0 | 2.44e-04 | 2.44e-03 |
| Single-GD | 12/0/0 | 2.44e-04 | 2.44e-03 |
| GP-PO-v2 | 11/0/1 | 4.88e-04 | 2.44e-03 |
| Single-SA | 9/1/2 | 3.37e-02 | 1.01e-01 |
| SW-UCB | 5/6/1 | 2.85e-01 | 5.70e-01 |
| Single-PSO | 2/6/4 | 7.22e-01 | 7.22e-01 |

Per-instance medians (UPHODS-v2, -noRN, -fixedW, SW-UCB):

method                UPHODS-v2  UPHODS-v2-noRN  UPHODS-v2-fixedW  SW-UCB
function         dim                                                     
Ackley           10       4.335           4.335             4.335   4.335
                 30       10.17           10.13             10.26   17.24
Griewank         10           2               2                 2       2
                 30        22.4            21.2              22.4    27.2
Rastrigin        10       52.39           51.02             52.12   57.44
                 30       192.3           192.5             192.3   280.2
Rosenbrock       10       83.45           115.6              83.2   83.45
                 30        8952            9995              9912    4702
ShiftedRastrigin 10       34.94           33.48             34.94   47.91
                 30       216.6           216.6             216.6   289.1
ShiftedSphere    10       42.61           42.68             42.61   42.65
                 30        4014            4354              3488    2227
