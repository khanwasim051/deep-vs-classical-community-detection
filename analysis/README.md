# Analysis scripts

`python analysis/run_all.py` recomputes every number, table and figure of the paper from `all_runs.csv`
(one row per graph, method, configuration and seed; written by `export_all.py`), `khat.json`,
`graph_stats.txt` and `graph_stats_base.txt`. Outputs go to `analysis/out/`.

| Script | Output | Paper |
|---|---|---|
| `select_configs.py` | `selection.csv` | label-free configuration selection (Section 4.4), Table 18 |
| `paired_tests.py` | `equiv_sel.json` | paired differences, bootstrap intervals, Holm-corrected Wilcoxon tests, non-inferiority (Table 7) |
| `rank_tests.py` | `sel.json` | Tables 5, 6, 8, 9, 10, 18; Friedman/Nemenyi tests |
| `synthetic.py` | `syn.json` | Tables 13 and 15, Section 6.1-6.2 paired tests |
| `controls.py` | `controls.json` | Tables 11 and 12, unknown-K paired comparisons |
| `misc_tables.py` | `misc_tables.json` | Tables 4, 16, 17 and 19 |
| `plot_cd.py`, `plot_pareto.py`, `plot_sweep.py` | `*.png` | Figs. 2, 3 and 4 |

`common.py` holds the graph and method lists and the published accuracies used in Table 4.
Bootstrap intervals use a fixed random seed (0) and 20,000 resamples, so they are reproduced exactly.
