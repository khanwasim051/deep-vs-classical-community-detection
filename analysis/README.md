# Analysis scripts

`python analysis/run_all.py` recomputes every number, table and figure of the paper from `all_runs.csv`
(one row per graph, method, configuration and seed; written by `export_all.py`), `khat.json`, `khat_bethe.json`,
`graph_stats.txt` and `graph_stats_base.txt`. Outputs go to `analysis/out/`.

| Script | Output | Paper |
|---|---|---|
| `select_configs.py` | `selection.csv` | label-free configuration selection (Section 4.4), Table 18 |
| `paired_tests.py` | `equiv_sel.json` | paired differences, bootstrap intervals, Holm-corrected Wilcoxon tests, non-inferiority (Table 7) |
| `rank_tests.py` | `sel.json` | Tables 5, 6, 8, 9, 10, 18; Friedman/Nemenyi tests |
| `synthetic.py` | `syn.json` | Tables 13 and 15, Sections 6.1-6.2 stratified paired tests, Supplementary Tables S21-S23 |
| `controls.py` | `controls.json` | Tables 11 and 12 (both estimators of K), unknown-K paired comparisons, Supplementary Tables S19-S20 |
| `misc_tables.py` | `misc_tables.json` | Tables 4, 16, 17 and 19 |
| `plot_overview.py`, `plot_cd.py`, `plot_pareto.py`, `plot_sweep.py` | `*.png`, `Fig1.tif`–`Fig4.tif` (600 dpi) | Figs. 1, 2, 3 and 4 |

`common.py` holds the graph and method lists and the published accuracies used in Table 4.
Bootstrap intervals use fixed random seeds and 20,000 resamples (seed 0 for unstratified, seed 1 for the stratified
intervals of `synthetic.py`), and the stratified signed-rank test uses 100,000 sign flips with seed 2, so every
interval and p value is reproduced exactly.
