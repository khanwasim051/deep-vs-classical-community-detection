"""Recompute every table and figure of the paper from all_runs.csv and the graph statistics (a few minutes).

    python analysis/run_all.py
"""
import os, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
for s in ["select_configs", "paired_tests", "rank_tests", "synthetic", "controls", "misc_tables", "plot_cd", "plot_pareto", "plot_sweep"]:
    print(f"== {s}")
    runpy.run_path(os.path.join(HERE, s + ".py"), run_name="__main__")
print("outputs in analysis/out")
