"""Generate (once) and check the 24 LFR graphs of the mixing-parameter sweep.

    python gen_sweep.py
Graphs are cached in data/lfr/; running it again only re-reads them.
"""
import time
from data import SWEEP, describe, load_dataset

for name in SWEEP:
    t0 = time.time()
    g = load_dataset(name, "data")
    print(f"{describe(g)}   ({time.time() - t0:.0f}s)", flush=True)
print("all sweep graphs ready")
