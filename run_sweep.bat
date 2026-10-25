pip install matplotlib
python gen_sweep.py
python run_each.py baselines.py --dataset sweep --methods Louvain,Leiden,LabelProp,RegSpectral,KMeans-feat --runs 1 --resume --out results_sweep
python run_each.py deep_baselines.py --dataset sweep --runs 1 --resume --out results_sweep
python run_each.py train.py --dataset sweep --runs 1 --quiet --resume --tag v2 --out results_sweep
python sweep_plot.py --results results_sweep
pause
