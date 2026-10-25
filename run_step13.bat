REM Step 13: two quick controls on the 24 sweep graphs (k-means with the deep methods' input dimension; Leiden given K)
python run_each.py baselines.py --dataset sweep --methods KMeans-featM,LeidenK --runs 1 --resume --out results_sweep
python sweep_plot.py --results results_sweep,results_sweep_pyagc,results_sweep_photo --set reference > sweep_ref.txt
echo Done. Send: results_sweep\sweep_table_ref.csv
pause
