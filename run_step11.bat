REM Step 11: rerun the mu sweep with PyAGC's Photo configuration for the four methods it helps (1 seed per graph)
REM Run from the (mixflow) prompt in D:\mixflow_cd. Safe to rerun: finished runs are skipped.
python run_each.py pyagc_run.py --dataset sweep --runs 1 --split-methods --methods GAE,DAEGC,DMoN,DinkNet --config Photo --out results_sweep_photo
python check_coverage.py --results results_sweep_photo --runs 1 > coverage_sweep_photo.txt
python sweep_plot.py --results results_sweep,results_sweep_pyagc,results_sweep_photo --set reference > sweep_ref.txt
type coverage_sweep_photo.txt
echo Done. Send: coverage_sweep_photo.txt and results_sweep\sweep_table_ref.csv
pause
