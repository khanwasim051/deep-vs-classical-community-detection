REM Step 9: add Infomap (5 seeds on the 15 graphs, 1 seed on the 24 sweep graphs), then refresh the summaries
REM Run from the (mixflow) prompt in D:\mixflow_cd. Safe to rerun: finished runs are skipped.
python baselines.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --methods Infomap --runs 5 --resume
python baselines.py --dataset sweep --methods Infomap --runs 1 --resume --out results_sweep
python stats.py --results results,results_pyagc --methods "MixFlow-CD_v2,RegSpectral,Louvain,Leiden,Infomap,LabelProp,KMeans-feat,GAE[PyAGC],DGI[PyAGC],DAEGC[PyAGC],DMoN[PyAGC],DinkNet[PyAGC],SSGC[PyAGC],NS4GC[PyAGC],CCASSG[PyAGC],MS2CAG[PyAGC]" > stats_all.txt
python sweep_plot.py --results results_sweep,results_sweep_pyagc --set reference > sweep_ref.txt
python runtime_table.py > runtime.txt
echo Done. Send: stats_all.txt, results_sweep\sweep_table_ref.csv, runtime_table.csv
pause
