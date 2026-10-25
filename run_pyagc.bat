python run_each.py pyagc_run.py --dataset all --runs 5 --split-methods
python check_coverage.py --results results_pyagc > coverage.txt
python stats.py --results results,results_pyagc --methods "MixFlow-CD_v2,RegSpectral,Louvain,Leiden,LabelProp,KMeans-feat,GAE[PyAGC],DGI[PyAGC],DAEGC[PyAGC],DMoN[PyAGC],DinkNet[PyAGC],SSGC[PyAGC],NS4GC[PyAGC],CCASSG[PyAGC],MS2CAG[PyAGC]" > stats_all.txt
type coverage.txt
pause
