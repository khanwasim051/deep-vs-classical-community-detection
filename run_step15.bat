REM Step 15: external validity (reviewer concern 5). Run AFTER Step 14 has finished.
REM Run from the (mixflow) prompt in D:\mixflow_cd. Long run: start it in the evening. Safe to stop and rerun.

REM 1. Descriptive statistics of every graph, including the realized mixing of each LFR graph
python graph_stats.py --dataset all,lfr6,lfr7,attr_extra,het,sweep,sweep_more > graph_stats.txt

REM 2. Six more attributed graphs (three of them heterophilous): every method, every candidate configuration, 5 seeds
python run_each.py baselines.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --methods Louvain,Leiden,Infomap,LabelProp,RegSpectral,KMeans-feat,LeidenK,SpecAttr --runs 5 --resume --out results_extra
python run_each.py baselines.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --methods RegSpectral_t0.25,RegSpectral_t0.5,RegSpectral_t2,Louvain_r0.5,Louvain_r2,Louvain_r4,Leiden_r0.5,Leiden_r2,Leiden_r4 --runs 5 --resume --out results_extra
python run_each.py pyagc_run.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --config Cora --out results_extra
python run_each.py pyagc_run.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --config Photo --out results_extra
python run_each.py pyagc_run.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --config Physics --out results_extra
python run_each.py pyagc_run.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --config HM --out results_extra
python run_each.py train.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --runs 5 --quiet --resume --tag v2 --out results_extra
python run_each.py train.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --runs 5 --quiet --resume --tag cfgB --mask 0.3 --out results_extra
python run_each.py train.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --runs 5 --quiet --resume --tag cfgC --lam_q 2.0 --out results_extra
python run_each.py train.py --dataset wiki,blogcatalog,chameleon,squirrel,actor,coauthor-cs --runs 5 --quiet --resume --tag cfgD --epochs_pre 100 --epochs_ft 100 --out results_extra

REM 3. Synthetic graphs: 18 LFR graphs with different degree and community-size heterogeneity, and 24 more sweep graphs
REM    (classical methods and controls now; the deep methods follow in Step 16 with their label-free selected configuration)
python run_each.py baselines.py --dataset het,sweep_more --methods Louvain,Leiden,Infomap,LabelProp,RegSpectral,KMeans-feat,LeidenK,KMeans-featM --runs 1 --resume --out results_syn
python run_each.py baselines.py --dataset het,sweep_more --methods RegSpectral_t0.25,RegSpectral_t0.5,RegSpectral_t2,Louvain_r0.5,Louvain_r2,Louvain_r4,Leiden_r0.5,Leiden_r2,Leiden_r4 --runs 1 --resume --out results_syn

REM 4. One file with every run
python export_all.py
python check_coverage.py --results results_extra > coverage_extra.txt
echo Done. Send: all_runs.csv, graph_stats.txt, coverage_extra.txt (and run_logs\pyagc_run\FAILED.txt if it exists)
pause
