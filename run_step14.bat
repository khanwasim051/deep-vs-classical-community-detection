REM Step 14: label-free configuration selection (reviewer concern 4).
REM Every tunable method gets four candidate configurations fixed in advance; the choice is made later by modularity, without labels.
REM Run from the (mixflow) prompt in D:\mixflow_cd. Long run: start it in the evening. Safe to stop and rerun.

REM 1. Classical methods: four values of their main parameter (the fourth value is the one already run)
python run_each.py baselines.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --methods RegSpectral_t0.25,RegSpectral_t0.5,RegSpectral_t2,Louvain_r0.5,Louvain_r2,Louvain_r4,Leiden_r0.5,Leiden_r2,Leiden_r4 --runs 5 --resume --out results_cfg

REM 2. Reference deep methods: PyAGC's four full-batch published configurations (Cora, Photo, Physics, HM)
python run_each.py pyagc_run.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --config Physics --out results_cfg
python run_each.py pyagc_run.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --config HM --out results_cfg
python run_each.py pyagc_run.py --dataset dolphins,ukfaculty,pubmed,lfr3,lfr4,lfr7 --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --config Photo --out results_sens
python run_each.py pyagc_run.py --dataset amazon-photo --runs 5 --split-methods --methods MS2CAG,SSGC,DGI,CCASSG,NS4GC,DinkNet,GAE,DAEGC,DMoN --config Cora --out results_cfg

REM 3. MixFlow-CD: three alternative settings (the default is already run)
python run_each.py train.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --runs 5 --quiet --resume --tag cfgB --mask 0.3 --out results_cfg
python run_each.py train.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --runs 5 --quiet --resume --tag cfgC --lam_q 2.0 --out results_cfg
python run_each.py train.py --dataset email-eu-core,football,dolphins,ukfaculty,kki,cora,citeseer,pubmed,amazon-photo,lfr1,lfr2,lfr3,lfr4,lfr6,lfr7 --runs 5 --quiet --resume --tag cfgD --epochs_pre 100 --epochs_ft 100 --out results_cfg

REM 4. One file with every run
python export_all.py
python check_coverage.py --results results_cfg > coverage_cfg.txt
echo Done. Send: all_runs.csv and coverage_cfg.txt (and run_logs\pyagc_run\FAILED.txt if it exists)
pause
