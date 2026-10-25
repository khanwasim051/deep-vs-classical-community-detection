# MixFlow-CD

**Tie-aware dual-attention encoder + mixture-prior normalizing flow for community detection**

Pure PyTorch, full-batch. It is sized for an RTX 3050 laptop GPU (6 GB VRAM, 16 GB RAM) and does not need `torch_scatter` or `torch_sparse`.

## What the model does

1. **Input features**
   - **Attributed graphs** (Cora, Citeseer, PubMed, Amazon-Computers, Amazon-Photo): the L2-normalized node attributes.
   - **Non-attributed graphs** (Email-Eu-core, Football, Dolphins, UKfaculty, KKI, LFR): spectral-diffusion features. A fixed PPR-type filter is applied to the top eigenvectors of the normalized adjacency.
2. **Tie classes**: every edge is labelled *weak* (no common neighbour), *medium* or *strong* by its Jaccard embeddedness.
3. **Encoder**:
   - **Tie-aware attention**: edge-masked, multi-head, with one value projection and one score bias per tie class.
   - **Prototype attention**: M learnable tokens pool from all nodes, and every node reads back from them. This gives global context in O(nMd) time instead of O(n²).
4. **Pre-training**: masked-feature reconstruction with a scaled cosine error (GraphMAE-style), plus an optional link-reconstruction loss.
5. **Clustering head (MixFlow)**:
   - A RealNVP flow `u = f(z)` maps embeddings into a space with a K-component diagonal Gaussian-mixture prior.
   - `log p(z)` is exact (change of variables), and community membership is the posterior responsibility.
   - `BatchNorm(affine=False)` on `z` prevents the encoder from cheating the likelihood by shrinking `z`.
   - Bounded coupling scales and a variance floor keep the density bounded.
6. **Fine-tuning loss**: `NLL/d − λ_Q·Q_soft + λ_c·collapse + λ_rec·reconstruction`.
   - `λ_Q` is set by a PI controller (ControlVAE-style). It makes the mean assignment entropy track a decreasing schedule.
   - Responsibilities are tempered (`tau = zdim`) inside the losses. Otherwise they are almost one-hot in 64 dimensions and the modularity gradient vanishes.
   - The predicted partition is `argmax`, which does not depend on `tau`.
7. **Input skip and sizes**: a linear skip from the input features to `z` stops the encoder from losing information the features already carry. For non-attributed graphs, the structural feature size and `z` size grow with K (`min(max(64, K+16), 512)`), so graphs with many communities (LFR-2, LFR-5, KKI) are not squeezed into 64 dimensions.
8. **Temperature control**: by default the PI controller drives the responsibility temperature so that assignment entropy follows the schedule (`--control tau`); the start temperature is found by bisection. `--control lam` gives the older behaviour (controller drives λ_Q).
9. **K unknown (optional)**: BIC on the exact likelihood (`--estimate_k 2-30`).

No labels are used for training or for choosing the reported epoch. `--select modularity` picks the epoch by hard modularity, which is label-free. `--diag` prints label-based metrics during training, for debugging only.

## Installation on the RTX 3050 laptop

The commands are the same on Windows (Anaconda Prompt) and Linux.

```bash
conda create -n mixflow python=3.11 -y
conda activate mixflow
# CUDA 12.1 build of PyTorch (works with any recent NVIDIA driver):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Running

```bash
# one dataset, 5 seeds
python train.py --dataset cora --runs 5

# everything (10 real-world + 5 LFR), then the classical baselines, then the tables and statistical tests
python train.py --dataset all --runs 5
python baselines.py --dataset all --runs 5
python stats.py --results results
```

You can also run `run_all.bat` (Windows) or `bash run_all.sh` (Linux). Results are appended to `results/<dataset>.csv`, one row per seed.

Datasets download automatically into `data/` the first time they are used. LFR graphs are generated once and cached.

### Ablations (each run gets its own method name in the CSV via `--tag`)

```bash
python train.py --dataset all --no_ties        --tag noTies     # plain attention instead of tie-aware
python train.py --dataset all --no_proto       --tag noProto    # no global prototype attention
python train.py --dataset all --flow_layers 0  --tag noFlow     # plain GMM on z (no flow)
python train.py --dataset all --static         --tag static     # fixed lambda_Q, no PI control
python train.py --dataset all --lam_link 1     --tag link       # add link reconstruction in pre-training
python train.py --dataset all --no_skip        --tag noSkip     # no input skip
python train.py --dataset all --encoder none   --tag headOnly   # mixture-flow head directly on the input features
python train.py --dataset all --control lam    --tag ctrlLam    # PI control of lambda_Q instead of temperature
python train.py --dataset all --estimate_k 2-60                 # K-unknown setting (BIC)
python train.py --dataset lfr6,lfr7                             # harder LFR graphs (mu = 0.6, 0.7)
```

### If you run out of GPU memory

This is most likely on Amazon-Computers, which has about 490k directed edges. Try these in order:

1. `--hid 64`
2. `--heads 2`
3. `--max_pos 50000`

## Dataset notes

The loader prints the node, edge and community counts for every dataset. Please check them against the counts you intend to report.

| Dataset | Source | Notes |
|---|---|---|
| Email-Eu-core | SNAP (full 1005 nodes) | Falls back to a GitHub mirror of the largest connected component (986 nodes) if SNAP is unreachable. A warning is printed when that happens. |
| Football, Dolphins | `vlivashkin/community-graphs` (GML with ground truth) | 115 / 62 nodes |
| UKfaculty | igraphdata (R) | The 2-member group 4 is dropped, giving 79 nodes, 552 edges and 3 groups |
| KKI | TUDataset | Disjoint union of 83 brain graphs; the "communities" are the graphs themselves. This makes it close to trivial for any component-aware method. |
| Amazon-Computers/Photo, Cora, Citeseer, PubMed | PyTorch Geometric | Standard versions |
| LFR 1–5 (+6, 7) | networkx generator | N, average degree, max degree, τ1, τ2 and μ follow the benchmark tuples. Community-size bounds are chosen to give roughly the target number of communities, so the generated count is printed. |

## Files

- `data.py`: dataset loaders and LFR generation
- `features.py`: structural features and tie classes
- `model.py`: encoder, MixFlow head, soft modularity, PI controller
- `train.py`: training and evaluation (ACC with Hungarian matching, NMI, ARI, Macro-F1, plus modularity, time and peak GPU memory)
- `baselines.py`: Louvain, Leiden, label propagation, spectral clustering, regularized spectral clustering (RegSpectral), k-means on features
- `deep_baselines.py`: GAE + k-means, DMoN, DAEGC (plain-PyTorch re-implementations, same inputs and protocol)
- `stats.py`: mean ± std table, Friedman / Iman–Davenport tests, Nemenyi critical difference
