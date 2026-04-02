# Supply_GNN_Causal

This repo runs Monte Carlo simulations for tau-hat estimation under multiple models and DGP variants.

## Main runner

Use `sim.py`:

```bash
python sim.py --help
```

Detailed args:
- `--model` (str, default `linear`): estimator family.
  - `linear`: linear DR estimator.
  - `gnn`: undirected GNN nuisance fits + DR aggregation.
  - `dirgnn`: directed dual-channel GNN nuisance fits + DR aggregation.
- `--DGP` (str, default `simple_undir`): data generator.
  - `undir`: undirected equilibrium treatment DGP.
  - `simple_undir`: undirected simple Bernoulli-treatment DGP.
  - `dir`: directed equilibrium treatment DGP.
- `--num_runs` (int): Monte Carlo replications (one tau-hat per draw).
- `--n` (int): sample size per draw.
- `--seed` (int): base seed; draw `i` uses `seed + i`.
- `--gen_graph` (str, default `rgg`): graph type inside DGP (`rgg` or `er`).
- `--tau_true` (float): reference true tau for MSE computation.
- `--p_treat` (float): Bernoulli treatment probability for `simple_undir`.
- `--features` (str, default `nodes`):
  - `X`: scalar/tabular covariate channel.
  - `nodes`: maps to internal `node_features`.
  - For `gnn`/`dirgnn`, passing `X` is auto-upgraded to `node_features`.
- `--clip` (float): propensity clipping level in DR score.
- `--L` (int): number of GNN hidden layers.
- `--output_dim` (int): hidden width of each GNN layer.
- `--variance_type` (str, default `skeleton`): per-draw SE estimator family for GNN models.
  - `iid`: iid-style variance.
  - `skeleton`: undirected/symmetrized network variance.
  - `directed`: directed network variance.
- `--variance_method` (str, default empty): method under chosen `variance_type`.
  - If empty, defaults are: `iid -> iid`, `skeleton -> max`, `directed -> dir_max`.
- Valid `variance_type` / `variance_method` pairs:
  - `iid`: method is effectively fixed to `iid` (argument is ignored).
  - `skeleton`: `u`, `pd`, `max`.
  - `directed`: `in_max`, `out_max`, `dir_max`, `dir_avg`.
- `--bandwidth` (int or omitted): fixed bandwidth for network variance; omitted means auto-selected.
- `--use_gpu` (int, default `1`): for GNN models, `1` allows CUDA when available and `0` forces CPU.
- `--metrics_csv` (path): output CSV file. Each invocation appends one row.

Notes:
- If `--model gnn` and `--variance_type directed`, code warns and falls back to `skeleton`.
  - In that fallback, if `--variance_method` is empty or starts with `dir`, it is reset to `max`.
- Device choice is printed to stdout (`use_gpu_arg`, CUDA availability, and actual device), but not saved in CSV.

Each run appends one row to `metrics_csv` with columns:
- `model,DGP,num_runs,n,seed,gen_graph,tau_true,p_treat,features,clip,L,output_dim,variance_type,variance_method,bandwidth`
- `MSE` (stored as `mse_tau_hat (se_mse_tau_hat)`, for example `0.118 (0.003241)`)
- `se_tau_hat_MC` (Monte Carlo SE of tau-hat across replications)
- `mean_sigma_hat` (average estimated per-draw standard error)
- `cover_rate` (share of runs where `tau_true` is inside `tau_hat +/- 1.96*se_hat`)

## Examples

### 1) Linear model on undirected simple DGP

```bash
python sim.py \
  --model linear \
  --DGP simple_undir \
  --num_runs 100 \
  --n 200 \
  --seed 123 \
  --gen_graph rgg \
  --tau_true 2.0 \
  --p_treat 0.5 \
  --features X \
  --clip 0.001 \
  --L 2 \
  --output_dim 6 \
  --metrics_csv results/metrics.csv
```

### 2) Undirected GNN with symmetrized variance (`K_max`)

```bash
python sim.py \
  --model gnn \
  --DGP undir \
  --num_runs 100 \
  --n 200 \
  --seed 123 \
  --gen_graph rgg \
  --tau_true 2.0 \
  --p_treat 0.5 \
  --features nodes \
  --clip 0.001 \
  --L 2 \
  --output_dim 6 \
  --variance_type skeleton \
  --variance_method max \
  --metrics_csv results/metrics.csv
```

### 3) Directed GNN with directed variance (`K_dir_max`)

```bash
python sim.py \
  --model dirgnn \
  --DGP dir \
  --num_runs 100 \
  --n 200 \
  --seed 123 \
  --gen_graph rgg \
  --tau_true 2.0 \
  --p_treat 0.5 \
  --features nodes \
  --clip 0.001 \
  --L 2 \
  --output_dim 6 \
  --variance_type directed \
  --variance_method dir_max \
  --metrics_csv results/metrics.csv
```

## SLURM usage

Default batch run (uses `test/sim_gnn.py` with directed settings from `action.slurm`):

```bash
sbatch action.slurm
```

Override command on submission:

```bash
sbatch --export=SIM_SCRIPT=sim.py,SIM_ARGS="--model dirgnn --DGP dir --num_runs 300 --n 500 --variance_type directed --variance_method dir_max --metrics_csv results/dirgnn_metrics.csv" action.slurm
```

Run undirected GNN with symmetrized variance via SLURM:

```bash
sbatch --export=SIM_SCRIPT=sim.py,SIM_ARGS="--model gnn --DGP undir --num_runs 300 --n 500 --variance_type skeleton --variance_method max --metrics_csv results/gnn_metrics.csv" action.slurm
```

## Timing example (per replicate)

If you want an approximate time-per-replication from one run, redirect logs and divide total evaluation time by `num_runs`:

```bash
python sim.py --model dirgnn --DGP dir --num_runs 2 --n 2000 --features nodes --variance_type skeleton --variance_method max > results/run.log 2>&1 && \
awk '/evaluation_time_sec=/{for(i=1;i<=NF;i++) if($i ~ /^evaluation_time_sec=/){split($i,a,"="); t=a[2]}} END{printf("seconds_all=%.6f\n", t)}' results/run.log
```
