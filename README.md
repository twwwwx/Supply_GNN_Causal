# Supply_GNN_Causal

This repo runs Monte Carlo simulation for the Section-5 spillover setup with vector exposure
`T_i = (D_i, rho_in_i, rho_out_i)` and four estimands:
`tau_dir`, `tau_in`, `tau_out`, `tau_tot`.

## Main runner

Use `sim.py`:

```bash
python sim.py --help
```

### Key arguments

- `--model` (str, default `dirgnn`): `gnn` or `dirgnn`.
- `--DGP` (str, default `spillover`): must be `spillover`.
- `--num_runs` (int): Monte Carlo replications.
- `--n` (int): sample size per replication.
- `--seed` (int): base seed; replication `i` uses `seed + i`.
- `--gen_graph` (str): `rgg` or `er`.
- `--tau_dir_true` (float): true direct effect in DGP.
- `--tau_in_true` (float): true upstream spillover effect in DGP.
- `--tau_out_true` (float): true downstream spillover effect in DGP.
  - `tau_tot_true` is derived internally as `tau_dir_true + tau_in_true + tau_out_true`.
- `--features` (str, default `nodes`):
  - `nodes` maps to internal `node_features`.
  - `X` is auto-upgraded to `node_features` for compatibility.
- `--clip` (float): propensity clipping floor for the 8-class generalized propensity.
- `--L` (int): number of GNN hidden layers.
- `--output_dim` (int): hidden width of each GNN layer.
- `--variance_type` (str): `iid`, `skeleton`, or `directed`.
- `--variance_method` (str):
  - `iid`: effectively `iid`
  - `skeleton`: `u`, `pd`, `max`
  - `directed`: `in_max`, `out_max`, `dir_max`, `dir_avg`
- `--bandwidth` (int or omitted): fixed bandwidth for network variance; omitted means auto-selected.
- `--use_gpu` (int, default `1`): `1` allows CUDA when available, `0` forces CPU.
- `--metrics_csv` (path): output CSV file. Each invocation appends one wide-format row.

Notes:
- If `--model gnn` and `--variance_type directed`, code warns and falls back to `skeleton`.
- Device choice is printed to stdout.

## Metrics output (wide format)

Each invocation appends one row with:

- Metadata:
  - `model,DGP,num_runs,n,seed,gen_graph`
  - `tau_dir_true,tau_in_true,tau_out_true,tau_tot_true`
  - `features,clip,L,output_dim,variance_type,variance_method,bandwidth`
- Compact estimand cells:
  - `tau_dir,tau_in,tau_out,tau_tot`
  - each cell is `mean_tau_hat(se_tau_hat_MC)`, for example `2.031(0.145200)`
- Per-estimand variance diagnostics:
  - `mean_sigma_hat_tau_dir,mean_sigma_hat_tau_in,mean_sigma_hat_tau_out,mean_sigma_hat_tau_tot`
  - `MSE_tau_dir,MSE_tau_in,MSE_tau_out,MSE_tau_tot` where each is `mse(se_mse)`
  - `cover_rate_tau_dir,cover_rate_tau_in,cover_rate_tau_out,cover_rate_tau_tot`

## Examples

### Directed dual-channel GNN with directed variance

```bash
python sim.py \
  --model dirgnn \
  --DGP spillover \
  --num_runs 2 \
  --n 200 \
  --seed 123 \
  --gen_graph rgg \
  --tau_dir_true 2.0 \
  --tau_in_true 1.0 \
  --tau_out_true 1.0 \
  --features nodes \
  --clip 0.001 \
  --L 2 \
  --output_dim 6 \
  --variance_type directed \
  --variance_method dir_max \
  --metrics_csv results/metrics.csv
```

### Undirected GNN with skeleton variance

```bash
python sim.py \
  --model gnn \
  --DGP spillover \
  --num_runs 100 \
  --n 200 \
  --seed 123 \
  --gen_graph er \
  --tau_dir_true 2.0 \
  --tau_in_true 1.0 \
  --tau_out_true 1.0 \
  --features nodes \
  --clip 0.001 \
  --L 2 \
  --output_dim 6 \
  --variance_type skeleton \
  --variance_method max \
  --metrics_csv results/metrics.csv
```

## SLURM usage

```bash
sbatch action.slurm
```
