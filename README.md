# Supply_GNN_Causal

This repo runs Monte Carlo simulations for tau-hat estimation under multiple models and DGP variants.

## Main runner

Use `sim.py`:

```bash
python sim.py --help
```

Key args:
- `--model`: `linear`, `gnn`, `dirgnn`
- `--DGP`: `undir`, `simple_undir`, `dir`
- `--num_runs`: number of Monte Carlo draws
- `--n`: sample size per draw
- `--variance_type`: `iid`, `skeleton`, `directed` (for `gnn`/`dirgnn`)
- `--variance_method`: kernel selector inside variance type
- `--bandwidth`: optional fixed bandwidth (otherwise auto from graph)
- `--metrics_csv`: output CSV path

Each run appends one row to `metrics_csv` with:
- model/DGP/args
- `mean_tau_hat`
- `mse_tau_hat`
- `se_tau_hat` (Monte Carlo SE of tau-hat across replications)
- `mean_se_hat` (average estimated SE from per-draw variance estimator)

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
