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
- `--num_runs`: number of Monte Carlo draws (default `1000`)
- `--n`: sample size per draw
- `--metrics_csv`: output CSV path

Each run appends one row to `metrics_csv` with:
- model/DGP/args
- `mean_tau_hat`
- `mse_tau_hat`

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

### 2) Undirected GNN on equilibrium undirected DGP

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
  --metrics_csv results/metrics.csv
```

### 3) Directed GNN on directed DGP

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
  --metrics_csv results/metrics.csv
```

## SLURM usage

You can submit the batch script:

```bash
sbatch action.slurm
```
