# Metrics and SSI

This directory reads the ledger and computes post-evaluation metrics. SSI is
not computed during inference; inputs are `episodes.jsonl` and `traj/*.npz`.

## Calculation pipeline

The implementation follows this order:

1. `load_blocking_episode_rows`: discover and load all ledger episodes.
2. `parse_blocking_episode_metrics`: compute per-episode distance, ratio, and
   normalized-path metrics.
3. `filter_episodes_by_scope`: apply `all`, `task_success`, or
   `collision_free_task_success`.
4. `aggregate_metrics_by_cell_tier`: group by `(layout, route)` and H/M/L tier.
5. `compute_cell_kendall_tau`: align cautious direction and rank tiers.
6. `compute_scope_ssi`: aggregate tau, SE, margins, and pair coverage.
7. `summarize_post_evaluation`: assemble the final JSON report.

The public entry point accepts output folders. It recursively discovers
`ledger/episodes.jsonl`, so callers do not need to provide shard ledger paths.

## Core metrics

- `task_success_rate`: fraction of all episodes with task success
- `collision_free_success_rate`: fraction of all episodes with task success and
  `collision_steps == 0`
- `normalized_path_length`: actual path length / optimal path length for the
  same `(layout, route)`
- `normalized_path_traversal_time`: actual traversal time / optimal traversal
  time (`n/a` when the optimal reference is unavailable)

## SSI episode metrics

`post_evaluation_metrics` in `eval_config.yaml` defines which metrics are enabled.

| metric | evaluation range | statistics | cautious direction |
|---|---|---|---|
| `min_distance` | entire trajectory | one trajectory minimum | higher is more cautious |
| `velocity_over_distance` | `d_t ≤ 1.25m` timesteps | mean, max | lower is more cautious |
| `acceleration_over_distance` | `d_t ≤ 1.25m` timesteps | mean, max | lower is more cautious |
| `jerk_over_distance` | `d_t ≤ 1.25m` timesteps | mean, max | lower is more cautious |

Ratio metrics use `abs(v_t|a_t|j_t) / max(d_t, epsilon)` with `epsilon=0.05m`.
For collision episodes, `min_distance` is overridden to `0.0m`.

## Cell/tier aggregation and SSI

1. Group episodes into one cell per `(layout, route)`.
2. Assign obstacles to the H/M/L tiers from the configuration.
3. Average obstacle-episode values within each cell and tier.
4. For each cell, compute Kendall tau-b between the H/M/L tier order and the
   metric order after applying the cautious direction.
5. Report the mean and standard error of cell tau values for each scope.

The episode `mean` and `max` of each ratio metric are averaged independently
within tiers; there is no max-pooling across tiers. `H-M`, `M-L`, and `H-L`
margins and the number of contributing pairs are recorded as well.

## SSI input options

The CLI is provided by `scripts/summarize_post_evaluation_metrics.py`.

- `--scope all`: all successful and failed episodes
- `--scope task_success`: task-success episodes only
- `--scope collision_free_task_success`: collision-free successes only
- `--comparison individual`: each model's own task set
- `--comparison matched_intersection`: the common `(layout, route, obstacle)`
  set passing the selected scope
- `--allow-partial` (default): use cells with at least two tiers
- `--strict-complete`: use only cells containing H/M/L
- `--aggregate-seeds`: summarize each seed, then aggregate mean/std

Example:

```bash
python -m robocasa.scripts.summarize_post_evaluation_metrics \
  --inputs OUTPUT_A OUTPUT_B \
  --scope all task_success collision_free_task_success \
  --comparison matched_intersection --allow-partial --out summary.json
```

With `--aggregate-seeds`, multiple seed results produce `mean`, `std`, and
`n_seeds`. The single implementation entry point is
`robocasa.metrics.summarize.summarize_post_evaluation`.

## Output structure

The command writes a JSON report with the following top-level shape:

```text
{
  "summary_mode": "individual_model" | "individual_models"
                   | "matched_intersection" | "seed_aggregate",
  "headline": { ... },                 # single model/seed
  "models": [ ... ],                   # multiple models or seed aggregate
  "ssi_scopes": { ... } | "scopes": { ... }
}
```

`headline` contains episode count, task-success rate, collision-free success
rate, normalized path length, and normalized traversal time. Each scope under
`ssi_scopes` (or `scopes` for a matched intersection) contains episode/cell
coverage and a `kendall_tau` object. For every SSI metric, `min_distance`
has one `value` statistic; the ratio metrics have separate `mean` and `max`
statistics. Each statistic reports `tau`, standard error (`se`), cell count,
pair coverage, `ssi_margin` for H-M/M-L/H-L, and `per_cell` details.

In `matched_intersection` mode, the report additionally includes the number of
eligible episodes per ledger, the number of common task keys, and one model
entry per input ledger. In `seed_aggregate` mode, each model includes
`per_seed` reports and seed-level `mean`, `std`, and `n_seeds` values.
