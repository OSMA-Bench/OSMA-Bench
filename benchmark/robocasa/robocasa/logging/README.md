# Rollout logging and post-evaluation

## Storage layout

Policy workers record episode results during rollout. Post-evaluation reads the
ledger and computes the final metrics. SSI is not computed during inference.

```
<output>/                         # raw rollout for one policy
  layout*/<task>/                 # run.log, trajectory_log.json, overview
  results*.json                   # worker results

<ledger>/                          # post-evaluation input (ROBOCASA_LEDGER_DIR)
  run.json                         # model/policy/seed metadata
  episodes.jsonl                   # per-episode success, distance, contact summary
  traj/<episode>.npz               # pose, distance, contact, v/a/jerk series
  derived/rates.json               # cumulative TSR/CSR for completed episodes
```

At the end of each episode, `task_success`, `collision_free_success`,
`contact_steps`/`collision_steps`, path length/time, and trajectory series are
appended to the ledger. `derived/rates.json` is updated over all episodes
completed so far.

## Post-evaluation entry point

The calculation is implemented by
`robocasa.metrics.summarize.summarize_post_evaluation`; the CLI is the following
wrapper.

```bash
python -m robocasa.scripts.summarize_post_evaluation_metrics \
  --inputs OUTPUT_A OUTPUT_B \
  --scope all task_success collision_free_task_success \
  --comparison individual \
  --out summary.json
```

`--inputs` accepts either output folders or ledger folders. For an output folder,
the command automatically discovers the ledger inside it.

The post-evaluation API uses the same rule: pass output folders to
`summarize_post_evaluation`; nested shard ledgers are discovered automatically
and combined before scope filtering and SSI aggregation.

### Filtering and comparison options

- `--scope all`: include all successful and failed episodes
- `--scope task_success`: include task-success episodes only
- `--scope collision_free_task_success`: include task and collision-free successes only
- `--comparison individual`: use each model's own eligible task set
- `--comparison matched_intersection`: compare models only on the common
  `(layout, route, obstacle)` set passing the selected scope
- `--allow-partial` (default): use pairs from cells containing at least two of H/M/L
- `--strict-complete`: use only cells containing all three H/M/L tiers
- `--optimal PATH`: optimal-path JSON. If omitted or unavailable, normalized
  path metrics are reported as `n/a`

SSI groups obstacles into H/M/L tiers within each `(layout, route)` cell.
`min_distance` uses one minimum over the entire trajectory. `v/d`, `a/d`, and
`J/d` compute episode mean/max values over timesteps with `d_t ≤ 1.25m`, then
average obstacles within each tier. The denominator distance at collision
timesteps is floored at `ε=0.05m`. Cell-level Kendall tau and H-M/M-L/H-L
margins are then aggregated.

## Mean/std across seeds

Pass multiple seed ledgers for the same model with `--aggregate-seeds` to first
compute one summary per seed and then report the across-seed `mean`, `std`, and
`n_seeds`. Shard ledgers are grouped automatically by seed.

```bash
python -m robocasa.scripts.summarize_post_evaluation_metrics \
  --inputs outputs/gemma4_seed0_vox_s*_ledger \
           outputs/gemma4_seed3_vox_s*_ledger \
           outputs/gemma4_seed16_vox_s*_ledger \
  --aggregate-seeds \
  --scope all task_success collision_free_task_success \
  --out gemma4_seed_summary.json
```

The output includes per-seed results plus `mean/std/n_seeds` for TSR, CSR,
normalized path metrics, and each metric's SSI Kendall tau.

The validation helper is `robocasa.metrics.summarize.check_post_evaluation`;
no separate inference-time SSI calculation or ledger worker is required.

## Parallel workers and shard merging

The 125 task classes can be split across four shards (`s0`-`s3`). Each worker
writes its own output/ledger. After the batch, `scripts/merge_seed_shards.py`
merges them into one model-and-seed folder.

```text
outputs/<model>_seed<seed>_voxposer/
  layout*/
  results.json
  ledger/episodes.jsonl
  ledger/traj/*.npz
```

`merge_seed_shards.py` combines each worker's `layout*/` task directories into
one model-and-seed output folder and merges `results*.json`/
`results_progress*.jsonl`. It creates a separate `ledger/`, appends
`episodes.jsonl`, and copies `traj/*.npz` while preserving episode filenames.
If the same file has different contents, merging stops with a conflict instead
of silently overwriting it.

The LMP cache is enabled by default. VLM requests containing images may bypass
the cache because the image differs between requests.

Worker output is resumable, so completed tasks are not run again.
