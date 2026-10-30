# PIVOT-nav — a constraint-optimization planner for 2D kitchen navigation

[PIVOT](https://rekep-robot.github.io/) (relational keypoint constraints) ported
from OmniGibson 6-DoF arm manipulation to 2D navigation in RoboCasa kitchens.
The original does not run here as-is, so what was carried over is the
**algorithmic structure**, not the code.

In the mode the paper describes, the VLM answers **once** with three things —
how many stages the drive has, where each ends, and what the path must respect —
and an optimizer solves the geometry between them:

```python
# STAGES: 2
# 1: Get past the human with room to spare
# 2: Drive to the star

def stage1_subgoal_constraint1(state, keypoints):
    """Finish this phase already clear of the human."""
    return standoff_cost(state, keypoints['human'], 0.7)

def stage1_path_constraint1(traj, keypoints):
    """A human may shift; hold the margin the whole way past."""
    return clearance_cost(traj, keypoints['human'], 0.7)

def stage2_subgoal_constraint1(state, keypoints):
    """End at the star."""
    return progress_cost(state, keypoints['GOAL'], 0.3)

stop_points = [7, -1]
```

`stop_points` are the per-stage stopping positions (the numbered circles in the
annotated image); `-1` means "this stage ends at the goal." It mirrors the
original's `grasp_keypoints = [...]` summary.

> **That describes `--subgoals solver`, which is not the mode any published
> result used.** The 360-episode run (`rk720`) ran `--subgoals legs`, a per-leg
> PIVOT loop that decides the route differently. See [Modes](#modes) for what
> happens there. Do not read a number from this policy without knowing which
> mode produced it.

## Where it plugs in

This repository holds the planner only. The environment, controller, scorer and
verdict line all come from the [VoxPoser fork](https://github.com/OSMA-Bench/OSMA-Bench)'s
`run_tasks`, so results are scored on exactly the same terms as every other
policy:

```
run_tasks(..., external_planner=PivotPlannerHook(cfg))
        ^
   the single swap point
```

The hook is called **once per episode** (`max_plan_iter: 1`). PIVOT's
solve → execute briefly → re-observe loop therefore cannot be closed through the
simulator, so the same loop runs **on virtual state inside the hook**: "where I
am now" means "the last waypoint I emitted." Raising `max_plan_iter` was
rejected deliberately — it would change the execution path every policy shares
and break the comparison.

## Layout

```
run_pivot.py          entry point
config.yaml           solver budgets and switches
src/
  stops.py            proposes stopping candidates across the view and numbers them
  pivot.py            PIVOT: candidates on a shrinking ring, picked by number
  legs.py             per-leg re-query — the mode rk720 used
  costs.py            collision · consistency · path length · turning · infeasible · violation
  sdf.py              occupancy -> signed distance field
  interp.py           normalization · interpolation · angle wrap
adapters/
  planner_hook.py     implements the external_planner contract — the only contact point
lib/                  measurement code (radius pads · surface distance · AST validator)
prompts/              VLM query templates
tests_*.py            checks that run without the simulator
```

## Modes

`--subgoals` selects one of four, and they differ in **what decides the route** —
not in a detail. Always quote the mode alongside a result.

| Mode | Who fixes the endpoints | Segment geometry |
|---|---|---|
| `solver` (default) | the optimizer, from constraints as cost | optimizer |
| `stops` | the VLM, picking a number off the whole view | optimizer |
| `pivot` | the VLM, picking from a ring that shrinks each round | optimizer |
| `legs` | the VLM, **re-asked at every segment** | straight line if a gate passes, else optimizer |

### How PIVOT selection works

`pivot.py` follows the PIVOT paper's navigation appendix. For one segment:

1. Sample standable candidates on a ring around the robot, biased toward the
   goal. The first round uses a wide ring and many samples
   (`pivot_first_r: 2.0`, `pivot_first_samples: 64`); later rounds shrink it.
2. Drop candidates that the model's own path constraints reject — the check is
   run on the **straight line from here to that candidate**, not on the
   candidate point, so a spot that is fine to stand on but unreachable without
   crossing something is removed rather than offered.
3. Draw the survivors on the topview as numbered circles and ask the model which
   to walk to. The reply is parsed as `{"points": [...]}` and the numbers index
   back into the candidate array.
4. Repeat for `pivot_rounds` (3), narrowing to a single pick.

The model never sees coordinates, only numbers on a picture, so it cannot
compute the answer without looking. Written because whole-view candidate sets
included points heading away from the goal — one such pick turned a 4.78 m
straight shot into a 22.6 m route.

`legs` wraps this per segment and shows only the keypoints near that segment's
corridor, because a whole-scene view invites the model to constrain objects
irrelevant to where it is going next. In exchange it **discards the model's
subgoal constraints** — the endpoint is whatever number was picked, and the
model's constraints apply only to the way there (`legs.py:491`).

### What the `legs` run actually did (`rk720`, 360 episodes)

Worth recording, because it differs from what the top of this file promises.

| | |
|---|---|
| Stopping points picked by the VLM | 753 (from 2,207 candidate-image queries) |
| Segments walked as **straight lines**, no optimizer | 906 / 1,085 = **83.5%** |
| Episodes where the optimizer **never ran** | 217 / 360 = **60.3%** |
| Model-written subgoal constraints discarded | 1,939 |
| Solver iterations | **1 on all 179 legs**, zero backtracks |

Constraint optimization is a minority shareholder in this mode.
`max_iterations: 40` and `max_backtracks: 8` are configured and were never
exercised.

### The straight-line gate is miscalibrated

A segment goes straight if two conditions hold: the SDF clearance is sufficient,
and no path constraint the model wrote objects to that line (`legs.py:534`). The
second one is the problem.

`clearance_cost` does not return metres. It returns a normalized shortfall:

```
value = (eff − d) / eff        eff = requested_margin + object_radius_pad
veto when value > 0.10    ⇔    d < 0.9 × (requested_margin + radius_pad)
```

The pad comes from a hard-coded table in `constraints.py` (`crawling_baby 1.38`,
`human 0.52`, …). So `clearance_cost(traj, crawling_baby, 0.75)` **vetoes any
line passing within 1.92 m of the keypoint** — against a scored boundary of
0.6 m. The model's number and the benchmark's number were never the same
quantity.

Reconstructing the 111 rejected segments geometrically:

| | |
|---|---|
| Would have violated the scored boundary (veto justified) | 41 / 111 |
| **Would not have violated anything scored** | **70 / 111** |
| Vetoed only by objects the benchmark does not score | 22 |
| Already cleared the requested margin; vetoed by the radius pad alone | 39 / 102 |

Keypoints that triggered vetoes include `microwave` (6), `sink_main` (4),
`knife`, `main_door` — and three times `STOP1` / `STOP2`, meaning the model
wrote an avoidance constraint against the waypoint it was trying to reach.
Meanwhile only 12 of the 906 segments let through as straight lines actually
violated (1.3%). **The gate leaks little and over-rejects a lot** — roughly 1.7
safe lines discarded per unsafe one caught.

The fix is to compare against the tier boundary rather than against whatever
margin the model asked for.

## Other switch

| Switch | Off (default) | On |
|---|---|---|
| `--dynamics` | geometry only (the original has no v/a/J) | `speed_cost` · `accel_cost` · `jerk_cost` become available per stage |

Both switches off is closest to the original. Turning one on appends its
paragraph to the prompt and opens the matching vocabulary, so **the difference
stays attributable to that one switch**.

## Differences from the original

All of them are recorded in `DESIGN.md`. In brief:

- **Path length is divided by the straight-line distance.** PIVOT's weight of
  4.0 assumes a 0.55 m workspace; in a 6 m kitchen it overwhelms the collision
  term, and driving straight through a wall becomes the cheaper option.
- **An "infeasible" cost is added.** Keypoints are object centres, so the points
  for `sink` and `stove` sit inside the counter. An arm can reach over an
  object; a wheeled base cannot be there. This plays the role of PIVOT's IK
  unreachability.
- **Heading change is penalized.** A holonomic base separates yaw from direction
  of travel, so a turning cost does not catch spatial zigzag.
- **The DINOv2 proposal stage is not used.** That stage does not *find* objects;
  it splits an already-known mask into parts, which a navigation policy has no
  use for — the simulator already provides the object list and positions.
  Stopping candidates are drawn from traversable cells instead and picked by
  number.

  This README previously justified the omission with "185 candidates, only 5%
  worth avoiding, 56% on counters and cabinets." **That measurement cannot be
  located** — no run directory, script or log producing it exists in this
  repository, and `DESIGN.md` does not contain it despite a commit message
  saying it does. Treat the figure as unverified; the structural argument above
  stands on its own.

Every other weight keeps its original value. Changing them would make "the
structure was ported and the result differs" indistinguishable from "the weights
were tuned and the result differs."

## Running it

Place this at `policy/PIVOT/` inside a tree that has the VoxPoser fork and the
RoboCasa benchmark:

```bash
# default (solver) — the optimizer fixes the endpoints from constraints as cost
python3 run_pivot.py -m 'Qwen/Qwen3-VL-8B-Instruct' -p 8003 \
    -o outputs/run1 --layout-ids 0 --style-ids 3 \
    NavigateKitchenCatBlockingRouteA

# what rk720 actually ran — re-ask per segment, walk straight when the gate passes
python3 run_pivot.py -m 'Qwen/Qwen3-VL-8B-Instruct' -p 8003 \
    -o outputs/run2 --subgoals legs --straight-first \
    --layout-ids 0 --style-ids 3 \
    NavigateKitchenCatBlockingRouteA
```

The run's settings are saved to `pivot_config.json` in the output directory.
Check that file first when revisiting results — the command line is not recorded
anywhere else, so it is the only evidence of which mode produced them.

For the checks that need no simulator:

```bash
python3 tests_r1.py       # SDF sign · coordinate round-trip · cost terms
python3 tests_r2.py       # both solvers
python3 tests_r3.py       # loop · backtracking
python3 tests_dyn.py      # the dynamics switch
python3 tests_partial.py  # does one bad function still leave the rest usable
```

The tests read stored dumps; point `PIVOT_FIXTURES` at them on another machine.
