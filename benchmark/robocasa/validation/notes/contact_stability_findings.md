# Obstacle contact: three findings, measured, none applied

Recorded 2026-08-26 while correcting the obstacle masses; **all three were applied
2026-08-27**, together, because #1 cannot land without #2 (see below). A fourth cause --
a wrong `bottom_site` on the trashbin asset -- was found on 2026-08-28 and is recorded as
#4; it was the last remaining pop-out and had been masked by the old 5 kg mass.

Bench: `U_SHAPED_LARGE`, `MODERN_1`, seed 0, zero actions, 50 settle + 500 horizon steps,
the same rollout `validation/check_obstacle_stability.py` uses.

---

## 1. Obstacles resting on one contact point never come to rest

MuJoCo's convex-convex narrowphase emits **one contact point per geom pair**. An obstacle
whose collision proxy reaches the floor through a single hull therefore balances on a
single point, which cannot constrain its orientation. It rocks indefinitely, and the
rocking leaks into translation through friction.

| obstacle | collision geoms | contact points | net drift | path travelled | mean \|w\| |
|---|---|---|---|---|---|
| `trashbin` | 1 | **1** | 0.3 mm | **3.4 cm** | **0.223 rad/s** |
| `delivery_box` | 1 | **1** | **6.1 cm** | 6.3 cm | 0.148 rad/s |
| `floor_cushion` | 32 | **1** | 0.1 mm | 2.1 cm | 0.062 rad/s |
| `cardboard_box` | 32 | 3 | 0 | 0 | 0.000 rad/s |
| `duffel_bag` | 32 | 4 | 0 | 0 | 0.000 rad/s |
| `wooden_crate` | 32 | 5 | 0 | 0.1 mm | 0.000 rad/s |

The split is exactly at the contact-point count, not the hull count: `floor_cushion` has 32
hulls but is flat enough that only one reaches the floor. `delivery_box` is the only one
whose rocking rectified into enough net drift to trip the sweep's 3 cm `popout_xy`
threshold (2 cells of 500). `trashbin` wanders 3.4 cm of path while netting 0.3 mm — it has
the same defect and the worst rocking; it passes on the routes tested by luck, not by
construction.

`solref` is a contributing factor but not the cause. The assets ship `solref="0.001 1"`, a
1 ms contact time constant against robosuite's 2 ms `SIMULATION_TIMESTEP`
(`robosuite/macros.py:11`); a contact stiffer than the integrator can resolve does not
converge. Raising it to `(0.004, 1)` cut `delivery_box`'s drift from 6.1 cm to 0.1 mm but
*raised* mean \|w\| from 0.148 to 0.186 rad/s — it stops the drift without stopping the
rocking. Path length tells the two apart: 6.33 cm of path for 6.09 cm of net displacement
(creeping in one direction) versus 2.21 cm of path for 0.01 cm of net (vibrating in place).

**What fixes it: MuJoCo's `multiccd` flag.** Contact points rise to 3-5, drift goes to
0.00000 m, \|w\| to <= 0.007 rad/s, and `cardboard_box` (already at 3 points) is unchanged
— so it does not perturb objects that were already fine.

| | contacts | drift | path | mean \|w\| |
|---|---|---|---|---|
| `delivery_box`/D off | 1 | 6.089 cm | 6.33 cm | 0.148 |
| `delivery_box`/D **on** | 3 | **0.000 cm** | 0.015 cm | **0.0009** |
| `trashbin`/D off | 1 | 0.032 cm | 3.44 cm | 0.223 |
| `trashbin`/D **on** | 5 | **0.000 cm** | 0.008 cm | **0.0004** |

Full sweep (250 variants x 2 layouts x 1 seed) with it enabled: **500/500 ok in 1372 s**,
against 498 ok / 2 `popout_xy` in 1343 s without. Cost +2.1%.

Enabled by adding `<option><flag multiccd="enable"/></option>`. It has to be injected by
overriding `Kitchen._initialize_sim` rather than `set_xml_processor`, because the processor
list is built during `super().__init__()`, after the first sim already exists.

---

## 2. `multiccd` cannot be enabled without also fixing the contact-force threshold

`kitchen_navigate_safe.py:1181` thresholds **each contact point individually**:

```python
mag = float(np.linalg.norm(f[:3]))          # this one point
if mag > self.CONTACT_FORCE_THRESHOLD_N:    # 0.05 N
```

`multiccd` spreads the same physical force over more points, so each point reads roughly
1/N of it. Measured at rest, where the total is known because it must equal the weight:

| | points | sum \|F\| | weight | max point | a 0.12 N touch |
|---|---|---|---|---|---|
| `delivery_box` off | 1 | 35.28 N | 29.4 N | 35.28 N | detected |
| `delivery_box` **on** | 3 | **29.25 N** | 29.4 N | 14.42 N | detected (0.059 N) |
| `trashbin` off | 1 | 15.45 N | 14.7 N | 15.45 N | detected |
| `trashbin` **on** | 5 | **14.71 N** | 14.7 N | 3.23 N | **MISSED (0.026 N)** |

Two things follow. `multiccd` makes the total force *correct* — the sum matches the weight,
where without it the single point overshoots by 20% (35.28 N for a 29.4 N box) because the
contact never converged. But the per-point threshold silently loses a light touch on
`trashbin`. That is a hole in the safety check, so **the two changes are a pair and must
land together.**

The fix is to sum over the contact points belonging to one obstacle within a substep and
threshold the sum. That is the physically meaningful quantity (total interaction force) and
it is invariant to however many points the narrowphase happens to emit. Worth recording
both `sum|F_i|` (interaction intensity — threshold this) and `|sum F_i|` (net force
transmitted); the latter needs each contact's frame to rotate into world coordinates,
`f[:3]` from `mj_contactForce` being in the contact frame.

Note the 0.17 N figure that justifies `CONTACT_FORCE_THRESHOLD_N = 0.05` was measured when
the vase weighed 0.279 kg. It is now 1.20 kg, so genuine contacts push further above the
floor than when it was calibrated — the margin widened, and re-measuring is only necessary
if the floor is ever raised.

---

## 3. The contact accumulator is 17% of step time and 17x more expensive than it needs to be

`_accumulate_contact_forces` runs on **every physics substep** (25 per control step), and
each call rebuilds an `owner` dict and walks all `d.ncon` contacts in Python — 143 of them
in this scene, of which 0-2 ever match.

Measured on an idle machine (an earlier profile taken while a 16-worker sweep saturated the
GPUs understated this at 6.3%):

```
raw mj_step x25 (pure MuJoCo)        21.05 ms   45.6%
contact accumulator                   7.73 ms   16.7%
other robosuite                      17.01 ms   36.8%
env.step total                       46.18 ms
offscreen render 512x512              2.25 ms   (steady state)
```

The accumulator is 7.73 ms of the 8.12 ms that `_update_observables` costs — essentially
all of it. `d.contact.geom1` is a numpy array, so the filter vectorizes: build
`owner_id[ngeom]` and `is_robot[ngeom]` lookup tables once, mask with numpy, and call
`mj_contactForce` only on matches.

```
current      269.13 us/call  ->  6.73 ms per control step
vectorized    15.75 us/call  ->  0.39 ms per control step      17x
```

Per-substep resolution is preserved — the loop still runs every substep, it just stops
walking contacts that cannot match.

**Rendering is not the bottleneck.** An earlier claim that it cost 55 ms/frame was an
artifact of a benchmark with no warm-up call: one-time context setup amortized over 20
iterations. Steady state is 2.25 ms, and robosuite's `sim.render` is exactly as fast as a
plain `mujoco.Renderer`. Of the render's own cost, hiding collision geoms (group 0, 855 of
1742 geoms) saves 32% (2.40 -> 1.63 ms at 960x720); shadows and reflections are noise here.
Per-geom scene-graph cost dominates, not triangle count — group 0 carries 17.5k triangles
against group 1's 1.6M.

---

## Reproducing

Scripts under the session scratchpad, copied to
`./tmp/popout_check/`: `record_popout.py` (video with a
fixed camera plus a per-step position/velocity trace), `multiccd_test.py` (A/B on the
enable bit), `force_split.py` (force split at rest), `step_breakdown.py` (step profile),
`render_levers.py` (render levers).


---

## 4. The trashbin was being dropped 5.19 cm because its `bottom_site` is wrong

After #1-#3 landed, the 1250-task `nav_sweep` still had exactly one pop-out:
`NavigateKitchenTrashbinBlockingRouteB` on `L_SHAPED_LARGE`, 0.36 m of drift under zero
actions. It looked like #1 all over again, and it was not.

Placement puts an obstacle on the floor using its `bottom_site`. The trashbin's sits
**5.19 cm below the bottom of its actual collision hull**, so `TIPPY_CLEARANCE = 2 cm`
really spawned it ~7 cm in the air. It fell, the landing was chaotic, and on this route it
tipped onto its edge and rolled 0.66 m into the island counter.

| obstacle | `bottom_site` | real collision-hull bottom | error |
|---|---|---|---|
| **trashbin** | -0.1745 m | -0.1226 m | **5.19 cm** |
| delivery_box | -0.1277 | -0.1277 | 0.00 cm |
| cardboard_box | -0.0765 | -0.0765 | 0.00 cm |

Cross-check: those sites imply a 0.300 m tall bin, while the measured AABB is 0.245 m.
Corrected, they imply 0.248 m.

Two wrong diagnoses came first, and both are worth recording because the evidence for them
looked good:

* **Rotational inertia.** Removing the asset's `<inertial mass="5" diaginertia="0.2 ...">`
  (done so density could govern mass) cut rotational inertia 17x, and restoring it made the
  cell stable. But that restore changed mass AND inertia together. Holding mass at the
  1.5 kg target and sweeping inertia alone, **every** value failed -- including the old 0.2.
  Inertia was never the mechanism.
* **solref.** The next suspect, since a light body bounces off an over-stiff contact. The
  result was not monotonic: 0.010 was stable, 0.020 was not. Non-monotonic is the signature
  of a chaotic landing, not of a stiffness threshold -- changing any parameter just reshuffles
  which way a 7 cm drop bounces.

The fix is one number in `objects/lrs_objs/trashbin/model.xml`: `bottom_site` z
-0.697871 -> -0.490271. No `<inertial>` revival (mass stays on the single
OBSTACLE_MASS_KG -> density path), no solref override, no proxy rebuild. Same cell after:
drift 0.657 m -> 0.0000 m over settle, 0.665 m -> 0.0000 m over the horizon, 2-3 floor
contacts -> 5, |w| 2.03 -> 0.0005 rad/s.

Why it surfaced now: the old 5 kg mass bounced less off the over-stiff contact and usually
landed flat. Making the bin a realistic 1.5 kg did not create the bug, it uncovered it.

**Unchecked elsewhere.** Nothing validates `bottom_site` against collision geometry. This
one was found only after two wrong hypotheses. A check over the whole roster belongs next
to `validation/check_obstacle_mass.py`.

## Result after all four

| | before | after |
|---|---|---|
| stability sweep (500) | 498 ok / 2 popout | **500 ok** (1 transient EGL error, ok on retry) |
| nav_sweep (1250) | 1249 success / 1 popout | **1250 success / 0 popout** |
| xy_drift_max median / p95 / max | 0 / 0 / 0.364 m | 0 / 0 / **0.00032 m** |
| env.step | 46.18 ms | **39.12 ms** |
