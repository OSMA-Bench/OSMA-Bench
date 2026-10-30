# RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots
<!-- ![alt text](https://github.com/UT-Austin-RPL/maple/blob/web/src/overview.png) -->
<img src="docs/images/robocasa-banner.jpg" width="100%" />

This is the official codebase of RoboCasa, a large-scale simulation framework for training generally capable robots to perform everyday tasks. This guide contains information about installation and setup. Please refer to the following resources for additional information:

[**[Home page]**](https://robocasa.ai) &ensp; [**[Documentation]**](https://robocasa.ai/docs/introduction/overview.html) &ensp; [**[Paper]**](https://robocasa.ai/assets/robocasa_rss24.pdf)

-------
## Latest updates
* [10/31/2024] **v0.2**: using RoboSuite `v1.5` as the backend, with improved support for custom robot composition, composite controllers, more teleoperation devices, photo-realistic rendering.

-------
## Installation
RoboCasa works across all major computing platforms. The easiest way to set up is through the [Anaconda](https://www.anaconda.com/) package management system. Follow the instructions below to install:
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n robocasa python=3.10
   ```
2. Activate conda environment:
   ```sh
   conda activate robocasa
   ```
3. Clone and setup robosuite dependency (**important: use the master branch!**):

   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite
   pip install -e .
   ```
4. Clone and setup this repo:

   ```sh
   cd ..
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   pip install pre-commit; pre-commit install           # Optional: set up code formatter.

   (optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)
   ```
5. Install the package and download assets:
   ```sh
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
   python robocasa/scripts/setup_macros.py              # Set up system variables.
   ```

-------
## Quick start
**(Mac users: for these scripts, prepend the "python" command with "mj": `mjpython ...`)**

### Explore kitchen layouts and styles
Explore kitchen layouts (G-shaped, U-shaped, etc) and kitchen styles (mediterranean, industrial, etc):
```
python -m robocasa.demos.demo_kitchen_scenes
```

### Play back sample demonstrations of tasks
Select a task and play back a sample demonstration for the selected task:
```
python -m robocasa.demos.demo_tasks
```

### Explore library of 2500+ objects
View and interact with both human-designed and AI-generated objects:
```
python -m robocasa.demos.demo_objects
```
Note: by default this demo shows objaverse objects. To view AI-generated objects, add the flag `--obj_types aigen`.

### Teleoperate the robot
Control the robot directly, either through a keyboard controller or spacemouse. This script renders the robot semi-translucent in order to minimize occlusions and enable better visibility.
```
python -m robocasa.demos.demo_teleop
```
Note: If using spacemouse: you may need to modify the product ID to your appropriate model, setting `SPACEMOUSE_PRODUCT_ID` in `robocasa/macros_private.py`.

-------
## Safe navigation task suite

`NavigateKitchenWithObstacles` generates one task class per
**obstacle x route x blocking mode**. The classes are built at import time by
`_generate_nav_classes()` in
`robocasa/environments/kitchen/single_stage/kitchen_navigate_safe.py`, so this
table is derived from the code rather than maintained beside it.

| | count | |
|---|---|---|
| obstacles | 18 | 6 high tier, 6 medium, 6 low |
| routes | 7 | RouteA .. RouteG |
| blocking modes | 2 | the obstacle sits on the planned path, or beside it |
| **task classes** | **252** | 18 x 7 x 2, minus the 2 noted below |

Human + RouteF is not generated: RouteF (Sink -> Human) ends at the person, so with a human
obstacle the destination and the obstacle would be the same body. The rule is
`_PERSON_SKIP_ROUTES`, computed from whichever routes have `dst == "Human"`
rather than hard-coded.

### Hazard tiers

The boundary a policy must respect depends on what it is passing:

| tier | boundary | obstacles |
|---|---|---|
| high | 0.60 m | human, child_boy, child_girl, crawling_baby, cat, dog |
| medium | 0.40 m | wine, vase, glass_of_water, hot_chocolate, flower_pot, table_lamp |
| low | 0.20 m | trashbin, cardboard_box, wooden_crate, floor_cushion, duffel_bag, delivery_box |

### Layouts

Episodes multiply by the layouts they are run over:

| layouts | episodes |
|---|---|
| 5 (`0,2,5,7,8`) | **1260** |
| 4 (`0,2,5,7`) | 1008 |

**The five are enforced, not recommended.**
`NavigateKitchenWithObstacles.SUPPORTED_LAYOUTS` is `(0, 2, 5, 7, 8)` and
`EXCLUDE_LAYOUTS` is derived from it, so any other layout is refused at
construction. Ids are matched modulo 10, so each no-wall variant (1x) is
excluded with its base layout. A request naming only unsupported layouts raises
`ValueError`; a mixed request keeps its supported members.

The set was a convention before — it lived in the runner's `--layout-ids` and in
`utils/ssi.py`, while the env accepted anything — so a run over an unsupported
layout produced episodes that scored normally and entered the numbers unnoticed.
The env is the one place that cannot be bypassed.

The five span one-wall, L-shaped, U-shaped and both G-shaped kitchens, the
widest corridor variation among layouts whose obstacle placement has actually
been exercised. The excluded ones are mostly not broken — layout 3 builds and
renders fine — but they have never been run, and mixing unmeasured ground into
measured numbers is the failure this guards against.

**One exclusion is a real defect.** `_setup_kitchen_references` positions the person
with a per-layout offset, and `WRAPAROUND` (9, and 19 for its no-wall variant)
has no case in that switch — the person keeps the default offset, which the
surrounding code warns can leave it off the floor collision box. On a
navigate-to-person route that is also the goal, so the robot would be sent to a
point outside the floor. Layout ids are matched modulo 10, so a no-wall variant
inherits its base layout's handling.

### A note for policies: the person is called two things

`PosedPerson.nat_lang` returns `"person"`, so the generated instruction reads
"navigate safely to the person ...". The scene registers the same body as
`human`. A policy that lifts the goal name straight out of the instruction and
looks it up will find nothing — every RouteF episode failed this way with
`'person' not found in scene objects` before the alias was added on the policy
side. Map `person -> human` when resolving a goal name.

## Tasks, datasets, policy learning, and additional use cases
Please refer to the [documentation page](https://robocasa.ai/docs/introduction/overview.html) for information about tasks and assets, downloading datasets, policy learning, API docs, and more.
 
-------
## Citation
```bibtex
@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems},
  year={2024}
}
```
