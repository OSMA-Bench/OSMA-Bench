# OSMA-Bench (anonymized for review)

Code for **OSMA-Bench: Benchmarking Obstacle-Conditioned Safety in Household
Robot Navigation**.

This repository is an **anonymized snapshot** prepared for double-blind peer
review. All git history, submodule pointers, and identifying metadata have been
stripped. The upstream codebases used by this work (`robosuite`, `robocasa`,
`voxposer`) have been inlined as plain directories, as are the
constraint-optimization planner (`policy/PIVOT`) and the vision-language-action
(VLA) policy runner (`policy/VLA`).

## Layout

```
benchmark/
  robosuite/   simulation core (inlined; large MuJoCo asset dirs omitted)
  robocasa/    kitchen environment suite (inlined; asset dirs omitted)
policy/
  Voxposer/    costmap + A* baseline policy (inlined)
  PIVOT/       constraint-optimization planner (inlined)
  VLA/         VLA checkpoint panel runner (inlined)
docker/        runtime container definitions
docs/          getting-started and design notes
scripts/       experiment & report generation
```

## Running

See `docs/` for environment setup. The `robosuite/` simulation core is retained
in full, including its committed MuJoCo model assets. For `robocasa/`, all source
and committed assets are retained (so `import robocasa` works out of the box),
but the large downloadable kitchen assets are **not** bundled — fetch them with
`python3 benchmark/robocasa/robocasa/scripts/download_kitchen_assets.py` and run
`setup_macros.py` to reconstitute the full environment. The remainder of the code
is self-contained.

## License

Each inlined upstream component retains its original license; see the
respective subdirectory's `LICENSE` file.
