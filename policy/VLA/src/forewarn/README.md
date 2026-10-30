# FOREWARN steering

FOREWARN-style policy steering (Wu et al., [arXiv 2502.01828](https://arxiv.org/abs/2502.01828)) for the VLA rollout runner.
Before the robot moves, several futures of the policy are played out in the simulator. A VLM looks at each one
from above and picks one, and the robot executes the picked future. Works with any `--model-profile`.

## Files

| File | Role |
|---|---|
| `src/forewarn/steer.py` | Decision loop, config loading, per-chunk noise (`chunk_noise`), `steering.jsonl` |
| `src/forewarn/snapshot.py` | Exact save/restore of the robosuite/robocasa env, and a digest to compare two states |
| `src/forewarn/verifier.py` | Topview filmstrips, VLM narration, and the selection prompt (paper Fig. 9) |
| `src/configs/steering/forewarn.yaml` | Number of candidates, lookahead, noise sampling, verifier endpoint |
| `examples/run_forewarn.sh` | One steered rollout in the simulator container |
| `tests/test_forewarn.py` | Config, answer parsing and snapshot tests; no policy server or VLM needed |

Hooks outside this folder:
- `src/run_vla_rollout.py --steering-config <yaml>` turns steering on.
- `src/policy_client.py` sends `policy_seed` with `/reset`.
- `src/adapters/pi05_opentau.py` applies `noise_scale` in `/act`.

## One decision

1. **Save the env state.** `snapshot.capture_env` copies the MuJoCo state, the controllers, the observation
   history and every RNG the env holds.
2. **Play out `num_candidates` (16) futures.** For each one, restore the saved state and run up to
   `lookahead_chunks` (5) chunks closed-loop in the env. A chunk is one policy call; its first `replan_steps`
   actions run before the next call (20 for `pi05_opentau`, so a future is 100 env steps and a 500-step episode
   holds 5 decisions). A future stops early if the env reports success.
   Each future keeps its base path, a topview frame at the start and after every chunk, whether it touched an
   obstacle, its smallest obstacle distance, and the digest of the state it ended in.
3. **Restore the state again**, and check the restore reproduced the start digest — the play-outs must leave
   nothing behind.
4. **The VLM picks one.** Each future's frames become one filmstrip, and the 16 are shuffled into Modes, so a
   Mode number says nothing about candidate order. `verifier.select_with` decides what the picking call gets:
   the 16 filmstrips (`images`), a sentence the VLM first writes about each one (`descriptions`), or both.
   It answers with one Mode; if it answers nothing parseable, candidate 0 runs (`fallback`).
5. **Replay the chosen future.** The runner executes the chunks that candidate already recorded, so the policy is
   not queried again. Replay is deterministic, so the next decision checks the env reached the state that
   candidate ended in (`commit_check`); a mismatch is logged, not raised.

## What the VLM sees

One topview camera, the one named `topview` in the scene (`CAMERA_NAME`, `TopviewCamera.from_env`). The robot's own
cameras go to the policy server only, never to the verifier.

The view is cropped once per decision, to the same crop for every candidate, so the 16 filmstrips are comparable:
a box around the start and the goal, tightened to what the paths, the goal and the obstacles actually cover.

A filmstrip is that candidate's frames in time order, 4 per row, each labelled with its time from the decision.
On every frame the whole future's path is drawn faint, the part already driven is drawn solid, and a dot marks
where the base is in that frame.

**What the picking call gets** is `verifier.select_with`:

| | The call sees | The answer | VLM calls per decision |
|---|---|---|---|
| `descriptions` | the 16 sentences, no image (Fig. 9 verbatim) | repeats the chosen sentence | 17 |
| `images` | the 16 filmstrips, redrawn smaller, each with `Mode N` across the top | a JSON object naming that Mode | 1 |
| `both` | the sentences and the filmstrips | a JSON object naming that Mode | 17 |

`images` asks for no narration, so nothing is written or logged about a candidate that decision; it also leaves
no sentence to repeat back, so a misread Mode number has nothing to correct it. `both` still writes the sentences
and lists them in the prompt; only `descriptions` reads the answer back against them (the `description` rule).

Asked for the answer the way Fig. 9 phrases it, a model shown 16 images narrates all of them before concluding.
Measured on one decision with Qwen3.5-9B, that answer claimed a collision for fifteen modes when the run had
recorded none, and chose the candidate that passed the obstacle closest of the sixteen. So the images prompt asks
for a JSON object instead — the mode, then one sentence of reason — and the request pins it to the schema
`verifier.mode_choice_format`, which the server decodes against. Nothing off the schema can come back, so the
answer is read with `json.loads` and no rule has to decide what a sentence meant.

The schema's values are the words the prompt uses, `Mode 1` … `Mode 16` and `none of them`, and that matters more
than it looks. A grammar removes every token off the schema, so an enum of bare numbers meets a model that was
about to write `"mode": "Mode 1"` with no legal `M` left, and the decoder takes the only other branch: on one
decision that was `none of them`, all sixteen rejected, and an enum of integers did the same. With `Mode N` values
the schema changed no decision at all — over the 25 decisions that have stored filmstrips, the same prompt with
the schema and without it picked the same candidate 25 times out of 25. Dropping `reason` from the schema also
moved the answer, so it stays. What the schema costs: about twice the completion tokens of a two-line answer
(45–65 against 22–29, 1.7 s a call), and roughly one second of grammar compilation on a server's first request.

Fig. 9's own prose answer (`descriptions`) is left unpinned, since a schema over it would stop being Fig. 9. It
keeps the rules: the last bare `Mode N` line (`mode_line`), before position is used at all, and an answer the token
budget cut off is not read as a choice when it names more than one mode (`..._truncated`, candidate 0 runs), since
a model working through the modes names the ones it rejects first. A cut-off JSON answer needs no such rule — it
does not parse, and candidate 0 runs. Whatever the setting, the prompt gets the task and `verifier.task_condition`,
which must not name the obstacle — the environment instruction stays the only task information.

One request carries all `num_candidates` images, so the VLM server has to accept that many in one message.
vLLM's own default is generous (999 per modality in 0.19.1), but a server launched with an explicit
`--limit-mm-per-prompt` below that answers HTTP 400 `At most N image(s) may be provided in one prompt` and the
decision falls back to candidate 0. Serve the verifier with `--limit-mm-per-prompt '{"image": 16}'` — at or above
`num_candidates` — so the run does not depend on how the server happened to be launched. Context is not the
constraint: 16 of these strips measured 4600 prompt tokens, well inside a 32768-token window.

The same server has to accept `response_format` with a `json_schema`, which vLLM has (measured on 0.19.1). One
that ignores it answers off the schema, which does not parse, and every decision falls back to candidate 0.

Alongside the filmstrips, one `dNN_paths.png` per decision puts all 16 paths, the obstacles and the goal on a
single image with the outcome of each. That one is for people; it is never sent to the VLM.

## What differs between candidates

Only the noise each policy call starts from. Every candidate starts from the same env state (including its RNGs),
the same observation history, the same instruction and the same weights.

**The noise.** The policies here generate actions by flow matching. For `pi05_opentau`, one policy call starts from a
20 × 32 table of numbers, one row per action and one column per action dimension. Each number is drawn independently from
a standard normal, N(0, 1): mean 0 and standard deviation 1.
The numbers are not limited to [0, 1]. About 68% fall in [-1, 1] and 95% in [-2, 2].

The table is the starting point of generation, not something added to the output.
OpenTau sets `x = noise`, then does `x += -0.1 * v` 10 times, where `v` is the model's output for the observation,
`x` and the current time. The final `x` is the chunk (`sample_actions` in OpenTau's `modeling_pi05.py`).
`forewarn.yaml` lists where each of the other models draws its noise.

Each draw is fixed by `policy_seed = SeedSequence([sampling.seed, decision_step, candidate, draw])`, which `/reset`
seeds the policy's RNG with. The same seed gives the same table; a new `sampling.seed` redraws every table.
With the shipped `sampling` block, `draw` is just the chunk index, so each of a candidate's 5 chunks starts from its
own N(0, 1) table and the 16 candidates differ by `candidate` alone.

**`shared_noise`.** `chunks: [first, last]` makes those chunks all reuse draw `first` instead of their own, and
multiplies that table by `noise_scale` before the 10 steps. They still output different actions, since each sees the
observation the previous chunk left. `[0, 0]` with `1.0` shares nothing and scales nothing, which is what ships.
`load_config` requires `0 <= first <= last < lookahead_chunks`.

Only `pi05_opentau` reads `noise_scale`. Other models must leave it at 1.0, or the steerer refuses to start.

## Run

Start the policy server (`src/vla_server.py`) and the OpenAI-compatible VLM at `verifier.base_url` first.

```bash
CONTAINER=robotics-safety-forewarn-g3 GPU=0 PORT=8105 PROFILE=pi05_opentau TASK=NavigateKitchenDogBlockingRouteA \
  examples/run_forewarn.sh
python -m unittest tests.test_forewarn
```

## Output

Everything is written to the run directory.
`meta.json` there also records the steering config, its sha256, and a summary (decisions, fallbacks, commit checks).

| Path | Content |
|---|---|
| `steering.jsonl` | One line per decision: the commit check; for each candidate, every chunk's `noise_draw` / `policy_seed` / `noise_scale`, plus its outcome, path and filmstrip; the VLM's descriptions and answer; the executed candidate |
| `steering/dNN_candKK.png` | Filmstrip of candidate KK at decision NN, as sent to the narration call; not written with `images` |
| `steering/dNN_modeMM.png` | The same strip redrawn smaller and labelled Mode MM, as sent to the picking call; not written with `descriptions` |
| `steering/dNN_paths.png` | All candidate paths on one image, for people |
