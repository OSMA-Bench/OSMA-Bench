"""FOREWARN steering pieces that need neither a policy server nor a VLM."""

from __future__ import annotations

import collections
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from forewarn import snapshot, verifier  # noqa: E402
from forewarn.steer import Steerer, load_config, chunk_noise  # noqa: E402

try:
    import mujoco
except ImportError:  # pragma: no cover - the simulator container has MuJoCo
    mujoco = None

CONFIG = SRC / "configs" / "steering" / "forewarn.yaml"


class ConfigTest(unittest.TestCase):
    def test_shipped_config_loads(self):
        config = load_config(CONFIG)
        # 5 chunks x 20 replan_steps for pi05_opentau = a decision every 100 env steps.
        self.assertEqual(config["lookahead_chunks"], 5)
        self.assertEqual(config["sampling"], {"seed": 0, "shared_noise": {"chunks": [0, 0], "noise_scale": 1.0}})

    def test_select_with_must_be_one_of_the_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(CONFIG.read_text().replace("select_with: images", "select_with: filmstrips"))
            with self.assertRaisesRegex(ValueError, "select_with"):
                load_config(path)

    def test_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(CONFIG.read_text().replace("shared_noise:", "held_noise:"))
            with self.assertRaisesRegex(ValueError, "sampling keys"):
                load_config(path)

    def test_shared_chunks_must_fit_the_lookahead(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(CONFIG.read_text().replace("chunks: [0, 0]", "chunks: [0, 5]"))
            with self.assertRaisesRegex(ValueError, "lookahead_chunks"):
                load_config(path)

    def test_the_shipped_sampling_table(self):
        """The chunk i / eps / x noise_scale table in forewarn.yaml, row by row."""
        sampling = load_config(CONFIG)["sampling"]
        noise = [chunk_noise(sampling, 0, 2, index) for index in range(5)]
        # chunks [0, 0] shares nothing, so every chunk is its own draw at the policy's own scale.
        self.assertEqual([n["noise_draw"] for n in noise], [0, 1, 2, 3, 4])
        self.assertEqual([n["noise_scale"] for n in noise], [1.0] * 5)
        # One policy_seed per draw. SeedSequence zero-pads keys, so the key must keep its chunk entry.
        self.assertEqual(len({n["policy_seed"] for n in noise}), 5)
        self.assertNotEqual(chunk_noise(sampling, 0, 2, 4)["policy_seed"], chunk_noise(sampling, 0, 3, 4)["policy_seed"])
        self.assertNotEqual(chunk_noise(sampling, 0, 2, 0)["policy_seed"], chunk_noise(sampling, 20, 2, 0)["policy_seed"])

    def test_noise_scale_needs_a_server_that_reads_it(self):
        kwargs = dict(
            client=None, instruction="", model_profile="groot_n1_5", query=None, step=None, observe=None,
            replan_steps=16, goal_xy=[0, 0], control_freq=20, stop_on_env_success=False, run_dir=Path("."),
        )
        # The shipped config ships scale 1.0, which every server can serve; only a scale above it needs /act support.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scaled.yaml"
            path.write_text(CONFIG.read_text().replace("noise_scale: 1.0", "noise_scale: 3.0"))
            with self.assertRaisesRegex(ValueError, "act_sampling_keys"):
                Steerer(None, load_config(path), server_meta={}, **kwargs)


class ParseSelectionTest(unittest.TestCase):
    DESCRIPTIONS = ["The robot drives left of the table.", "The robot drives right of the table."]

    def test_repeated_description_wins_over_a_mode_number(self):
        answer = "Mode 1. The robot drives right of the table."
        self.assertEqual(verifier.parse_selection(answer, self.DESCRIPTIONS)["mode"], 2)

    def test_leading_mode_none_and_out_of_range(self):
        self.assertEqual(verifier.parse_selection("The chosen mode is 2.", self.DESCRIPTIONS)["mode"], 2)
        self.assertEqual(verifier.parse_selection("None of them.", self.DESCRIPTIONS)["status"], "none")
        self.assertEqual(verifier.parse_selection("mode 7", self.DESCRIPTIONS)["status"], "unparsed")

    def test_a_walkthrough_is_read_as_its_last_bare_mode_line(self):
        """Qwen3.6-35B wrote this when shown 4 images; the modes it rejects are mentioned first."""
        answer = (
            "Based on the task description, the robot must avoid obstacles.\n"
            "- **Mode 1**: The robot collides with the obstacle at +3.0s.\n"
            "- **Mode 2**: The robot reaches the coffee machine without any collisions.\n"
            "\n"
            "Mode 2 is the only mode that fulfills the task goal.\n"
            "\n"
            "Mode 2"
        )
        parsed = verifier.parse_selection(answer, self.DESCRIPTIONS)
        self.assertEqual((parsed["mode"], parsed["rule"]), (2, "mode_line"))


class ModeChoiceTest(unittest.TestCase):
    """The answer the images calls ask for: a JSON object the server decodes against a schema."""

    def test_the_schema_lists_every_mode_in_the_prompt_s_own_words(self):
        """Bare numbers are off the model's vocabulary and were measured changing the choice."""
        schema = verifier.mode_choice_format(3)["json_schema"]
        self.assertEqual(schema["schema"]["properties"]["mode"]["enum"],
                         ["Mode 1", "Mode 2", "Mode 3", "none of them"])
        self.assertEqual(schema["schema"]["required"], ["mode", "reason"])
        self.assertTrue(schema["strict"])

    def test_a_mode_and_a_rejection_are_read_off_the_object(self):
        chosen = verifier.parse_mode_choice('{"mode": "Mode 2", "reason": "it stays clear."}', 3)
        self.assertEqual((chosen["status"], chosen["mode"], chosen["rule"]), ("selected", 2, "json_mode"))
        none = verifier.parse_mode_choice('{"mode": "none of them", "reason": "all collide."}', 3)
        self.assertEqual((none["status"], none["mode"]), ("none", None))

    def test_an_answer_that_is_not_the_object_is_unparsed(self):
        """Only reachable off-schema: a server that ignored it, or a reply cut off mid-object."""
        for text in ['{"mode": "Mode 2", "reason": "it stays cl', "Mode 2, it stays clear.", '{"reason": "x"}']:
            self.assertEqual(verifier.parse_mode_choice(text, 3)["rule"], "json_missing")
        for text in ('{"mode": "Mode 7", "reason": "x"}', '{"mode": "2", "reason": "x"}'):
            self.assertEqual(verifier.parse_mode_choice(text, 3)["rule"], "json_out_of_range")


class SelectionPromptTest(unittest.TestCase):
    DESCRIPTIONS = ["drives left", "drives right"]

    def test_descriptions_are_listed_or_left_to_the_images(self):
        written = verifier.build_selection_prompt("go", "carefully", self.DESCRIPTIONS, "descriptions")
        self.assertIn("`Mode 2. drives right`", written)
        self.assertNotIn("top-left corner", written)
        shown = verifier.build_selection_prompt("go", "carefully", self.DESCRIPTIONS, "images")
        self.assertIn("`Mode 2`", shown)
        self.assertNotIn("drives right", shown)
        self.assertIn("written as `Mode N`, using the number on its image", shown)
        both = verifier.build_selection_prompt("go", "carefully", self.DESCRIPTIONS, "both")
        self.assertIn("`Mode 2. drives right`", both)
        self.assertIn("using the number on its image", both)


class SelectCallsTest(unittest.TestCase):
    """What select() asks the server for, with chat() replaced by a recorder."""

    CONFIG = {
        "task_condition": "carefully", "max_parallel": 5, "narration_max_tokens": 200,
        "selection_max_tokens": 300, "select_with": "images",
    }

    def run_select(self, select_with, answer, finish_reason="stop"):
        calls = []

        def fake_chat(config, text, pngs, max_tokens, response_format=None):
            calls.append({"images": len(pngs), "text": text, "format": response_format})
            picking = len(calls) > (0 if select_with == "images" else 2)
            return {"text": answer if picking else "drives on",
                    "finish_reason": finish_reason if picking else "stop"}

        config = dict(self.CONFIG, select_with=select_with)
        pngs = [b"a", b"b"]
        original, verifier.chat = verifier.chat, fake_chat
        try:
            chosen, detail = verifier.select(config, 0, "go", pngs, [["now"], ["now"]], [1, 0], pngs)
        finally:
            verifier.chat = original
        return chosen, detail, calls

    ANSWER = '{"mode": "Mode 2", "reason": "it stays clear."}'

    def test_images_makes_one_call_and_skips_narration(self):
        chosen, detail, calls = self.run_select("images", self.ANSWER)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["images"], 2)  # every candidate, in mode order
        self.assertEqual(detail["narrations"], [])
        self.assertEqual((detail["status"], chosen), ("selected", 0))  # order[1] = candidate 0

    def test_the_images_calls_ask_for_the_mode_choice_schema(self):
        for select_with in ("images", "both"):
            _, _, calls = self.run_select(select_with, self.ANSWER)
            schema = calls[-1]["format"]["json_schema"]
            self.assertEqual(schema["name"], "mode_choice")
            self.assertEqual(schema["schema"]["properties"]["mode"]["enum"],
                             ["Mode 1", "Mode 2", "none of them"])

    def test_a_truncated_answer_is_not_read_as_a_selection(self):
        """The budget cut the object off before its closing brace, so there is nothing to read."""
        chosen, detail, _ = self.run_select("images", '{"mode": "Mode 2", "reason": "it stays cl', "length")
        self.assertEqual(detail["parse"]["rule"], "json_missing")
        self.assertEqual((detail["status"], chosen), ("unparsed", None))

    def test_descriptions_narrates_every_candidate_sends_no_image_and_no_schema(self):
        chosen, detail, calls = self.run_select("descriptions", "Mode 1. drives on")
        self.assertEqual([call["images"] for call in calls], [1, 1, 0])
        self.assertEqual([call["format"] for call in calls], [None, None, None])
        self.assertEqual(detail["narrations"], ["drives on", "drives on"])
        self.assertEqual(chosen, 1)  # order[0] = candidate 1

    def test_a_truncated_walkthrough_is_not_read_as_a_selection(self):
        """Fig. 9 prose has no schema to hold it, so a cut-off walkthrough still needs the rule:
        the budget cut the answer off mid-list, so its first number is a mode it was rejecting."""
        answer = "Mode 1: the robot collides. Mode 2: the robot collides"
        chosen, detail, _ = self.run_select("descriptions", answer, finish_reason="length")
        self.assertEqual(detail["parse"]["rule"], "leading_mode_truncated")
        self.assertEqual((detail["status"], chosen), ("unparsed", None))


XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.3"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""


# The snapshot walks objects by the module their class lives in, as it does for robosuite/robocasa.
class Controller:
    pass


class Renderer:
    pass


class Env:
    pass


Controller.__module__ = "robosuite.controllers.fake"
Renderer.__module__ = "robosuite.renderers.fake"
Env.__module__ = "robocasa.environments.fake"


def build_env():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = Env()
    env.sim = SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data))
    env.timestep = 0
    env.controller = Controller()
    env.controller.goal = np.zeros(3)
    env.controllers = [env.controller]
    env.history = collections.deque([1, 2], maxlen=3)
    env.contacts = {"dog": False}
    env.rng = np.random.default_rng(3)
    env.renderer = Renderer()
    return env


def advance(env, steps=25):
    model, data = env.sim.model._model, env.sim.data._data
    for _ in range(steps):
        data.xfrc_applied[1, 0] = 5.0 + np.random.rand()
        mujoco.mj_step(model, data)
        env.timestep += 1
        env.history.append(env.timestep)
    model.body_quat[1] = [0.0, 0.0, 0.0, 1.0]
    env.controller.goal += env.rng.random()
    env.contacts["cat"] = True
    env.intrusion = {"boundary_threshold": 0.3}


@unittest.skipUnless(mujoco is not None, "needs mujoco")
class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.global_rng = (np.random.get_state(), random.getstate())

    def tearDown(self):
        np.random.set_state(self.global_rng[0])
        random.setstate(self.global_rng[1])

    def test_restore_reproduces_the_captured_digest(self):
        env = build_env()
        start = snapshot.capture_env(env)
        controller = env.controller
        advance(env)
        self.assertNotEqual(snapshot.digest(snapshot.capture_env(env)), snapshot.digest(start))
        snapshot.restore_env(env, start)
        self.assertEqual(snapshot.digest(snapshot.capture_env(env)), snapshot.digest(start))
        self.assertEqual((env.timestep, list(env.history), env.contacts), (0, [1, 2], {"dog": False}))
        self.assertFalse(hasattr(env, "intrusion"))
        self.assertIs(env.controllers[0], controller)

    def test_a_restored_branch_replays_to_the_same_end_digest(self):
        """The commit check's premise: from one snapshot, the same steps end in the same state."""
        env = build_env()
        start = snapshot.capture_env(env)
        advance(env, 60)
        end = snapshot.digest(snapshot.capture_env(env))
        snapshot.restore_env(env, start)
        advance(env, 60)
        self.assertEqual(snapshot.digest(snapshot.capture_env(env)), end)

    def test_renderers_are_left_alone(self):
        env = build_env()
        renderer = env.renderer
        start = snapshot.capture_env(env)
        self.assertIn("robosuite.renderers.fake.Renderer", start.skipped_types)
        snapshot.restore_env(env, start)
        self.assertIs(env.renderer, renderer)

    def test_structural_mismatch_raises(self):
        env = build_env()
        start = snapshot.capture_env(env)
        env.controller = SimpleNamespace()
        with self.assertRaises(snapshot.SnapshotRestoreError):
            snapshot.restore_env(env, start)


if __name__ == "__main__":
    unittest.main()
