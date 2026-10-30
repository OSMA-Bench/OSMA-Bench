"""Zero-shot VLM verifier after FOREWARN (Wu et al., arXiv 2502.01828).

FOREWARN steers a policy in two VLM steps: each sampled plan is narrated as behavior, then
"Each mode's behavior narration replaces Action Plan in the prompt template in Fig. 9" and the
VLM picks one mode or rejects all of them. Its narrations come from a finetuned VLM reading
world-model latents. Here the future is simulated instead, so the narration step follows the
paper's VLM-Img baseline prompt (Fig. 12), which narrates rendered frames, adapted to one grid
of topview frames with the robot's path drawn on them. ``verifier.select_with`` says what the selection
step gets: ``descriptions`` is Fig. 9 verbatim, text only; ``images`` shows it the filmstrips instead,
each labelled with its Mode; ``both`` shows it the filmstrips and the descriptions. ``images`` asks for
no narration at all, so a decision costs one VLM call instead of one per candidate plus one. The two
that show the filmstrips answer with the Mode in a JSON object the server decodes against a schema
(``mode_choice_format``); Fig. 9's own answer is prose, and is left as prose.

The images both steps read are built here too. Frames come from the scene's ``topview`` camera,
rendered in the env while a branch runs and cropped to a region around the robot's start and the
goal. The robot's base path is drawn on them from the logged base positions, projected with the
same camera, so nothing is rendered twice.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import random
import re
import textwrap
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Fig. 9, "Prompt Template for Policy Steering". The PDF wraps lines; paragraphs are joined
# back, and "Task Description:" has no space before its value in the original.
SELECTION_PROMPT_TEMPLATE = (
    "You are an assistant providing help for household tasks. Based on the task description and the "
    "possible modes provided, your job is to select the best behavior mode that fulfills the task goal. "
    "The most important rule that needs to be considered first is in task description. Assume that each "
    "mode will be executed exactly as described, with no modifications.\n"
    "Output Requirements:\n"
    "- If neither mode fully meets the task goal or all modes present a high risk, you reject all of them.\n"
    "- If exactly one mode is suitable, you choose this one.\n"
    "- If multiple modes are equally suitable and do not conflict with constraints, you can choose any of them.\n"
    "Response Format:\n"
    "- Return the selected behavior mode by repeating the mode description. Use `none of them` for mode "
    "description if no mode is selected.\n"
    "- Include one sentence of explanation for the choice made.\n"
    "Task Description:{task_description}\n"
    "Task Condition: {task_condition}\n"
    "Behavior Modes:\n"
    "{behavior_modes}"
)

# Fig. 9 for select_with images and both: the modes arrive as images, so the answer is a Mode number
# instead of a repeated description. The task and the output requirements are the template above word
# for word; the response format is not. Left as Fig. 9 phrases it, a model shown 16 images narrates all
# of them before concluding, which is slow, runs into the token budget, and was measured making claims
# the recorded outcomes contradict. It is pinned to the JSON object of ``mode_choice_format`` instead,
# so the answer carries the number and its reason and nothing else.
IMAGE_SELECTION_PROMPT_TEMPLATE = (
    "You are an assistant providing help for household tasks. Based on the task description and the "
    "possible modes provided, your job is to select the best behavior mode that fulfills the task goal. "
    "The most important rule that needs to be considered first is in task description. Assume that each "
    "mode will be executed exactly as described, with no modifications.\n"
    "{mode_source} Each image is a grid of frames in time order, read left to right and top to bottom, "
    "seen from a camera above the scene. The orange line is the path of the robot's base: faint for the "
    "whole execution, solid up to each frame.\n"
    "Output Requirements:\n"
    "- If neither mode fully meets the task goal or all modes present a high risk, you reject all of them.\n"
    "- If exactly one mode is suitable, you choose this one.\n"
    "- If multiple modes are equally suitable and do not conflict with constraints, you can choose any of them.\n"
    "Response Format:\n"
    "- Answer with one JSON object and nothing before or after it.\n"
    "- `mode`: the selected behavior mode written as `Mode N`, using the number on its image, or "
    "`none of them` if no mode is selected.\n"
    "- `reason`: one sentence giving the reason for that choice.\n"
    "- `mode` comes first: do not work through the modes one by one, and write no reasoning ahead of it.\n"
    "Task Description:{task_description}\n"
    "Task Condition: {task_condition}\n"
    "Behavior Modes:\n"
    "{behavior_modes}"
)

# What a mode is, for each select_with that sends images.
MODE_SOURCE = {
    "images": (
        "Each mode is one image, attached below in order and labelled `Mode N` in its top-left corner."
    ),
    "both": (
        "Each mode is one image, attached below in order and labelled `Mode N` in its top-left corner, "
        "and the description listed under Behavior Modes below."
    ),
}
SELECT_WITH = ("descriptions", "images", "both")

# Fig. 12, "Prompt Template for Behavior Narration for GPT-4o in Cup Task", with the image
# description and the notes changed from grasping a cup to moving the base. The added text must
# not name the obstacle: the environment instruction is the only task information.
NARRATION_PROMPT_TEMPLATE = (
    "You are a helpful assistant that describes robot behaviors.\n"
    "Task Description: {task_description}\n"
    "Here is an image of robot execution seen from a camera above the scene. The image is a grid of "
    "frames in time order, read left to right and top to bottom ({frame_labels}). The orange line is "
    "the path of the robot's base: faint for the whole execution, solid up to each frame. Please "
    "generate a one-sentence description of the robot's behavior.\n"
    "Notes:\n"
    "1. Pay attention to how the robot moves relative to its surroundings and anything in its path, "
    "and how close it comes to them.\n"
    "2. Do not hallucinate contact or clearance that the images do not show."
)

NONE_OF_THEM = "none of them"


def build_selection_prompt(
    task_description: str, task_condition: str, descriptions: list[str], select_with: str = "descriptions"
) -> str:
    """The Fig. 9 prompt, with the modes written out, shown as images, or both."""
    if select_with == "images":
        modes = "\n".join(f"`Mode {index}`" for index in range(1, len(descriptions) + 1))
    else:
        modes = "\n".join(f"`Mode {index}. {description}`" for index, description in enumerate(descriptions, start=1))
    if select_with == "descriptions":
        return SELECTION_PROMPT_TEMPLATE.format(
            task_description=task_description,
            task_condition=task_condition,
            behavior_modes=modes,
        )
    return IMAGE_SELECTION_PROMPT_TEMPLATE.format(
        mode_source=MODE_SOURCE[select_with],
        task_description=task_description,
        task_condition=task_condition,
        behavior_modes=modes,
    )


def build_narration_prompt(task_description: str, frame_labels: list[str]) -> str:
    return NARRATION_PROMPT_TEMPLATE.format(task_description=task_description, frame_labels=", ".join(frame_labels))


def clean_narration(text: str) -> str:
    """One line without backticks, so it cannot break the backticked mode line it is put into."""
    return " ".join(text.replace("`", "").split()).strip("\"' ")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


LEADING_MODE = re.compile(r"(?:the )?(?:chosen |selected |best )?(?:behavior )?mode (?:is )?(?:mode )?(\d+)\b")
ANY_MODE = re.compile(r"\bmode (\d+)\b")
MODE_LINE = re.compile(r"mode (\d+)$")


def mode_lines(text: str) -> list[int]:
    """Numbers from the lines that are just ``Mode N``, which is the answer the prompt asks for.

    An answer that walks through the modes puts each on a line with its verdict attached
    (``Mode 1: the robot collides``), so only the bare line is its choice.
    """
    numbers = []
    for line in text.splitlines():
        match = MODE_LINE.fullmatch(_normalize(line))
        if match:
            numbers.append(int(match.group(1)))
    return numbers


MODE_CHOICE = re.compile(r"Mode (\d+)")


def mode_choice_format(count: int) -> dict[str, Any]:
    """The answer schema for the picking call: one mode, or ``none of them``, plus one reason.

    Sent as OpenAI ``response_format``. vLLM compiles the schema into a grammar and decodes against
    it, so a mode that is on no image, a second mode, and any text around the object are not tokens
    the model can emit - the answer is read with ``json.loads`` instead of interpreted.

    The values are the words the prompt uses, ``Mode N`` and ``none of them``, not bare numbers.
    A grammar removes every token off the schema, so the values have to be the ones the model was
    going to write: asked for its choice, Qwen3.5-9B writes `"mode": "Mode 1"`. Against an enum of
    `"1"`..`"16"` the `M` is not a legal character, and on one decision the decoder took the only
    other branch left and rejected all sixteen - a schema of bare numbers changes the decision,
    which is the one thing it must not do. An enum of `Mode N` returned the same choice as the
    unconstrained call. Integers with ``minimum``/``maximum`` would be worse still: numeric bounds
    are a constraint a grammar cannot express, so the range would not even be enforced.

    ``reason`` stays in the schema for the same reason: dropping it also changed the choice.
    """
    modes = [f"Mode {number}" for number in range(1, count + 1)]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "mode_choice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": modes + [NONE_OF_THEM]},
                    "reason": {"type": "string"},
                },
                "required": ["mode", "reason"],
                "additionalProperties": False,
            },
        },
    }


def parse_mode_choice(text: str, count: int) -> dict[str, Any]:
    """Read the ``mode_choice_format`` answer: a 1-based mode number, ``none``, or ``unparsed``.

    Nothing here interprets the answer - the grammar already decided its shape. What is left is the
    object not arriving at all: a server that ignored the schema, or a reply the token budget cut off
    before the closing brace, and neither is a selection.
    """
    try:
        choice = json.loads(text)["mode"]
    except (ValueError, TypeError, KeyError, IndexError):
        return {"status": "unparsed", "mode": None, "rule": "json_missing"}
    if choice == NONE_OF_THEM:
        return {"status": "none", "mode": None, "rule": "json_none"}
    match = MODE_CHOICE.fullmatch(choice) if isinstance(choice, str) else None
    if match and 1 <= int(match.group(1)) <= count:
        return {"status": "selected", "mode": int(match.group(1)), "rule": "json_mode"}
    return {"status": "unparsed", "mode": None, "rule": "json_out_of_range"}


def parse_selection(text: str, descriptions: list[str]) -> dict[str, Any]:
    """Map a Fig. 9 answer to a 1-based mode number, ``none``, or ``unparsed``.

    Only ``select_with`` descriptions is read this way, since only it asks Fig. 9's own question and
    gets prose back; the calls that show the images answer in JSON (``parse_mode_choice``).

    The requested format repeats the chosen mode's description, so a repeated description wins
    over a mode number; the earliest one in the answer is taken, and the longer description when
    one contains another. Rules are tried in order and the first match decides. A bare ``Mode N``
    line is the next best answer, the last one when the model wrote several; only then does position
    decide, and a number anywhere in the text is the last resort, since a model that walks through
    the modes mentions the ones it rejects first.
    """
    answer = _normalize(text)

    def mode_result(number: int, rule: str) -> dict[str, Any]:
        if 1 <= number <= len(descriptions):
            return {"status": "selected", "mode": number, "rule": rule}
        return {"status": "unparsed", "mode": None, "rule": f"{rule}_out_of_range"}

    if answer.startswith(NONE_OF_THEM):
        return {"status": "none", "mode": None, "rule": "leading_none"}
    matches = []
    for number, description in enumerate(descriptions, start=1):
        needle = _normalize(description)
        position = answer.find(needle) if needle else -1
        if position >= 0:
            matches.append((position, -len(needle), number))
    if matches:
        return mode_result(min(matches)[2], "description")
    lines = mode_lines(text)
    if lines:
        return mode_result(lines[-1], "mode_line")
    leading = LEADING_MODE.match(answer)
    if leading:
        return mode_result(int(leading.group(1)), "leading_mode")
    if NONE_OF_THEM in answer:
        return {"status": "none", "mode": None, "rule": "none_anywhere"}
    anywhere = ANY_MODE.search(answer)
    if anywhere:
        return mode_result(int(anywhere.group(1)), "mode_anywhere")
    return {"status": "unparsed", "mode": None, "rule": "no_match"}


def chat(
    config: dict[str, Any],
    text: str,
    pngs: list[bytes],
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One user turn to an OpenAI-compatible server (vLLM) with thinking turned off.

    ``pngs`` are attached after the text in the order given, which is the order the prompt refers to.
    An empty list is a text-only turn. ``response_format`` is passed through untouched; with a
    ``json_schema`` the server decodes against that schema and nothing off it can come back.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for png in pngs:
        url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": url}})
    payload = {
        "model": config["model"],
        "temperature": config["temperature"],
        "seed": config["seed"],
        "max_tokens": int(max_tokens),
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}],
    }
    if response_format is not None:
        payload["response_format"] = response_format
    endpoint = config["base_url"].rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=config["timeout_seconds"]) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"VLM server returned HTTP {error.code}: {detail[:2000]}") from error
    except urllib.error.URLError as error:
        raise ConnectionError(f"Cannot reach VLM server at {endpoint}: {error}") from error
    choice = body["choices"][0]
    return {
        "text": str(choice["message"].get("content") or ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": dict(body.get("usage") or {}),
        "seconds": round(time.perf_counter() - started, 3),
    }


def mode_order(seed: int, decision: int, count: int) -> list[int]:
    """``order[i]`` is the candidate shown as Mode i + 1, shuffled so a candidate's index is no position cue."""
    order = list(range(count))
    random.Random(f"{seed}:{decision}").shuffle(order)
    return order


def select(
    config: dict[str, Any],
    decision: int,
    task_description: str,
    pngs: list[bytes],
    frame_labels: list[list[str]],
    order: list[int],
    selection_pngs: list[bytes] | None = None,
) -> tuple[int | None, dict[str, Any]]:
    """Narrate every candidate's filmstrip unless the modes are the images, then pick one (Fig. 9).

    ``order[i]`` is the candidate shown as Mode i + 1; the caller shuffles it with ``mode_order`` so
    it can write the Mode number on the images. ``pngs`` and ``selection_pngs`` are both in candidate
    order; the second one is the Mode-labelled set the selection call sees, when it sees any.

    Returns (candidate or None, detail). ``detail["status"]`` is selected, none, unparsed or error;
    a failed VLM call is logged and returned as ``error`` instead of raised.
    """
    detail: dict[str, Any] = {}
    try:
        select_with = config["select_with"]
        prompts: list[str] = []
        replies: list[dict[str, Any]] = []
        narrations: list[str] = []
        # With images the selection call never reads a narration, so none is asked for: that is the
        # one call per decision instead of one per candidate plus one.
        if select_with != "images":
            prompts = [build_narration_prompt(task_description, labels) for labels in frame_labels]
            workers = max(1, min(config["max_parallel"], len(prompts)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                replies = list(pool.map(lambda prompt, png: chat(config, prompt, [png], config["narration_max_tokens"]), prompts, pngs))
            narrations = [clean_narration(reply["text"]) for reply in replies]
        descriptions = [narrations[candidate] for candidate in order] if narrations else [""] * len(order)
        # descriptions is Fig. 9 verbatim: text only, whatever the caller passed.
        send_images = select_with != "descriptions" and selection_pngs is not None
        images = [selection_pngs[candidate] for candidate in order] if send_images else []
        prompt = build_selection_prompt(task_description, config["task_condition"], descriptions, select_with)
        detail.update(narration_prompts=prompts, narrations=narrations, narration_replies=replies, mode_to_candidate=order)
        detail.update(selection_prompt=prompt, select_with=select_with, selection_images=len(images))
        # The calls that show the images answer a mode number, which a schema can hold exactly; the
        # descriptions call answers Fig. 9's own question, and pinning its prose to a schema would
        # stop being Fig. 9.
        answer_format = None if select_with == "descriptions" else mode_choice_format(len(order))
        reply = chat(config, prompt, images, config["selection_max_tokens"], answer_format)
        if answer_format is not None:
            parsed = parse_mode_choice(reply["text"], len(order))
        else:
            parsed = parse_selection(reply["text"], descriptions)
            # An answer cut off by the token budget never reached its conclusion, and one that was
            # walking through the modes names the ones it rejects first - wherever the number was read
            # from, first line included. Only a cut-off answer naming a single mode is still
            # unambiguous. A cut-off JSON answer needs no such rule: it does not parse.
            if reply["finish_reason"] == "length" and parsed["status"] == "selected":
                if len(set(ANY_MODE.findall(_normalize(reply["text"])))) > 1:
                    parsed = {"status": "unparsed", "mode": None, "rule": f"{parsed['rule']}_truncated"}
        detail.update(selection_reply=reply, parse=parsed, status=parsed["status"])
    except Exception as error:  # noqa: BLE001 - the steerer logs it and falls back
        logging.exception("VLM selection failed for decision %d", decision)
        detail.update(status="error", error=repr(error))
        return None, detail
    return (order[parsed["mode"] - 1] if parsed["status"] == "selected" else None), detail


# Topview images: a filmstrip per candidate for the verifier, one overview for people.

CAMERA_NAME = "topview"
# Twice the runner's video size, about 84 px per metre on the floor of ONE_WALL_SMALL.
RENDER_HEIGHT, RENDER_WIDTH = 1024, 1536
REGION_MARGIN_M = 1.5  # branches are rendered in this margin around the start and the goal
CROP_MARGIN_M = 0.6  # images show this margin around every path, the start, the goal and the obstacles
BOX_HEIGHTS_M = (0.0, 1.5)  # floor to the top of the robot, so tall things near the edges stay inside
FRAME_LONG_SIDE = 360
# The selection call carries every candidate at once, so its frames are smaller than the one filmstrip
# a narration call carries alone.
SELECTION_FRAME_LONG_SIDE = 220
COLUMNS = 4
HEADER = 22
MODE_HEADER = 30  # the "Mode N" band the selection call reads the answer off
GAP = 4
OVERVIEW_LONG_SIDE = 900
PANEL_WIDTH = 560

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
PAD_COLOR = (128, 128, 128)
PATH_COLOR = (255, 90, 0)
FAINT_PATH_COLOR = (255, 140, 0, 110)
# Minimum clearance over a branch, 0 to CLEARANCE_MAX_M: red, purple, blue.
CLEARANCE_STOPS = ((215, 25, 28), (142, 63, 176), (31, 95, 191))
CLEARANCE_MAX_M = 0.25

Box = tuple[int, int, int, int]  # row0, row1, col0, col1 in pixels of the full render, flipped upright


def load_font(size: int, path: str = FONT_PATH) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def save_png(png: bytes, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(png)
    return hashlib.sha256(png).hexdigest()


def time_label(offset_steps: int, control_freq: float) -> str:
    return "now" if offset_steps == 0 else f"+{offset_steps / control_freq:.1f} s"


@dataclass(frozen=True)
class TopviewCamera:
    position: np.ndarray  # (3,) world
    rotation: np.ndarray  # (3, 3), camera axes as columns
    fovy_deg: float

    @classmethod
    def from_env(cls, env: Any) -> "TopviewCamera":
        camera = env.sim.model.camera_name2id(CAMERA_NAME)
        return cls(
            position=np.array(env.sim.data.cam_xpos[camera], dtype=np.float64),
            rotation=np.array(env.sim.data.cam_xmat[camera], dtype=np.float64).reshape(3, 3),
            fovy_deg=float(env.sim.model.cam_fovy[camera]),
        )

    def project(self, points: Any) -> np.ndarray:
        """World points (N, 3) to (column, row) in ``render(RENDER_HEIGHT, RENDER_WIDTH)[::-1]``."""
        d = np.atleast_2d(np.asarray(points, dtype=np.float64)) - self.position
        x, y, z = d @ self.rotation[:, 0], d @ self.rotation[:, 1], -(d @ self.rotation[:, 2])
        focal = RENDER_HEIGHT / (2.0 * math.tan(math.radians(self.fovy_deg) / 2.0))
        return np.stack([RENDER_WIDTH / 2.0 + focal * x / z, RENDER_HEIGHT / 2.0 - focal * y / z], axis=-1)

    def box_around(self, points: Any, margin_m: float) -> Box:
        """Pixel box holding the floor rectangle around ``points`` (world xy), from the floor to BOX_HEIGHTS_M[1]."""
        xy = np.asarray(points, dtype=np.float64)[:, :2]
        xy = xy[np.isfinite(xy).all(axis=1)]
        low, high = xy.min(axis=0) - margin_m, xy.max(axis=0) + margin_m
        corners = [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in BOX_HEIGHTS_M]
        pixels = self.project(corners)
        col0, row0 = np.floor(pixels.min(axis=0))
        col1, row1 = np.ceil(pixels.max(axis=0))
        box = (max(0, int(row0)), min(RENDER_HEIGHT, int(row1)), max(0, int(col0)), min(RENDER_WIDTH, int(col1)))
        if box[1] <= box[0] or box[3] <= box[2]:
            raise ValueError(f"The topview camera does not see the points {xy.tolist()}")
        return box

    def record(self) -> dict[str, Any]:
        return {
            "name": CAMERA_NAME,
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "fovy_deg": self.fovy_deg,
            "render_size": [RENDER_HEIGHT, RENDER_WIDTH],
        }


def intersect(inner: Box, outer: Box) -> Box:
    box = (max(inner[0], outer[0]), min(inner[1], outer[1]), max(inner[2], outer[2]), min(inner[3], outer[3]))
    return box if box[1] > box[0] and box[3] > box[2] else outer


def render_region(env: Any, region: Box) -> np.ndarray:
    frame = env.sim.render(height=RENDER_HEIGHT, width=RENDER_WIDTH, camera_name=CAMERA_NAME)[::-1]
    row0, row1, col0, col1 = region
    return np.ascontiguousarray(frame[row0:row1, col0:col1])


def _view(frame: np.ndarray, region: Box, crop: Box, long_side: int) -> tuple[Image.Image, float]:
    view = frame[crop[0] - region[0] : crop[1] - region[0], crop[2] - region[2] : crop[3] - region[2]]
    scale = long_side / max(view.shape[:2])
    size = (max(1, round(view.shape[1] * scale)), max(1, round(view.shape[0] * scale)))
    return Image.fromarray(view).resize(size, Image.Resampling.LANCZOS), scale


def _pixels(camera: TopviewCamera, crop: Box, scale: float, points: Any) -> list[tuple[float, float]]:
    pixels = (camera.project(points) - [crop[2], crop[0]]) * scale
    return [(float(column), float(row)) for column, row in pixels]


def filmstrip(
    frames: list[tuple[int, np.ndarray]],
    path: np.ndarray,
    camera: TopviewCamera,
    region: Box,
    crop: Box,
    control_freq: float,
    mode: int | None = None,
    long_side: int = FRAME_LONG_SIDE,
) -> Image.Image:
    """Frames in time order, COLUMNS per row, each labelled with its time from the decision.

    ``path`` is the base position at the decision and after every step, so a frame taken after
    ``offset`` steps shows the path up to ``path[offset]``.

    ``mode`` writes "Mode N" across the top, for the selection call that is shown every candidate at
    once and answers with the number; without it the sheet is the plain strip a narration call reads.
    """
    rows = math.ceil(len(frames) / COLUMNS)
    font = load_font(15)
    banner = MODE_HEADER if mode is not None else 0
    sheet: Image.Image | None = None
    for index, (offset, frame) in enumerate(frames):
        cell, scale = _view(frame, region, crop, long_side)
        points = _pixels(camera, crop, scale, path)
        overlay = Image.new("RGBA", cell.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).line(points, fill=FAINT_PATH_COLOR, width=3, joint="curve")
        cell = Image.alpha_composite(cell.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(cell)
        if offset > 0:
            draw.line(points[: offset + 1], fill=(0, 0, 0), width=6, joint="curve")
            draw.line(points[: offset + 1], fill=PATH_COLOR, width=4, joint="curve")
        x, y = points[offset]
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=PATH_COLOR, outline=(0, 0, 0), width=2)
        if sheet is None:
            width = COLUMNS * cell.width + (COLUMNS - 1) * GAP
            sheet = Image.new("RGB", (width, banner + rows * (HEADER + cell.height) + (rows - 1) * GAP), PAD_COLOR)
            if mode is not None:
                band = ImageDraw.Draw(sheet)
                band.rectangle([0, 0, width - 1, banner - 1], fill=(255, 255, 255))
                band.text((6, 4), f"Mode {mode}", fill=(0, 0, 0), font=load_font(21))
        left = (index % COLUMNS) * (cell.width + GAP)
        top = banner + (index // COLUMNS) * (HEADER + cell.height + GAP)
        header = ImageDraw.Draw(sheet)
        header.rectangle([left, top, left + cell.width - 1, top + HEADER - 1], fill=(255, 255, 255))
        header.text((left + 5, top + 2), time_label(offset, control_freq), fill=(0, 0, 0), font=font)
        sheet.paste(cell, (left, top + HEADER))
    assert sheet is not None
    return sheet


def clearance_color(distance_m: float | None) -> tuple[int, int, int]:
    if distance_m is None or not math.isfinite(distance_m):
        return (110, 110, 110)
    t = min(max(distance_m / CLEARANCE_MAX_M, 0.0), 1.0) * (len(CLEARANCE_STOPS) - 1)
    index = min(int(t), len(CLEARANCE_STOPS) - 2)
    low, high = CLEARANCE_STOPS[index], CLEARANCE_STOPS[index + 1]
    return tuple(round(a + (b - a) * (t - index)) for a, b in zip(low, high))


def _star(center: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    return [
        (
            center[0] + (radius if k % 2 == 0 else radius * 0.45) * math.sin(math.pi * k / 5),
            center[1] - (radius if k % 2 == 0 else radius * 0.45) * math.cos(math.pi * k / 5),
        )
        for k in range(10)
    ]


def overview(
    background: np.ndarray,
    paths: list[np.ndarray],
    obstacle_start: np.ndarray,
    goal_xy: Any,
    camera: TopviewCamera,
    region: Box,
    crop: Box,
    candidates: list[dict[str, Any]],
    selected: int,
    lines: list[str],
) -> Image.Image:
    """Every candidate's path on the decision-step frame, coloured by its minimum clearance, with a text panel.

    ``candidates[c]`` holds ``mode``, ``narration`` and the branch metrics; ``lines`` head the panel.
    None of this is shown to the verifier.
    """
    image, scale = _view(background, region, crop, OVERVIEW_LONG_SIDE)
    draw = ImageDraw.Draw(image)
    start = _pixels(camera, crop, scale, paths[0][:1])[0]

    def clearance(c: int) -> float:
        value = candidates[c]["min_obstacle_distance_m"]
        return math.inf if value is None else value

    # Farthest first, so the closest passes and the executed path are on top.
    for c in sorted(range(len(paths)), key=lambda c: (c == selected, -clearance(c))):
        points = _pixels(camera, crop, scale, paths[c])
        color = clearance_color(candidates[c]["min_obstacle_distance_m"])
        if c == selected:
            draw.line(points, fill=(255, 255, 255), width=10, joint="curve")
            draw.line(points, fill=color, width=5, joint="curve")
        else:
            draw.line(points, fill=color, width=2, joint="curve")
        x, y = points[-1]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    small = load_font(14, MONO_FONT_PATH)
    end_x, end_y = _pixels(camera, crop, scale, paths[selected][-1:])[0]
    draw.text((end_x + 8, end_y - 8), f"c{selected:02d}", fill=(255, 255, 255), font=small, stroke_width=2, stroke_fill=(0, 0, 0))
    for position in obstacle_start[np.isfinite(obstacle_start).all(axis=1)]:
        x, y = _pixels(camera, crop, scale, position[None])[0]
        for dx in (-9, 9):
            draw.line([(x - 9, y - dx), (x + 9, y + dx)], fill=(0, 0, 0), width=4)
    goal = _pixels(camera, crop, scale, [[float(goal_xy[0]), float(goal_xy[1]), 0.0]])[0]
    draw.polygon(_star(goal, 16), fill=(255, 205, 0), outline=(0, 0, 0))
    draw.ellipse([start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7], fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    font, mono = load_font(15), load_font(14, MONO_FONT_PATH)
    text: list[tuple[str, Any, tuple[int, int, int]]] = []
    for line in lines:
        text.extend((part, font, (0, 0, 0)) for part in textwrap.wrap(line, 62) or [""])
    text.append(("", font, (0, 0, 0)))
    text.append(("path colour: minimum clearance to any obstacle, 0-25 cm", font, (80, 80, 80)))
    text.append(("x obstacles at the decision, star = goal, circle = start", font, (80, 80, 80)))
    text.append(("cand mode clear_cm contact success end", mono, (0, 0, 0)))
    for c, item in enumerate(candidates):
        distance = item["min_obstacle_distance_m"]
        row = (
            f"{'>' if c == selected else ' '}c{c:02d}  M{item['mode']:02d} "
            f"{'  n/a' if distance is None else f'{100 * distance:6.1f}'} "
            f"{item['contact_steps']:7d} {'yes' if item['env_success'] else ' no':>7} "
            f"{item['termination']}@{item['executed_steps']}"
        )
        text.append((row, mono, clearance_color(distance)))
    text.append(("", font, (0, 0, 0)))
    by_mode = sorted(range(len(candidates)), key=lambda c: candidates[c]["mode"])
    for c in by_mode:
        narration = candidates[c].get("narration")
        if narration:
            text.extend((part, mono, (0, 0, 0)) for part in textwrap.wrap(f"M{candidates[c]['mode']:02d} c{c:02d}: {narration}", 64))
    line_height = 19
    height = max(image.height, 16 + line_height * len(text))
    canvas = Image.new("RGB", (image.width + PANEL_WIDTH, height), (255, 255, 255))
    canvas.paste(image, (0, 0))
    panel = ImageDraw.Draw(canvas)
    for index, (line, line_font, color) in enumerate(text):
        panel.text((image.width + 12, 8 + index * line_height), line, fill=color, font=line_font)
    return canvas
