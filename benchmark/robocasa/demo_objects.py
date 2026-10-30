"""Visualize MJCF models.
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import robosuite
from robosuite.devices import Device
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import find_elements
from robosuite.utils.mjcf_utils import string_to_array as s2a
from termcolor import colored

# pynput's X backend uses python-xlib, which can fail over SSH X11 forwarding
# ("no xauthority details available"). The keyboard handling exists only to
# catch Q (quit) and N (next object) — the mujoco viewer already has its own
# key_callback that can do this without pynput. We defer deciding which to use
# until runtime (see _make_keyboard below).
try:
    from pynput.keyboard import Controller, Key, Listener
    _PYNPUT_IMPORTED = True
except Exception as _e:
    print(colored(f"[demo_objects] pynput import failed ({_e.__class__.__name__}: {_e}); "
                  "using mujoco viewer key_callback only.", "yellow"))
    _PYNPUT_IMPORTED = False

import robocasa
from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
from robocasa.scripts.download_kitchen_assets import (
    DOWNLOAD_ASSET_REGISTRY,
    download_and_extract_zip,
)


def edit_model_xml(xml_str):
    """
    This function edits the model xml with custom changes, including resolving relative paths,
    applying changes retroactively to existing demonstration files, and other custom scripts.
    Environment subclasses should modify this function to add environment-specific xml editing features.
    Args:
        xml_str (str): Mujoco sim demonstration XML file as string
    Returns:
        str: Edited xml file as string
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue

        old_path_split = old_path.split("/")
        # maybe replace all paths to robosuite assets
        check_lst = [
            loc for loc, val in enumerate(old_path_split) if val == "robosuite"
        ]
        if len(check_lst) > 0:
            ind = max(check_lst)  # last occurrence index
            new_path_split = path_split + old_path_split[ind + 1 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)

    return ET.tostring(root, encoding="utf8").decode("utf8")


def read_model(
    filepath,
    hide_sites=True,
    show_bbox=False,
    show_coll_geoms=False,
):
    with open(filepath, "r") as file:
        xml = file.read()

    xml = edit_model_xml(xml)
    root = ET.fromstring(xml)

    # add white background
    asset = find_elements(root, tags="asset")
    skybox = ET.fromstring(
        """<texture builtin="flat" height="256" rgb1="1 1 1" rgb2="1 1 1" type="skybox" width="256"/>"""
    )
    asset.append(skybox)

    # add lighting
    worldbody = find_elements(root, tags="worldbody")
    light = ET.fromstring(
        """<light pos="2.0 -2.0 2.0" dir="0.01 0.01 -1" specular="0.3 0.3 0.3" ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" directional="true" castshadow="false"/>"""
    )
    worldbody.append(light)

    # make collision geoms (in)visible
    geoms = find_elements(root, tags="geom", return_first=False)
    for g in geoms:
        if g.get("group", None) == "0":
            if show_coll_geoms:
                g.set("rgba", "1.0 0.0 0.0 0.5")
            else:
                g.set("rgba", "1.0 0.0 0.0 0.0")

    if show_bbox:
        sites = {}
        for site in find_elements(root, tags="site", return_first=False):
            name = site.get("name", None)
            if name is not None:
                sites[name] = s2a(site.get("pos"))

        ext_bbox_center = None
        ext_bbox_size = None
        if "ext_p0" in sites:
            ext_bbox_center = np.array(
                [
                    np.mean([sites["ext_p0"][0], sites["ext_px"][0]]),
                    np.mean([sites["ext_p0"][1], sites["ext_py"][1]]),
                    np.mean([sites["ext_p0"][2], sites["ext_pz"][2]]),
                ]
            )
            ext_bbox_size = np.array(
                [
                    sites["ext_px"][0] - sites["ext_p0"][0],
                    sites["ext_py"][1] - sites["ext_p0"][1],
                    sites["ext_pz"][2] - sites["ext_p0"][2],
                ]
            )
        elif "bottom_site" in sites:
            ext_bbox_center = np.mean([sites["top_site"], sites["bottom_site"]], axis=0)
            ext_bbox_size = (
                np.array(
                    [
                        sites["horizontal_radius_site"][0],
                        sites["horizontal_radius_site"][1],
                        sites["top_site"][2] - ext_bbox_center[2],
                    ]
                )
                * 2
            )

        if (ext_bbox_center is not None) and (ext_bbox_size is not None):
            ext_bbox_site = ET.fromstring(
                """<site type="box" pos="{pos}" size="{hsize}" rgba="0 1 0 0.2"/>""".format(
                    pos=a2s(ext_bbox_center),
                    hsize=a2s(ext_bbox_size / 2),
                )
            )
            worldbody.append(ext_bbox_site)

    if hide_sites:
        # hide all sites
        for site in find_elements(root, tags="site", return_first=False):
            site.set("rgba", "0 0 0 0")
    else:
        for site in find_elements(root, tags="site", return_first=False):
            rgba = s2a(site.get("rgba"))
            # rgba[-1] = 1.0
            site.set("rgba", a2s(rgba))

    info = {}

    # initialize model
    xml = ET.tostring(root, encoding="unicode")
    if filepath is not None:
        os.chdir(os.path.dirname(filepath))
    t = time.time()
    model = mujoco.MjModel.from_xml_string(xml)
    sim = MjSim(model)
    info["sim_load_time"] = time.time() - t

    return sim, info


def render_model(
    sim,
    cam_settings=None,
    state=None,
):
    if cam_settings is None:
        cam_settings = {}

    def key_callback(keycode):
        if state is None:
            return
        try:
            ch = chr(keycode).lower()
        except (ValueError, OverflowError):
            return
        if ch == "q":
            state["kill"] = 1
        elif ch == "n":
            state["reset"] = 1

    # mujoco.viewer.launch(sim.model._model, sim.data._data)
    viewer = mujoco.viewer.launch_passive(
        sim.model._model,
        sim.data._data,
        key_callback=key_callback,
        show_right_ui=False,
    )

    # viewer.cam.lookat = cam_settings["lookat"]
    # viewer.cam.azimuth = cam_settings["azimuth"]
    viewer.cam.distance = cam_settings["distance"]
    viewer.cam.elevation = cam_settings["elevation"]

    return viewer


class _ViewerKeyboard:
    """Mujoco-viewer-driven keyboard stand-in.

    Exposes the same `start_control` / `get_controller_state` surface
    as the pynput-based DemoKeyboard, but the key state is written by
    the viewer's key_callback via a shared dict.
    """

    def __init__(self):
        self._state = {"reset": 0, "kill": 0}

    def start_control(self):
        self._state["reset"] = 0
        self._state["kill"] = 0

    def get_controller_state(self):
        return dict(self._state)

    @property
    def state(self):
        return self._state


class _PynputKeyboard(Device):
    """Keyboard listener backed by pynput (requires a working display connection)."""

    def __init__(self):
        self._reset_state = 0
        self._kill_state = 0
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def start_control(self):
        self._reset_state = 0
        self._kill_state = 0

    def get_controller_state(self):
        return dict(reset=self._reset_state, kill=self._kill_state)

    def on_press(self, key):
        pass

    def on_release(self, key):
        try:
            if key.char == "q":
                self._kill_state = 1
            elif key.char == "n":
                self._reset_state = 1
        except AttributeError:
            pass


def _make_keyboard():
    """Try pynput; on any failure (e.g. python-xlib handshake error over SSH
    X11 forwarding), fall back to the viewer-driven keyboard."""
    if _PYNPUT_IMPORTED:
        try:
            return _PynputKeyboard()
        except Exception as e:
            print(colored(
                f"[demo_objects] pynput Listener failed ({e.__class__.__name__}: {e}); "
                "falling back to mujoco viewer key_callback.",
                "yellow",
            ))
    return _ViewerKeyboard()


DemoKeyboard = _make_keyboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjcf",
        type=str,
        default=None,
        help="(optional) path to specific model xml file to visualize. skip to sample random models",
    )
    parser.add_argument(
        "--show_bbox",
        action="store_true",
        help="(optional) visualize exterior bounding box (based on ext_ sites)",
    )
    parser.add_argument(
        "--show_coll_geoms",
        action="store_true",
        help="(optional) whether to hide collision geoms (group 0)",
    )
    parser.add_argument(
        "--obj_types",
        type=str,
        nargs="+",
        default=["objaverse"],
        help="(optional) object types. choose among [objaverse, aigen]",
    )
    args = parser.parse_args()

    cam_settings = {
        "distance": 0.3,
        "elevation": -30,
    }

    device = DemoKeyboard()

    obj_registries = args.obj_types

    if "aigen" in obj_registries:

        aigen_objs_path = os.path.join(
            robocasa.__path__[0], "models/assets/objects/aigen_objs"
        )
        if os.path.exists(aigen_objs_path) is False:
            download_config = DOWNLOAD_ASSET_REGISTRY["aigen_objs"]
            download_config[
                "message"
            ] = "Unable to find AI-generated objects locally. Downloading files."
            download_config["prompt_before_download"] = True
            download_and_extract_zip(**download_config)
            print(
                colored(
                    f"Ending script. Rerun script to use new AI-generated objects",
                    "yellow",
                )
            )
            exit()

    while True:
        if args.mjcf is not None:
            filepath = args.mjcf
        else:
            mjcf_kwargs, sampled_object_info = sample_kitchen_object(
                groups="all",
                obj_registries=obj_registries,
            )
            filepath = sampled_object_info["mjcf_path"]
            cat = sampled_object_info["cat"].replace("_", " ")
            aigen = "aigen_objs" in filepath
            print()
            print(colored(f"Category: {cat}", "green"))
            print(colored(f"AI-Generated? {aigen}", "green"))
            print(colored(f"Model path: {filepath}", "green"))

        sim, info = read_model(
            filepath=filepath,
            hide_sites=False,
            show_bbox=args.show_bbox,
            show_coll_geoms=args.show_coll_geoms,
        )

        if args.mjcf is None:
            print()
            print(colored("(Press N to view another object, Q to end demo)", "yellow"))
        else:
            print(colored("(Press Q to end demo)", "yellow"))

        device.start_control()

        # When pynput is unavailable, route viewer keypresses (Q/N) into the
        # _ViewerKeyboard's shared state dict so the main loop below picks them up.
        shared_state = device.state if isinstance(device, _ViewerKeyboard) else None

        viewer = render_model(
            sim=sim,
            cam_settings=cam_settings,
            state=shared_state,
        )

        time.sleep(0.5)  # add delay to prevent ghost windows from opening

        while viewer.is_running():
            if device.get_controller_state()["kill"]:
                exit()
            if device.get_controller_state()["reset"]:
                break

        viewer.close()
        del viewer

        if args.mjcf is not None:
            break
