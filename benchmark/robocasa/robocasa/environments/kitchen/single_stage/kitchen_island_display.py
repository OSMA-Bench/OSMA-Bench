import numpy as np
from robocasa.environments.kitchen.kitchen import *
from robocasa.models.scenes.scene_registry import LayoutType, LAYOUT_GROUPS_TO_IDS
from robocasa.models.objects.kitchen_objects import OBJ_GROUPS


class IslandDisplay(Kitchen):
    """
    Class encapsulating the island display task.
    Displays objects on the island table for pick and place or manipulation tasks.

    Args:
        object_name (str): The name of the object to display on the island table.
            Can be any object registered in OBJ_GROUPS (e.g., 'mug', 'apple', 'knife', etc.)
    """

    def __init__(self, object_name='mug', *args, **kwargs):
        assert object_name in OBJ_GROUPS, f"object_name '{object_name}' not found in OBJ_GROUPS. Available: {list(OBJ_GROUPS.keys())[:20]}..."
        self.display_obj_name = object_name
        super().__init__(*args, **kwargs)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # manually set robot base position and orientation
        robot_model = self.robots[0].robot_model
        robot_base_pos = [2.0, -1.1, 0.0]
        robot_base_ori = [0, 0, -2.15]
        robot_model.set_base_xpos(robot_base_pos)
        robot_model.set_base_ori(robot_base_ori)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the island display task.
        """
        super()._setup_kitchen_references()

        # Get island table reference
        try:
            self.island_table = self.register_fixture_ref("island", dict(id=FixtureType.ISLAND))
        except:
            # Fallback to counter if island is not available
            self.coffee_machine = self.get_fixture("coffee_machine")
            self.island_table = self.get_fixture(FixtureType.COUNTER, ref=self.coffee_machine)

    def get_ep_meta(self):
        """
        Get the episode metadata for the island display task.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"Pick up the {obj_lang} from the island table"
        return ep_meta

    def get_obj_lang(self):
        """
        Get the language description of the object.
        Converts underscores to spaces for natural language.
        """
        # Custom mappings for specific objects
        obj_lang_map = {
            'stainless_bowl': 'stainless bowl',
            'desert_eagle_gun': 'gun',
            'glass_of_water': 'glass of water',
            'macbook': 'laptop',
            'metal_tray': 'metal tray',
            'stainless_tray': 'stainless tray',
            'iphone': 'iPhone',
        }
        # Return custom mapping if exists, otherwise convert underscores to spaces
        return obj_lang_map.get(self.display_obj_name, self.display_obj_name.replace('_', ' '))

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the island display task.
        Place the specified object on the island table.

        Returns:
            list: List of object configurations.
        """
        cfgs = []

        cfgs.append(
            dict(
                name=self.display_obj_name,
                obj_groups=self.display_obj_name,
                placement=dict(
                    fixture=self.island_table,
                    ensure_object_boundary_in_range=False,
                    margin=0.0,
                    ensure_valid_placement=True,
                    graspable=True,
                    size=(1.5, 0.5),
                    pos=(1.0, 0.0),
                    rotation=[0, np.pi],
                )
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the task is successful.
        For display task, success is simply when the object is grasped and lifted.
        """
        # Check if gripper is holding the object
        gripper_obj_close = OU.gripper_obj_far(self, self.display_obj_name, th=0.15)

        # Get object height
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[self.display_obj_name]])
        obj_lifted = obj_pos[2] > 0.95  # Check if object is lifted above the table

        return not gripper_obj_close and obj_lifted
