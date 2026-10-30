from robocasa.environments.kitchen.kitchen import *


class ManipulateDoorSafe(Kitchen):
    """
    Class encapsulating the atomic manipulate door tasks.

    Args:
        behavior (str): "open" or "close". Used to define the desired
            door manipulation behavior for the task.

        door_id (str): The door fixture id to manipulate.
    """

    def __init__(
        self, behavior="open", door_id=FixtureType.DOOR_TOP_HINGE_SINGLE, *args, **kwargs
    ):
        self.door_id = door_id
        assert behavior in ["open", "close"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the door tasks.
        """
        super()._setup_kitchen_references()
        self.door_fxtr = self.register_fixture_ref("door_fxtr", dict(id=self.door_id)) # cabinet
        self.init_robot_base_pos = self.door_fxtr
        
        self.coffee_machine = self.get_fixture("coffee_machine")
        if "src_fixture" in self.fixture_refs:
            self.src_fixture = self.fixture_refs["src_fixture"]
            self.target_fixture = self.fixture_refs["target_fixture"]
        else:
            # choose a valid random start and destination fixture
            fixtures = list(self.fixtures.values())
            valid_src_fixture_classes = [
                # "CoffeeMachine",
                # "Toaster",
                # "Stove",
                # "Stovetop",
                # "SingleCabinet",
                # "HingeCabinet",
                # "OpenCabinet",
                # "Drawer",
                # "Microwave",
                # "Sink",
                # "Hood",
                # "Oven",
                "Fridge",
                # "Dishwasher",
            ]
            # keep choosing src fixture until it is a valid fixture
            while True:
                self.src_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.src_fixture).__name__
                if fxtr_class not in valid_src_fixture_classes:
                    continue
                break
            
            fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
            valid_target_fxtr_classes = [
                cls
                for cls in fxtr_classes
                if fxtr_classes.count(cls) == 1
                and cls
                in [
                    "CoffeeMachine",
                    # "Toaster",
                    # "Stove",
                    # "Stovetop",
                    # "OpenCabinet",
                    # "Microwave",
                    # "Sink",
                    # "Hood",
                    # "Oven",
                    # "Fridge",
                    # "Dishwasher",
                ]
            ]

            while True:
                self.target_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.target_fixture).__name__
                if (
                    self.target_fixture == self.src_fixture
                    or fxtr_class not in valid_target_fxtr_classes
                ):
                    continue
                if fxtr_class == "Accessory":
                    continue
                # don't sample closeby fixtures
                # if (
                #     OU.fixture_pairwise_dist(self.src_fixture, self.target_fixture)
                #     <= 1.0
                # ):
                #     continue
                break            
            self.fixture_refs["src_fixture"] = self.src_fixture
            self.fixture_refs["target_fixture"] = self.target_fixture

    def get_ep_meta(self):
        """
        Get the episode metadata for the door tasks.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        if isinstance(self.door_fxtr, Microwave):
            door_fxtr_name = "microwave"
            door_name = "door"
        elif isinstance(self.door_fxtr, SingleCabinet):
            door_fxtr_name = "cabinet"
            door_name = "door"
        elif isinstance(self.door_fxtr, HingeCabinet):
            door_fxtr_name = "cabinet"
            door_name = "doors"
        elif isinstance(self.door_fxtr, Drawer):
            door_fxtr_name = "drawer"
            door_name = "doors"
        ep_meta["lang"] = f"{self.behavior} the {door_fxtr_name} {door_name}"
        return ep_meta

    def _reset_internal(self):
        """
        Reset the environment internal state for the door tasks.
        This includes setting the door state based on the behavior.
        """
        if self.behavior == "open":
            self.door_fxtr.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        elif self.behavior == "close":
            self.door_fxtr.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        # set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened
        super()._reset_internal()

        # Set main_door position to a sensible location in the room
        # Place near the center-left of the room, away from obstacles
        self._set_main_door_position()

    def _set_main_door_position(self):
        """
        Set the main entrance door to a proper position in the room.
        """
        try:
            # Get the main_door body id
            door_body_id = self.sim.model.body_name2id("main_door_main")

            # Calculate door position based on room layout
            # Place door in the open floor area, between robot and fridge
            robot_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_base")]
            fridge_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("fridge_main_group_main")]

            # Position door in the middle of the room floor area
            door_x = (robot_pos[0] + fridge_pos[0]) / 2  # Midpoint between robot and fridge
            door_y = robot_pos[1] + 1.0  # Slightly in front of robot
            door_z = 1.0  # Lift door so bottom is at floor level (model origin is at center)

            # Set door position
            door_jnt_addr = self.sim.model.body_jntadr[door_body_id]
            if door_jnt_addr >= 0:
                # Object has a joint - set qpos
                qpos_addr = self.sim.model.jnt_qposadr[door_jnt_addr]
                self.sim.data.qpos[qpos_addr:qpos_addr+3] = [door_x, door_y, door_z]
                # Set rotation (quaternion wxyz) - face the door perpendicular to y-axis
                self.sim.data.qpos[qpos_addr+3:qpos_addr+7] = [0.707, 0, 0, 0.707]  # 90 degree rotation around z

            self.sim.forward()
        except Exception as e:
            print(f"[Warning] Could not set main_door position: {e}")


    def get_person_camera(self):
        """
        Get the name of the camera tracking the person.
        Returns 'voxview' which is attached to the person body.
        """
        return 'voxview'

    def _check_success(self):
        """
        Check if the door manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        door_state = self.door_fxtr.get_door_state(env=self)

        success = True
        for joint_p in door_state.values():
            if self.behavior == "open":
                if joint_p < 0.90:
                    success = False
                    break
            elif self.behavior == "close":
                if joint_p > 0.05:
                    success = False
                    break

        return success

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the door tasks. This includes the object placement configurations.
        Place one object inside the door fixture and 1-4 distractors on the counter.
        """
        cfgs = []

        # cfgs.append(
        #     dict(
        #         name="door_obj",
        #         obj_groups="all",
        #         graspable=True,
        #         microwavable=(True if isinstance(self.door_fxtr, Microwave) else None),
        #         placement=dict(
        #             fixture=self.door_fxtr,
        #             size=(0.30, 0.30),
        #             pos=(None, -1.0),
        #         ),
        #     )
        # )
        cfgs.append(
        )
        # cfgs.append(
        #     dict(
        #         name=f"coffee",
        #         obj_groups='coffee',
        #         placement=dict(
        #             fixture="floor_room",
        #             sample_region_kwargs=dict(
        #                 ref=self.target_fixture,
        #             ),
        #             size=(0.6, 0.6),
        #             offset=(-0.41, 1.1),
        #             pos=("ref", -0.5),
        #             rotation=[np.pi/4, 0]
                    
        #         ),
        #     )
        # )
        # cfgs.append(
        #     dict(
        #         name=f"coffee",
        #         obj_groups='coffee',
        #         placement=dict(
        #             fixture=self.coffee_machine,
        #             ensure_object_boundary_in_range=False,
        #             margin=0.0,
        #             ensure_valid_placement=True, #False,
        #             graspable=True,
        #             size=(0.5,0.5),
        #             pos=(2.5,-2.0),
        #             # pos=(0,0),
        #             rotation=[np.pi, np.pi]
        #             # rotation=[3*np.pi/4, np.pi],
        #         ),
        #     )
        # )

        # distractors
        # num_distr = self.rng.integers(1, 4)
        # for i in range(num_distr):
        #     cfgs.append(
        #         dict(
        #             name=f"distr_counter_{i+1}",
        #             obj_groups="all",
        #             placement=dict(
        #                 fixture=self.get_fixture(
        #                     FixtureType.COUNTER, ref=self.door_fxtr
        #                 ),
        #                 sample_region_kwargs=dict(
        #                     ref=self.door_fxtr,
        #                 ),
        #                 size=(1.0, 0.50),
        #                 pos=(None, -1.0),
        #                 offset=(0.0, 0.10),
        #             ),
        #         )
        #     )

        return cfgs


class OpenDoor(ManipulateDoorSafe):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="open", *args, **kwargs)


# class OpenSingleDoor(OpenDoor):
#     def __init__(self, door_id=FixtureType.DOOR_TOP_HINGE_SINGLE, *args, **kwargs):
#         super().__init__(door_id=door_id, *args, **kwargs)


# class OpenDoubleDoor(OpenDoor):
#     def __init__(self, door_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):
#         super().__init__(door_id=door_id, *args, **kwargs)


# class CloseDoor(ManipulateDoorSafe):
#     def __init__(self, behavior=None, *args, **kwargs):
#         super().__init__(behavior="close", *args, **kwargs)


# class CloseSingleDoor(CloseDoor):
#     def __init__(self, door_id=FixtureType.DOOR_TOP_HINGE_SINGLE, *args, **kwargs):
#         super().__init__(door_id=door_id, *args, **kwargs)


# class CloseDoubleDoor(CloseDoor):
#     def __init__(self, door_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):
#         super().__init__(door_id=door_id, *args, **kwargs)
