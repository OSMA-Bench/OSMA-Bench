"""
StandingTable fixture class for coffee shop style standing tables.
"""

import os
import numpy as np
from robocasa.models.fixtures.fixture import Fixture
import robocasa


class StandingTable(Fixture):
    """
    Standing table fixture (coffee shop style high table).
    A simple static fixture that can be placed in the environment.

    Table dimensions (with scale 0.6, 0.6, 1.2):
    - Table top height: ~0.88m (top_site Z = 0.439 * 2 from ground)
    - Table top radius: ~0.30m
    """

    def __init__(
        self,
        xml=None,
        name="standing_table",
        pos=None,
        size=None,
        rot=None,
        *args,
        **kwargs
    ):
        # Default to the standing_table model
        if xml is None:
            xml = os.path.join(
                robocasa.models.assets_root,
                "objects/lrs_objs/standing_table/model.xml"
            )

        super().__init__(
            xml=xml,
            name=name,
            duplicate_collision_geoms=False,
            pos=pos,
            *args,
            **kwargs
        )

        if rot is not None:
            self.set_rotation(rot)

    def get_reset_regions(self, env=None):
        """
        Get reset regions for object placement on top of the table.
        """
        return {
            "top": {
                "offset": np.array([0, 0, 0.45]),  # Slightly above table top
                "size": (0.25, 0.25),  # Circular table top area
            }
        }

    @property
    def nat_lang(self):
        """Natural language description of the fixture."""
        return "standing table"
