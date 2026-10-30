import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class MJCFObject(MujocoXMLObject):
    """
    Blender object with support for changing the scaling
    """

    def __init__(
        self,
        name,
        mjcf_path,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=None,
        density=None,
        friction=(0.95, 0.3, 0.1),
        margin=None,
        rgba=None,
        priority=None,
    ):
        # get scale in x, y, z
        if isinstance(scale, float):
            scale = [scale, scale, scale]
        elif isinstance(scale, tuple) or isinstance(scale, list):
            assert len(scale) == 3
            scale = tuple(scale)
        else:
            raise Exception("got invalid scale: {}".format(scale))
        scale = np.array(scale)

        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.margin = margin

        self.priority = priority

        self.rgba = rgba

        # read default xml
        xml_path = mjcf_path
        folder = os.path.dirname(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Stamp the category's density onto every geom, so the object weighs what its
        # OBJ_CATEGORIES entry says rather than whatever the asset was exported with.
        #
        # This is deliberately opt-in: density=None (the default) leaves the asset's own
        # value alone. A blanket override would re-mass every object in the zoo, and at
        # least one asset depends on its exported value -- lrs_objs/main_door carries
        # density="1" on a 0.9 x 1.5 m door panel and would get 100x heavier.
        #
        # Only group="0" geoms actually end up carrying mass: robosuite's base.xml compiles
        # with inertiagrouprange="0 0", so the group="1" visual geom is excluded from the
        # inertia computation no matter what density it is given (verified by scaling the
        # visual density 100x and watching the mass not move). Stamping every geom anyway
        # keeps the XML internally consistent and costs nothing, and the densities in
        # OBJ_CATEGORIES were calibrated against the realized mass, so they hold either way.
        #
        # NOTE: `_get_geoms` below looks like it already does this, and also sets solimp,
        # solref, friction, margin, rgba and priority. It does not: it overrides a robosuite
        # method that no longer exists, so nothing ever calls it and none of those
        # attributes has ever reached a geom. Only density is revived here -- switching the
        # others on would change contact dynamics for every object in the zoo at once.
        # solref rides the same opt-in switch, for the same reason and with the same
        # default-is-None semantics. It matters because the assets were exported with
        # solref="0.001 1" -- a 1 ms contact time constant against robosuite's 2 ms
        # SIMULATION_TIMESTEP (robosuite/macros.py). A contact stiffer than the integrator
        # can resolve does not converge: measured on delivery_box, the box sits 2 mm inside
        # the floor carrying 49.5 N against its own 29.4 N weight, and never stops moving
        # (0.023 m/s after 550 steps of zero action), creeping 6.1 cm across the floor.
        # Raising the time constant above the timestep lets the contact settle.
        if self.solref is not None:
            for geom in root.iter("geom"):
                geom.set("solref", array_to_string(np.array(self.solref)))

        if self.density is not None:
            for geom in root.iter("geom"):
                geom.set("density", str(self.density))

        # write modified xml (and make sure to postprocess any paths just in case)
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        xml_str = self.postprocess_model_xml(xml_str)
        time_str = str(time.time()).replace(".", "_")
        new_xml_path = os.path.join(folder, "{}_{}.xml".format(time_str, os.getpid()))
        f = open(new_xml_path, "w")
        f.write(xml_str)
        f.close()

        # initialize object with new xml we wrote
        super().__init__(
            fname=new_xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=scale,
        )

        # clean up xml - we don't need it anymore
        if os.path.exists(new_xml_path):
            os.remove(new_xml_path)

    def postprocess_model_xml(self, xml_str):
        """
        New version of postprocess model xml that only replaces robosuite file paths if necessary (otherwise
        there is an error with the "max" operation)
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

    def _get_geoms(self, root, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through

            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        geom_pairs = super(MJCFObject, self)._get_geoms(root=root, _parent=_parent)

        # modify geoms according to the attributes
        for i, (parent, element) in enumerate(geom_pairs):
            element.set("solref", array_to_string(self.solref))
            element.set("solimp", array_to_string(self.solimp))
            element.set("density", str(self.density))
            element.set("friction", array_to_string(self.friction))
            if self.margin is not None:
                element.set("margin", str(self.margin))

            if (self.rgba is not None) and (element.get("group") == "1"):
                element.set("rgba", array_to_string(self.rgba))

            if self.priority is not None:
                # set high priorit
                element.set("priority", str(self.priority))

        return geom_pairs

    @property
    def horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        site_values = string_to_array(horizontal_radius_site.get("pos"))
        return np.linalg.norm(site_values[0:2])

    def get_bbox_points(self, trans=None, rot=None):
        """
        Get the full 8 bounding box points of the object
        rot: a rotation matrix
        """
        bbox_offsets = []

        bottom_offset = self.bottom_offset
        top_offset = self.top_offset
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        horiz_radius = string_to_array(horizontal_radius_site.get("pos"))[:2]

        center = np.mean([bottom_offset, top_offset], axis=0)
        half_size = [horiz_radius[0], horiz_radius[1], top_offset[2] - center[2]]

        bbox_offsets = [
            center + half_size * np.array([-1, -1, -1]),  # p0
            center + half_size * np.array([1, -1, -1]),  # px
            center + half_size * np.array([-1, 1, -1]),  # py
            center + half_size * np.array([-1, -1, 1]),  # pz
            center + half_size * np.array([1, 1, 1]),
            center + half_size * np.array([-1, 1, 1]),
            center + half_size * np.array([1, -1, 1]),
            center + half_size * np.array([1, 1, -1]),
        ]

        if trans is None:
            trans = np.array([0, 0, 0])
        if rot is not None:
            rot = T.quat2mat(rot)
        else:
            rot = np.eye(3)

        points = [(np.matmul(rot, p) + trans) for p in bbox_offsets]
        return points
