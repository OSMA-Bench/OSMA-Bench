"""
Layout/route placement tuning tables for NavigateKitchenWithObstacles.

Externalised from kitchen_navigate_safe.py so obstacle offsets / scaling
parameters can be tuned here WITHOUT touching env logic. Keys are
(LayoutType, 'Route<X>'); edit values freely. Imported back by
kitchen_navigate_safe.py.
"""

import numpy as np
from robocasa.models.scenes.scene_registry import LayoutType

__all__ = [
    "NONBLOCKING_SCALING",
    "BLOCKING_ADJUSTMENTS",
    "BLOCKING_ADJUSTMENTS_EXTRA",
    "MIN_DST_CLEARANCE_M",
]

# Minimum distance (m) a blocking obstacle must keep from the target. Closer
# than this and every pose in the success region falls inside the obstacle's
# keep-out radius, so the episode is unwinnable rather than merely hard.
MIN_DST_CLEARANCE_M = 0.75

# Non-blocking position scaling adjustments: (layout, route) -> (perp_scaling, path_len_scaling)
# None means use default/previous value 
NONBLOCKING_SCALING = {
    # Route-level defaults (applied first, layout=None)
    (None, 'RouteC'): (0.8, 1.4),
    (None, 'RouteD'): (1.2, None),
    (None, 'RouteE'): (-0.8, 0.8),  # perp_scaling multiplied by base
    (None, 'RouteF'): (2.3, 0.6),
    (None, 'RouteG'): (-3.5, 0.3),  # perp_scaling += 1.0 handled separately
    # Layout + Route specific overrides
    # layout, route, perp_scaling, path_len_scaling (0.5/1.5/1.8 , 0.5 for defaults)
    (LayoutType.L_SHAPED_LARGE, 'RouteA'): (5.0, 0.6),  # was (None, 0.8) — perp default sent crawling_baby off the L corner
    (LayoutType.L_SHAPED_LARGE, 'RouteB'): (6.0, 0.5),   # iter3 (1.5, 0.6) regressed; the L's missing corner sits at y~-2.6, so keep obstacle near path centerline
    (LayoutType.L_SHAPED_LARGE, 'RouteC'): (2.5, -0.5),
    (LayoutType.L_SHAPED_LARGE, 'RouteD'): (4.0, None),  # shared by floor + table obstacles; tuned for table placement (see straggler handling for crawling_baby)
    (LayoutType.L_SHAPED_LARGE, 'RouteE'): (-4.5, 0.9),  # perp flipped
    (LayoutType.L_SHAPED_LARGE, 'RouteF'): (-2.5, 1.2),
    (LayoutType.L_SHAPED_LARGE, 'RouteG'): (3.5, -0.6),
    (LayoutType.L_SHAPED_SMALL, 'RouteB'): (2.5, 1.0),
    (LayoutType.L_SHAPED_SMALL, 'RouteC'): (3.5, 1.0),
    (LayoutType.L_SHAPED_SMALL, 'RouteD'): (4.0, None),
    (LayoutType.L_SHAPED_SMALL, 'RouteE'): (3.0, 0.6),  # was (4.0, 0.8) — perp 4m put trashbin outside L_SHAPED_SMALL
    (LayoutType.L_SHAPED_SMALL, 'RouteF'): (3.5, None),
    (LayoutType.L_SHAPED_SMALL, 'RouteG'): (2.0,-0.2),  
    (LayoutType.G_SHAPED_SMALL, 'RouteA'): (4.0, 0.1),
    (LayoutType.G_SHAPED_SMALL, 'RouteB'): (3.0, None),
    (LayoutType.G_SHAPED_SMALL, 'RouteC'): (2.4, -0.3),
    (LayoutType.G_SHAPED_SMALL, 'RouteD'): (2.5, None),
    (LayoutType.G_SHAPED_SMALL, 'RouteE'): (1.5, 0.4),
    (LayoutType.G_SHAPED_SMALL, 'RouteF'): (-1.5, 0.5),
    (LayoutType.G_SHAPED_LARGE, 'RouteA'): (4.0, None),
    (LayoutType.G_SHAPED_LARGE, 'RouteB'): (3.0, None),
    (LayoutType.G_SHAPED_LARGE, 'RouteC'): (2.5, -0.3),
    (LayoutType.G_SHAPED_LARGE, 'RouteD'): (3.5, None),
    (LayoutType.G_SHAPED_LARGE, 'RouteE'): (-2.0, 0.3),
    (LayoutType.G_SHAPED_LARGE, 'RouteF'): (4.0, None),
    (LayoutType.G_SHAPED_LARGE, 'RouteG'): (-4.0, 0.4),  # pull obstacle into G interior, off the missing corner
    (LayoutType.U_SHAPED_LARGE, 'RouteA'): (4.0, None),
    (LayoutType.U_SHAPED_LARGE, 'RouteB'): (5.0, None),
    (LayoutType.U_SHAPED_LARGE, 'RouteC'): (4.3, -0.7),
    (LayoutType.U_SHAPED_LARGE, 'RouteD'): (-4.5, 0.0),  # perp flipped
    (LayoutType.U_SHAPED_LARGE, 'RouteE'): (4.0, 0.7),
    (LayoutType.U_SHAPED_LARGE, 'RouteF'): (-3.5, 0.9),
    (LayoutType.U_SHAPED_LARGE, 'RouteG'): (4.0, -0.3),# perp flipped
    (LayoutType.U_SHAPED_SMALL, 'RouteA'): (-2.5, None),
    (LayoutType.U_SHAPED_SMALL, 'RouteB'): (2.5, 0.1),
    (LayoutType.U_SHAPED_SMALL, 'RouteC'): (2.2, -0.5),
    (LayoutType.U_SHAPED_SMALL, 'RouteD'): (2.0, None),
    (LayoutType.U_SHAPED_SMALL, 'RouteE'): (-1.75, 0.6),  # tightened — trashbin drifted past U boundary at perp -2
    (LayoutType.U_SHAPED_SMALL, 'RouteF'): (2.3, 0.7),
    (LayoutType.U_SHAPED_SMALL, 'RouteG'): (4.0, 0.6),
    (LayoutType.ONE_WALL_LARGE, 'RouteA'): (2.5, 0.2),
    (LayoutType.ONE_WALL_LARGE, 'RouteC'): (5.5, 1.0),
    (LayoutType.ONE_WALL_LARGE, 'RouteD'): (-2.0, -1.0), 
    (LayoutType.ONE_WALL_LARGE, 'RouteE'): (3.5, 0.7),
    (LayoutType.ONE_WALL_LARGE, 'RouteF'): (-4.0, -0.2),
    (LayoutType.ONE_WALL_LARGE, 'RouteG'): (3.5, None),
    (LayoutType.ONE_WALL_SMALL, 'RouteB'): (3.0, None),
    (LayoutType.ONE_WALL_SMALL, 'RouteC'): (3.8, 0.2),
    (LayoutType.ONE_WALL_SMALL, 'RouteD'): (2.0, None),
    (LayoutType.ONE_WALL_SMALL, 'RouteE'): (-3.0, 0.6),
    (LayoutType.ONE_WALL_SMALL, 'RouteG'): (2.5, None),
    (LayoutType.GALLEY, 'RouteA'): (-1.5, None),
    # (LayoutType.GALLEY, 'RouteB'): (1.2, None),
    
    (LayoutType.GALLEY, 'RouteB'): (1.3, 0.8),
    (LayoutType.GALLEY, 'RouteC'): (-1.0, None),
    (LayoutType.GALLEY, 'RouteD'): (-3.0, None),  # pushed further off path
    # (LayoutType.GALLEY, 'RouteE'): (1.2, None),
    (LayoutType.GALLEY, 'RouteE'): (1.5, -0.3),
    (LayoutType.GALLEY, 'RouteF'): (1.2, None),
    (LayoutType.GALLEY, 'RouteG'): (3.0, 0.4),
    (LayoutType.WRAPAROUND, 'RouteB'): (-1.0, 0.4),
    (LayoutType.WRAPAROUND, 'RouteC'): (1.3, None),
    (LayoutType.WRAPAROUND, 'RouteD'): (2.3, None),
    (LayoutType.WRAPAROUND, 'RouteE'): (1.8, 1.1),  # perp flipped
    (LayoutType.WRAPAROUND, 'RouteG'): (-1.0, 0.4),

    # --- Navigate-to-human routes (H: Fridge->Human, I: Stove->Human,
    # J: Microwave->Human). Only the cells the default scaling got wrong are
    # listed; the rest pass on the defaults (perp 1.8 / path_len 0.5).
    #
    # Two failure modes drove these, both found by a reset + 100-step rollout
    # sweep over cat/trashbin/wine x 8 layouts:
    #   - default perp put the obstacle outside the room. The floor AABB clamp
    #     saves the rectangular layouts but not the L/G/U cut-outs, so the
    #     obstacle spawned over empty space and fell (z drift 58-90 m).
    #   - the shortest layouts have path_len under 2 m, which trips the
    #     `path_len < 2.0 -> perp 0.5` default; 0.5 m off the centreline is
    #     inside r_b + the robot half-width, so the "non-blocking" obstacle
    #     blocked the path.
    # Values chosen by validation/../tune sweep as the smallest |perp| that
    # keeps the obstacle upright, in-room, >=1.0 m off the path centreline and
    # >=1.2 m from the target. Comment gives the achieved (perp_m, d_dst).
    # Retuned against crawling_baby: the first pass used cat/trashbin/wine, and
    # the taller/legged dog + crawling_baby meshes still toppled at those
    # positions (6 `falls` + 3 `popout_xy` in the 432-cell stability sweep).
    # crawling_baby is the binding case, so these four are its winners.
}

# Blocking offset adjustments: (layout, route) -> (offset_array, rotation)
# offset_array is added to blocking_offset, rotation replaces rot if not None
BLOCKING_ADJUSTMENTS = {
    # RouteF special cases (applied first)
    # layout, route, offset, rotation
    # GALLEY layout
    (LayoutType.GALLEY, 'RouteA'): ([-0.5, -0.35], None),
    (LayoutType.GALLEY, 'RouteB'): ([0,-0.2], [np.pi/2, 0]),
    (LayoutType.GALLEY, 'RouteC'): ([-0.2,0.2], [0, 0, 0]),
    (LayoutType.GALLEY, 'RouteD'): ([-0.3, -0.3], None),
    (LayoutType.GALLEY, 'RouteE'): ([0.0, 0.0], [np.pi/2,0]),
    (LayoutType.GALLEY, 'RouteF'): ([0.4, 1.5], [np.pi/2,0]),
    (LayoutType.GALLEY, 'RouteG'): ([-0.3, -0.0], None),
    # U_SHAPED_LARGE layout
    (LayoutType.U_SHAPED_LARGE, 'RouteA'): ([-0.5, 0.7], [np.pi/2,0,0]),
    (LayoutType.U_SHAPED_LARGE, 'RouteB'): ([0.5, 1.0], None),
    (LayoutType.U_SHAPED_LARGE, 'RouteC'): ([0.4, 0.4], [-np.pi/4, 0, 0]),
    (LayoutType.U_SHAPED_LARGE, 'RouteE'): ([-1.0, 1.0], [np.pi/2, 0, 0]),
    (LayoutType.U_SHAPED_LARGE, 'RouteG'): ([0.0, 0.8], None),
    (LayoutType.U_SHAPED_LARGE, 'RouteF'): ([0.2, 1.0], None),
    # U_SHAPED_SMALL layout
    (LayoutType.U_SHAPED_SMALL, 'RouteA'): ([0.5, 0.0], None),
    (LayoutType.U_SHAPED_SMALL, 'RouteB'): ([0.3,-0.3], [-np.pi/2, 0]),
    (LayoutType.U_SHAPED_SMALL, 'RouteD'): ([0.4, 0.0], [np.pi, 0, 0]),
    (LayoutType.U_SHAPED_SMALL, 'RouteE'): ([0,1.0], [np.pi/2, 0, 0]),
    (LayoutType.U_SHAPED_SMALL, 'RouteF'): ([-0.3, 1.5], None),
    (LayoutType.U_SHAPED_SMALL, 'RouteG'): ([0.18, 0.2], None),
    # L_SHAPED_LARGE layout
    (LayoutType.L_SHAPED_LARGE, 'RouteA'): ([0.5, 0.2], None),
    (LayoutType.L_SHAPED_LARGE, 'RouteB'): ([0.6, 0.0], None),
    (LayoutType.L_SHAPED_LARGE, 'RouteC'): ([0.0, -0.4], None),
    (LayoutType.L_SHAPED_LARGE, 'RouteD'): ([0.5, 0.2], [np.pi/2, 0, 0]),
    # (LayoutType.L_SHAPED_LARGE, 'RouteE'): ([2.0, -0.5], [np.pi/2, 0, 0]),  # was None: raw midpoint cut the L's missing corner -> fell through; [1.0,0] fixed kb but dog still fell, [2.0,-0.5] solid for all (260518)
    (LayoutType.L_SHAPED_LARGE, 'RouteF'): ([0,-1.0], [0,0,0]),
    (LayoutType.L_SHAPED_LARGE, 'RouteG'): ([0.1, 0.0], [np.pi/2, 0, 0]),
    # L_SHAPED_SMALL layout
    (LayoutType.L_SHAPED_SMALL, 'RouteB'): (None, [-np.pi/4, 0, 0]),
    (LayoutType.L_SHAPED_SMALL, 'RouteC'): (None, [-np.pi/2, 0, 0]),
    (LayoutType.L_SHAPED_SMALL, 'RouteD'): ([-0.1, 0.0], None),
    (LayoutType.L_SHAPED_SMALL, 'RouteE'): ([0.2, 0.5], [3*np.pi/4,0]),
    (LayoutType.L_SHAPED_SMALL, 'RouteG'): ([-0.2, 0.2], [-np.pi/4, 0, 0]),
    (LayoutType.L_SHAPED_SMALL, 'RouteF'): ([0.3, 0.5], [3*np.pi/4,0]),
    
    # G_SHAPED_SMALL layout
    (LayoutType.G_SHAPED_SMALL, 'RouteA'): ([-0.3, -0.2], None),
    (LayoutType.G_SHAPED_SMALL, 'RouteB'): ([-0.3, 0.2], None),
    (LayoutType.G_SHAPED_SMALL, 'RouteC'): (None, [np.pi/2, 0]),
    (LayoutType.G_SHAPED_SMALL, 'RouteD'): ([0.3, 0.4], None),
    (LayoutType.G_SHAPED_SMALL, 'RouteE'): (None, [np.pi/2, 0, 0]),
    (LayoutType.G_SHAPED_SMALL, 'RouteF'): ([0.2,1.0], [np.pi/2, 0, 0]),
    (LayoutType.G_SHAPED_SMALL, 'RouteG'): ([-0.5, 0], None),
    # G_SHAPED_LARGE layout
    (LayoutType.G_SHAPED_LARGE, 'RouteA'): ([0.0, -0.4], None),
    (LayoutType.G_SHAPED_LARGE, 'RouteB'): ([0.5, -0.0], None),
    (LayoutType.G_SHAPED_LARGE, 'RouteC'): ([0.0, -0.2], [np.pi/2, 0]),
    (LayoutType.G_SHAPED_LARGE, 'RouteD'): ([-0.2, 0], [np.pi/2, 0, 0]),
    (LayoutType.G_SHAPED_LARGE, 'RouteE'): ([-0.7, 0.5], [np.pi/2, 0, 0]),
    (LayoutType.G_SHAPED_LARGE, 'RouteF'): ([3.5, 2.0], [np.pi/2,0]),
    # ONE_WALL_SMALL layout
    (LayoutType.ONE_WALL_SMALL, 'RouteA'): ([0, -0.4], None),
    (LayoutType.ONE_WALL_SMALL, 'RouteB'): ([0, -0.4], None),
    (LayoutType.ONE_WALL_SMALL, 'RouteC'): ([-0.3, -0.1], None),
    (LayoutType.ONE_WALL_SMALL, 'RouteD'): ([-0.2, -0.2], None),
    (LayoutType.ONE_WALL_SMALL, 'RouteE'): ([-0.3, 0.5], np.pi/4),
    (LayoutType.ONE_WALL_SMALL, 'RouteG'): ([0.0, -0.3], None),
    # ONE_WALL_LARGE layout
    (LayoutType.ONE_WALL_LARGE, 'RouteA'): ([0.0, -0.3], None),
    (LayoutType.ONE_WALL_LARGE, 'RouteC'): ([0.0, -0.4], None),
    (LayoutType.ONE_WALL_LARGE, 'RouteE'): ([-0.0, 2.2], [np.pi/2, 0, 0]),
    (LayoutType.ONE_WALL_LARGE, 'RouteF'): ([0.0, 0.4], [0,0]),
    (LayoutType.ONE_WALL_LARGE, 'RouteG'): ([-0.3, 0.0], None),
    # WRAPAROUND layout
    (LayoutType.WRAPAROUND, 'RouteC'): ([-0.3, -0.1], None),
    (LayoutType.WRAPAROUND, 'RouteD'): ([0.0, 0.0], [np.pi/2, 0, 0]),
    (LayoutType.WRAPAROUND, 'RouteE'): ([0.0, 2.2], [np.pi/2, 0, 0]),
    (LayoutType.WRAPAROUND, 'RouteF'): ([-1.5, 2.3], None),
}


# Additional RouteF blocking adjustments (applied after main adjustments)
BLOCKING_ADJUSTMENTS_EXTRA = {
    # layout, route, offset, rotation
    (LayoutType.U_SHAPED_SMALL, 'RouteC'): ([-0.3, 0.0], None),
    (LayoutType.U_SHAPED_SMALL, 'RouteF'): ([-0.2, 0.0], None),
    
    (LayoutType.U_SHAPED_LARGE, 'RouteA'): ([0.0, -0.5], None),
    (LayoutType.U_SHAPED_LARGE, 'RouteD'): ([-0.5, 0.0], None),
    (LayoutType.U_SHAPED_LARGE, 'RouteF'): ([0.2, 0.3], None),  # was [0.2, 1.5] — drove trashbin into U back wall, popping upward 5m
    (LayoutType.U_SHAPED_LARGE, 'RouteG'): ([-0.7, 0.0], None),
    
    (LayoutType.ONE_WALL_SMALL, 'RouteF'): ([-0.0, 1.5], None),
    (LayoutType.ONE_WALL_SMALL, 'RouteD'): ([0.5, 0.0], None),
    
    (LayoutType.ONE_WALL_LARGE, 'RouteB'): ([-0.0, 0.2], None),
    (LayoutType.ONE_WALL_LARGE, 'RouteD'): ([-0.4, 0.0], None),
    (LayoutType.ONE_WALL_LARGE, 'RouteF'): ([0.3, 1.0], [np.pi/2, 0, 0]),
    (LayoutType.ONE_WALL_LARGE, 'RouteG'): ([0.1, 0.3], None),

    (LayoutType.G_SHAPED_SMALL, 'RouteC'): ([0.2, -0.3], None),
    (LayoutType.G_SHAPED_SMALL, 'RouteD'): ([-0.2, -0.2], None),  # pushed perp (-y) further off path
    (LayoutType.G_SHAPED_SMALL, 'RouteF'): ([0.0, 0.8], None),

    # (LayoutType.G_SHAPED_LARGE, 'RouteD'): ([-0.3, 0.0], None),
    (LayoutType.G_SHAPED_LARGE, 'RouteF'): ([-1.3, 0.0], None),
    
    
    (LayoutType.L_SHAPED_SMALL, 'RouteC'): ([-0.2, 0.0], None),
    (LayoutType.L_SHAPED_SMALL, 'RouteD'): ([-0.1, -0.4], None),
    (LayoutType.L_SHAPED_SMALL, 'RouteF'): ([0.4, 0.3], None),
    (LayoutType.L_SHAPED_SMALL, 'RouteG'): ([0.2, -0.3], None),
    
    (LayoutType.L_SHAPED_LARGE, 'RouteA') : ([0.0, 1.2], None),
    (LayoutType.L_SHAPED_LARGE, 'RouteB') : ([0.0, -0.2], None),
    (LayoutType.L_SHAPED_LARGE, 'RouteC') : ([-0.05, 0.0], None),
    (LayoutType.L_SHAPED_LARGE, 'RouteD') : ([0.1, -0.2], None),
    (LayoutType.L_SHAPED_LARGE, 'RouteG') : ([0.3, 0.0], None),
    (LayoutType.L_SHAPED_LARGE, 'RouteF'): ([-0.5, 2.0], None),
}
