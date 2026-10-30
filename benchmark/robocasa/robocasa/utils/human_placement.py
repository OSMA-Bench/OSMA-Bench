"""Shared placement constants for the posed_human fixture.

Kept in its own leaf module (no robocasa imports) so every kitchen environment
can pull it in without creating an import cycle.
"""

# World Z at which the posed_human fixture is placed.
#
# The mesh origin sits at mid-body, so this is (mesh height / 2) rather than a
# floor height. The previous value of 0.832 left the model's feet 2.3 cm below
# the floor plane, which both looked wrong and fed sub-floor vertices into the
# surface-to-surface distance used by the High-tier safety boundary. Measured
# against the floor in NavigateKitchenHumanBlockingRouteA: at 0.855 the lowest
# mesh vertex sits at +0.0001 m.
POSED_HUMAN_BASE_Z = 0.855
