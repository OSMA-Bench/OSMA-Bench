# Plan: General Collision Detection Function in Kitchen Base Class

## Context
The codebase has fragmented collision detection: `_check_person_door_contact()` in `kitchen_doors_safe.py` uses MuJoCo contacts, while `collision_with_robot()` in `human.py` uses distance-based heuristics. We need a general `check_collision(obj_a, obj_b)` method on the `Kitchen` base class that uses MuJoCo's native contact system and works for any pair of objects (robot, person, dog, door, etc.).

## Implementation

### File: `robocasa/environments/kitchen/kitchen.py`

Add two methods to the `Kitchen` class:

#### 1. `_get_geom_ids_by_name(self, obj_name) -> set[int]`
Private helper that resolves an object name to its set of MuJoCo geom IDs.

Resolution strategy (in order):
1. **`"robot"` keyword** → match all geoms whose name starts with the robot's `naming_prefix` (e.g. `"robot0_"`, `"mobilebase0_"`)
2. **Fixture ref names** (e.g. `"posed_person"`, `"main_door"`, `"standing_table"`) → match geoms containing the fixture's `naming_prefix` (case-insensitive)
3. **Object names** from `self.objects` (e.g. `"obstacle_1"`, `"dog"`) → get `root_body`, then collect all geoms under that body subtree via `self.sim.model.body_name2id` + child geom iteration
4. **Fallback** → substring match on `obj_name` against all geom names (case-insensitive), same pattern as existing `_check_person_door_contact`

Cache results in `self._geom_id_cache[obj_name]` (cleared on each reset).

#### 2. `check_collision(self, obj_a, obj_b) -> bool`
Public method. Uses `_get_geom_ids_by_name` to resolve both arguments, then iterates `self.sim.data.contact[:self.sim.data.ncon]` to check if any contact pair matches.

```python
def check_collision(self, obj_a, obj_b):
    """
    Check if two objects are in contact using MuJoCo's contact detection.

    Args:
        obj_a (str): Name of first object (e.g. "robot", "posed_person", "obstacle_1", "main_door")
        obj_b (str): Name of second object (same options as obj_a)

    Returns:
        bool: True if any geom of obj_a is in contact with any geom of obj_b
    """
```

#### 3. Clear cache in `_setup_references`
Add `self._geom_id_cache = {}` at the end of the existing `_setup_references` method so geom IDs are re-resolved after each reset (model may change).

### Refactor existing code
- Replace `_check_person_door_contact()` in `kitchen_doors_safe.py` (lines 163-199) with `self.check_collision("posed_person", "main_door")`
- Replace `collision_with_robot()` calls in `kitchen_move_hot_object_to_table.py` (line 426) with `self.check_collision("robot", "posed_person")`

### Key files
- **Modify:** `robocasa/environments/kitchen/kitchen.py` (~30 lines added)
- **Modify:** `robocasa/environments/kitchen/single_stage/kitchen_doors_safe.py` (simplify `_check_person_door_contact`)
- **Modify:** `robocasa/environments/kitchen/single_stage/kitchen_close_door_safe.py` (same simplification if applicable)
- **Modify:** `robocasa/environments/kitchen/single_stage/kitchen_move_hot_object_to_table.py` (replace `collision_with_robot`)
- **No changes to:** `robocasa/models/fixtures/human.py` (keep existing methods for backward compat)

## Verification
1. Run an existing door-safe task and confirm `check_collision("posed_person", "main_door")` returns same results as the old `_check_person_door_contact()`
2. Run a move-hot-object task and confirm `check_collision("robot", "posed_person")` detects contact
3. Test with navigation-safe obstacles: `check_collision("robot", "obstacle_1")`
