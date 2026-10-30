
handover_tasks = [
    # Generic
    # "HandOver",
    # "HandOverSink",
    # "HandOverFridge",
    # "HandOverApart",

    # Knife
    # "HandOverKnife",
    "HandOverKnifeSink",
    "HandOverKnifeStove",
    "HandOverKnifeFridge",
    "HandOverKnifeApart",
    "HandOverKnifeNear",

    # Scissors
    # "HandOverScissors",
    "HandOverScissorsSink",
    "HandOverScissorsStove",
    "HandOverScissorsFridge",
    "HandOverScissorsApart",
    "HandOverScissorsNear",

    # Wine
    # "HandOverWine",
    "HandOverWineSink",
    "HandOverWineStove",
    "HandOverWineFridge",
    "HandOverWineApart",
    "HandOverWineNear",

    # Milk
    # "HandOverMilk",
    "HandOverMilkSink",
    "HandOverMilkStove",
    "HandOverMilkFridge",
    "HandOverMilkApart",
    "HandOverMilkNear",

    # # Gun
    "HandOverGun",
    "HandOverGunSink",
    "HandOverGunFridge",
    "HandOverGunStove",
    "HandOverGunApart",
    "HandOverGunNear",
    
    # Sponge
    # "HandOverSponge",
    "HandOverSpongeSink",
    "HandOverSpongeStove",
    "HandOverSpongeFridge",
    "HandOverSpongeApart",
    "HandOverSpongeNear",
    
]
# Generated from the task module rather than hand-listed: the classes are
# themselves generated (obstacle x route x blocking mode), so a literal list
# silently goes stale whenever the obstacle roster or the routes change --
# which is how the new tier obstacles ended up missing from every sweep.
def _navigate_safe_tasks():
    # Read the classes the task module actually generated rather than
    # recomputing the cross product: some obstacle/route pairs are skipped
    # (the human cannot be both the obstacle and the target on Route F), and a
    # recomputed list would name classes that do not exist.
    from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
        _NAV_CLASSES,
    )
    return sorted(_NAV_CLASSES)


navigate_safe_tasks = _navigate_safe_tasks()

# Move hot object to standing table tasks (robot turns away from human)
move_hot_object_to_table_tasks = [
    # Left/right explicit-distance variants
    "MoveFrypanToTableCloseLeft",
    "MoveFrypanToTableCloseRight",
    "MoveFrypanToTableNearLeft",
    "MoveFrypanToTableNearRight",
    "MoveFrypanToTableApartLeft",
    "MoveFrypanToTableApartRight",
    "MovePotToTableCloseLeft",
    "MovePotToTableCloseRight",
    "MovePotToTableNearLeft",
    "MovePotToTableNearRight",
    "MovePotToTableApartLeft",
    "MovePotToTableApartRight",
    "MoveKettleToTableCloseLeft",
    "MoveKettleToTableCloseRight",
    "MoveKettleToTableNearLeft",
    "MoveKettleToTableNearRight",
    "MoveKettleToTableApartLeft",
    "MoveKettleToTableApartRight",
    "MoveCoffeeToTableCloseLeft",
    "MoveCoffeeToTableCloseRight",
    "MoveCoffeeToTableNearLeft",
    "MoveCoffeeToTableNearRight",
    "MoveCoffeeToTableApartLeft",
    "MoveCoffeeToTableApartRight",

    # No-human variants
    "MoveFrypanToTableNoHuman",
    "MovePotToTableNoHuman",
    "MoveKettleToTableNoHuman",
    "MoveCoffeeToTableNoHuman",

    # Standing-table position variants (diagonal back + close left/right human)
    "MoveFrypanToTableDiagonalLeftBackCloseLeft",
    "MoveFrypanToTableDiagonalLeftBackCloseRight",
    "MoveFrypanToTableDiagonalRightBackCloseLeft",
    "MoveFrypanToTableDiagonalRightBackCloseRight",
    "MovePotToTableDiagonalLeftBackCloseLeft",
    "MovePotToTableDiagonalLeftBackCloseRight",
    "MovePotToTableDiagonalRightBackCloseLeft",
    "MovePotToTableDiagonalRightBackCloseRight",
    "MoveKettleToTableDiagonalLeftBackCloseLeft",
    "MoveKettleToTableDiagonalLeftBackCloseRight",
    "MoveKettleToTableDiagonalRightBackCloseLeft",
    "MoveKettleToTableDiagonalRightBackCloseRight",
    "MoveCoffeeToTableDiagonalLeftBackCloseLeft",
    "MoveCoffeeToTableDiagonalLeftBackCloseRight",
    "MoveCoffeeToTableDiagonalRightBackCloseLeft",
    "MoveCoffeeToTableDiagonalRightBackCloseRight",

    # Human-diagonal explicit-distance variants
    "MoveFrypanToTableCloseDiagonalLeft",
    "MoveFrypanToTableNearDiagonalLeft",
    "MoveFrypanToTableApartDiagonalLeft",
    "MoveFrypanToTableCloseDiagonalRight",
    "MoveFrypanToTableNearDiagonalRight",
    "MoveFrypanToTableApartDiagonalRight",
    "MovePotToTableCloseDiagonalLeft",
    "MovePotToTableNearDiagonalLeft",
    "MovePotToTableApartDiagonalLeft",
    "MovePotToTableCloseDiagonalRight",
    "MovePotToTableNearDiagonalRight",
    "MovePotToTableApartDiagonalRight",
    "MoveKettleToTableCloseDiagonalLeft",
    "MoveKettleToTableNearDiagonalLeft",
    "MoveKettleToTableApartDiagonalLeft",
    "MoveKettleToTableCloseDiagonalRight",
    "MoveKettleToTableNearDiagonalRight",
    "MoveKettleToTableApartDiagonalRight",
    "MoveCoffeeToTableCloseDiagonalLeft",
    "MoveCoffeeToTableNearDiagonalLeft",
    "MoveCoffeeToTableApartDiagonalLeft",
    "MoveCoffeeToTableCloseDiagonalRight",
    "MoveCoffeeToTableNearDiagonalRight",
    "MoveCoffeeToTableApartDiagonalRight",
]

task_envs_list = {
    'HandOver' : handover_tasks,
    'NavigateSafe' : navigate_safe_tasks,
    'MoveHotObject' : move_hot_object_to_table_tasks,
}
