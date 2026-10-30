from robosuite.environments.base import make

# Manipulation environments
from robocasa.environments.kitchen.kitchen import Kitchen, KitchenDemo
from robocasa.environments.kitchen.multi_stage.baking.cupcake_cleanup import (
    CupcakeCleanup,
)
from robocasa.environments.kitchen.multi_stage.baking.organize_baking_ingredients import (
    OrganizeBakingIngredients,
)
from robocasa.environments.kitchen.multi_stage.baking.pastry_display import (
    PastryDisplay,
)
from robocasa.environments.kitchen.multi_stage.boiling.fill_kettle import FillKettle
from robocasa.environments.kitchen.multi_stage.boiling.heat_multiple_water import (
    HeatMultipleWater,
)
from robocasa.environments.kitchen.multi_stage.boiling.veggie_boil import VeggieBoil
from robocasa.environments.kitchen.multi_stage.brewing.arrange_tea import ArrangeTea
from robocasa.environments.kitchen.multi_stage.brewing.kettle_boiling import (
    KettleBoiling,
)
from robocasa.environments.kitchen.multi_stage.brewing.prepare_coffee import (
    PrepareCoffee,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.arrange_vegetables import (
    ArrangeVegetables,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.bread_setup_slicing import (
    BreadSetupSlicing,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.clearing_the_cutting_board import (
    ClearingTheCuttingBoard,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.meat_transfer import (
    MeatTransfer,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.organize_vegetables import (
    OrganizeVegetables,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.bowl_and_cup import (
    BowlAndCup,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.candle_cleanup import (
    CandleCleanup,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.clearing_cleaning_receptacles import (
    ClearingCleaningReceptacles,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.condiment_collection import (
    CondimentCollection,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.dessert_assembly import (
    DessertAssembly,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.drinkware_consolidation import (
    DrinkwareConsolidation,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.food_cleanup import (
    FoodCleanup,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.defrost_by_category import (
    DefrostByCategory,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.microwave_thawing import (
    MicrowaveThawing,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.quick_thaw import (
    QuickThaw,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.thaw_in_sink import (
    ThawInSink,
)
from robocasa.environments.kitchen.multi_stage.frying.assemble_cooking_array import (
    AssembleCookingArray,
)
from robocasa.environments.kitchen.multi_stage.frying.frying_pan_adjustment import (
    FryingPanAdjustment,
)
from robocasa.environments.kitchen.multi_stage.frying.meal_prep_staging import (
    MealPrepStaging,
)
from robocasa.environments.kitchen.multi_stage.frying.searing_meat import SearingMeat
from robocasa.environments.kitchen.multi_stage.frying.setup_frying import SetupFrying
from robocasa.environments.kitchen.multi_stage.making_toast.bread_selection import (
    BreadSelection,
)
from robocasa.environments.kitchen.multi_stage.making_toast.cheesy_bread import (
    CheesyBread,
)
from robocasa.environments.kitchen.multi_stage.making_toast.prepare_toast import (
    PrepareToast,
)
from robocasa.environments.kitchen.multi_stage.making_toast.sweet_savory_toast_setup import (
    SweetSavoryToastSetup,
)
from robocasa.environments.kitchen.multi_stage.meat_preparation.prep_for_tenderizing import (
    PrepForTenderizing,
)
from robocasa.environments.kitchen.multi_stage.meat_preparation.prep_marinating_meat import (
    PrepMarinatingMeat,
)
from robocasa.environments.kitchen.multi_stage.mixing_and_blending.colorful_salsa import (
    ColorfulSalsa,
)
from robocasa.environments.kitchen.multi_stage.mixing_and_blending.setup_juicing import (
    SetupJuicing,
)
from robocasa.environments.kitchen.multi_stage.mixing_and_blending.spicy_marinade import (
    SpicyMarinade,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.heat_mug import HeatMug
from robocasa.environments.kitchen.multi_stage.reheating_food.make_loaded_potato import (
    MakeLoadedPotato,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.simmering_sauce import (
    SimmeringSauce,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.waffle_reheat import (
    WaffleReheat,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.warm_croissant import (
    WarmCroissant,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.beverage_sorting import (
    BeverageSorting,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.restock_bowls import (
    RestockBowls,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.restock_pantry import (
    RestockPantry,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.stocking_breakfast_foods import (
    StockingBreakfastFoods,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.clean_microwave import (
    CleanMicrowave,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.countertop_cleanup import (
    CountertopCleanup,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.prep_for_sanitizing import (
    PrepForSanitizing,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.push_utensils_to_sink import (
    PushUtensilsToSink,
)
from robocasa.environments.kitchen.multi_stage.serving_food.dessert_upgrade import (
    DessertUpgrade,
)
from robocasa.environments.kitchen.multi_stage.serving_food.pan_transfer import (
    PanTransfer,
)
from robocasa.environments.kitchen.multi_stage.serving_food.place_food_in_bowls import (
    PlaceFoodInBowls,
)
from robocasa.environments.kitchen.multi_stage.serving_food.prepare_soup_serving import (
    PrepareSoupServing,
)
from robocasa.environments.kitchen.multi_stage.serving_food.serve_steak import (
    ServeSteak,
)
from robocasa.environments.kitchen.multi_stage.serving_food.wine_serving_prep import (
    WineServingPrep,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.arrange_bread_basket import (
    ArrangeBreadBasket,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.beverage_organization import (
    BeverageOrganization,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.date_night import (
    DateNight,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.seasoning_spice_setup import (
    SeasoningSpiceSetup,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.set_bowls_for_soup import (
    SetBowlsForSoup,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.size_sorting import (
    SizeSorting,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.bread_and_cheese import (
    BreadAndCheese,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.cereal_and_bowl import (
    CerealAndBowl,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.make_fruit_bowl import (
    MakeFruitBowl,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.veggie_dip_prep import (
    VeggieDipPrep,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.yogurt_delight_prep import (
    YogurtDelightPrep,
)
from robocasa.environments.kitchen.multi_stage.steaming_food.multistep_steaming import (
    MultistepSteaming,
)
from robocasa.environments.kitchen.multi_stage.steaming_food.steam_in_microwave import (
    SteamInMicrowave,
)
from robocasa.environments.kitchen.multi_stage.steaming_food.steam_vegetables import (
    SteamVegetables,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.drawer_utensil_sort import (
    DrawerUtensilSort,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.organize_cleaning_supplies import (
    OrganizeCleaningSupplies,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.pantry_mishap import (
    PantryMishap,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.shaker_shuffle import (
    ShakerShuffle,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.snack_sorting import (
    SnackSorting,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.dry_dishes import (
    DryDishes,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.dry_drinkware import (
    DryDrinkware,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.pre_soak_pan import (
    PreSoakPan,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.sorting_cleanup import (
    SortingCleanup,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.stack_bowls import (
    StackBowlsInSink,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.afterwash_sorting import (
    AfterwashSorting,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.clear_clutter import (
    ClearClutter,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.drain_veggies import (
    DrainVeggies,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.prewash_food_assembly import (
    PrewashFoodAssembly,
)
from robocasa.environments.kitchen.single_stage.kitchen_coffee import (
    CoffeePressButton,
    CoffeeServeMug,
    CoffeeSetupMug,
)
from robocasa.environments.kitchen.single_stage.kitchen_coffee_test import (
    CoffeeServeMug_test,
    CoffeeSetupMug_test,
)
from robocasa.environments.kitchen.single_stage.kitchen_doors import (
    CloseDoor,
    CloseDoubleDoor,
    CloseSingleDoor,
    OpenDoor,
    OpenDoubleDoor,
    OpenSingleDoor,
)
from robocasa.environments.kitchen.single_stage.kitchen_drawer import (
    CloseDrawer,
    OpenDrawer,
)
from robocasa.environments.kitchen.single_stage.kitchen_microwave import (
    TurnOffMicrowave,
    TurnOnMicrowave,
)
from robocasa.environments.kitchen.single_stage.kitchen_navigate import NavigateKitchen
from robocasa.environments.kitchen.single_stage.kitchen_pnp import (
    PnPCabToCounter,
    PnPCounterToCab,
    PnPCounterToMicrowave,
    PnPCounterToSink,
    PnPMicrowaveToCounter,
    PnPSinkToCounter,
)
from robocasa.environments.kitchen.single_stage.kitchen_sink import (
    TurnOffSinkFaucet,
    TurnOnSinkFaucet,
)
from robocasa.environments.kitchen.single_stage.kitchen_stove import (
    TurnOffStove,
    TurnOnStove,
)
### Low-level Robotics Safety (LRS) environments
from robocasa.environments.kitchen.single_stage.kitchen_handover import (
    # Base class
    HandOver,
    # Object-only classes (random human location)
    HandOverKnife,
    HandOverScissors,
    HandOverWine,
    HandOverApple,
    HandOverSponge,
    HandOverGun,
    # Location-only classes (default object: knife)
    HandOverStove,
    HandOverFridge,
    HandOverApart,
    HandOverSink,
    # Knife + Location
    HandOverKnifeStove,
    HandOverKnifeFridge,
    HandOverKnifeApart,
    HandOverKnifeSink,
    # Scissors + Location
    HandOverScissorsStove,
    HandOverScissorsFridge,
    HandOverScissorsApart,
    HandOverScissorsSink,
    # Wine + Location
    HandOverWineStove,
    HandOverWineFridge,
    HandOverWineApart,
    HandOverWineSink,
    # Apple + Location
    HandOverAppleStove,
    HandOverAppleFridge,
    HandOverAppleApart,
    HandOverAppleSink,
    # Sponge + Location
    HandOverSpongeStove,
    HandOverSpongeFridge,
    HandOverSpongeApart,
    HandOverSpongeSink,
    # Gun + Location
    HandOverGunStove,
    HandOverGunFridge,
    HandOverGunApart,
    HandOverGunSink,
)
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    # Base class
    NavigateKitchenWithObstacles,
    # Human obstacle, Blocking + Route
    NavigateKitchenHumanBlockingRouteA,
    NavigateKitchenHumanBlockingRouteB,
    NavigateKitchenHumanBlockingRouteC,
    NavigateKitchenHumanBlockingRouteD,
    NavigateKitchenHumanBlockingRouteE,
    NavigateKitchenHumanBlockingRouteG,
    # Human obstacle, Non-Blocking + Route
    NavigateKitchenHumanNonBlockingRouteA,
    NavigateKitchenHumanNonBlockingRouteB,
    NavigateKitchenHumanNonBlockingRouteC,
    NavigateKitchenHumanNonBlockingRouteD,
    NavigateKitchenHumanNonBlockingRouteE,
    NavigateKitchenHumanNonBlockingRouteG,
    # Dog Blocking + Route
    NavigateKitchenDogBlockingRouteA,
    NavigateKitchenDogBlockingRouteB,
    NavigateKitchenDogBlockingRouteC,
    NavigateKitchenDogBlockingRouteD,
    NavigateKitchenDogBlockingRouteE,
    NavigateKitchenDogBlockingRouteF,
    NavigateKitchenDogBlockingRouteG,
    # Dog Non-Blocking + Route
    NavigateKitchenDogNonBlockingRouteA,
    NavigateKitchenDogNonBlockingRouteB,
    NavigateKitchenDogNonBlockingRouteC,
    NavigateKitchenDogNonBlockingRouteD,
    NavigateKitchenDogNonBlockingRouteE,
    NavigateKitchenDogNonBlockingRouteF,
    NavigateKitchenDogNonBlockingRouteG,
    # Cat Blocking + Route
    NavigateKitchenCatBlockingRouteA,
    NavigateKitchenCatBlockingRouteB,
    NavigateKitchenCatBlockingRouteC,
    NavigateKitchenCatBlockingRouteD,
    NavigateKitchenCatBlockingRouteE,
    NavigateKitchenCatBlockingRouteF,
    NavigateKitchenCatBlockingRouteG,
    # Cat Non-Blocking + Route
    NavigateKitchenCatNonBlockingRouteA,
    NavigateKitchenCatNonBlockingRouteB,
    NavigateKitchenCatNonBlockingRouteC,
    NavigateKitchenCatNonBlockingRouteD,
    NavigateKitchenCatNonBlockingRouteE,
    NavigateKitchenCatNonBlockingRouteF,
    NavigateKitchenCatNonBlockingRouteG,
    # Wine Blocking + Route
    NavigateKitchenWineBlockingRouteA,
    NavigateKitchenWineBlockingRouteB,
    NavigateKitchenWineBlockingRouteC,
    NavigateKitchenWineBlockingRouteD,
    NavigateKitchenWineBlockingRouteE,
    NavigateKitchenWineBlockingRouteF,
    NavigateKitchenWineBlockingRouteG,
    # Wine Non-Blocking + Route
    NavigateKitchenWineNonBlockingRouteA,
    NavigateKitchenWineNonBlockingRouteB,
    NavigateKitchenWineNonBlockingRouteC,
    NavigateKitchenWineNonBlockingRouteD,
    NavigateKitchenWineNonBlockingRouteE,
    NavigateKitchenWineNonBlockingRouteF,
    NavigateKitchenWineNonBlockingRouteG,
    # GlassOfWater Blocking + Route
    NavigateKitchenGlassOfWaterBlockingRouteA,
    NavigateKitchenGlassOfWaterBlockingRouteB,
    NavigateKitchenGlassOfWaterBlockingRouteC,
    NavigateKitchenGlassOfWaterBlockingRouteD,
    NavigateKitchenGlassOfWaterBlockingRouteE,
    NavigateKitchenGlassOfWaterBlockingRouteF,
    NavigateKitchenGlassOfWaterBlockingRouteG,
    # GlassOfWater Non-Blocking + Route
    NavigateKitchenGlassOfWaterNonBlockingRouteA,
    NavigateKitchenGlassOfWaterNonBlockingRouteB,
    NavigateKitchenGlassOfWaterNonBlockingRouteC,
    NavigateKitchenGlassOfWaterNonBlockingRouteD,
    NavigateKitchenGlassOfWaterNonBlockingRouteE,
    NavigateKitchenGlassOfWaterNonBlockingRouteF,
    NavigateKitchenGlassOfWaterNonBlockingRouteG,
    # HotChocolate Blocking + Route
    NavigateKitchenHotChocolateBlockingRouteA,
    NavigateKitchenHotChocolateBlockingRouteB,
    NavigateKitchenHotChocolateBlockingRouteC,
    NavigateKitchenHotChocolateBlockingRouteD,
    NavigateKitchenHotChocolateBlockingRouteE,
    NavigateKitchenHotChocolateBlockingRouteF,
    NavigateKitchenHotChocolateBlockingRouteG,
    # HotChocolate Non-Blocking + Route
    NavigateKitchenHotChocolateNonBlockingRouteA,
    NavigateKitchenHotChocolateNonBlockingRouteB,
    NavigateKitchenHotChocolateNonBlockingRouteC,
    NavigateKitchenHotChocolateNonBlockingRouteD,
    NavigateKitchenHotChocolateNonBlockingRouteE,
    NavigateKitchenHotChocolateNonBlockingRouteF,
    NavigateKitchenHotChocolateNonBlockingRouteG,
    # Vase Blocking + Route
    NavigateKitchenVaseBlockingRouteA,
    NavigateKitchenVaseBlockingRouteB,
    NavigateKitchenVaseBlockingRouteC,
    NavigateKitchenVaseBlockingRouteD,
    NavigateKitchenVaseBlockingRouteE,
    NavigateKitchenVaseBlockingRouteF,
    NavigateKitchenVaseBlockingRouteG,
    # Vase Non-Blocking + Route
    NavigateKitchenVaseNonBlockingRouteA,
    NavigateKitchenVaseNonBlockingRouteB,
    NavigateKitchenVaseNonBlockingRouteC,
    NavigateKitchenVaseNonBlockingRouteD,
    NavigateKitchenVaseNonBlockingRouteE,
    NavigateKitchenVaseNonBlockingRouteF,
    NavigateKitchenVaseNonBlockingRouteG,
    # Flower pot Blocking + Route
    NavigateKitchenFlowerPotBlockingRouteA,
    NavigateKitchenFlowerPotBlockingRouteB,
    NavigateKitchenFlowerPotBlockingRouteC,
    NavigateKitchenFlowerPotBlockingRouteD,
    NavigateKitchenFlowerPotBlockingRouteE,
    NavigateKitchenFlowerPotBlockingRouteF,
    NavigateKitchenFlowerPotBlockingRouteG,
    # Flower pot Non-Blocking + Route
    NavigateKitchenFlowerPotNonBlockingRouteA,
    NavigateKitchenFlowerPotNonBlockingRouteB,
    NavigateKitchenFlowerPotNonBlockingRouteC,
    NavigateKitchenFlowerPotNonBlockingRouteD,
    NavigateKitchenFlowerPotNonBlockingRouteE,
    NavigateKitchenFlowerPotNonBlockingRouteF,
    NavigateKitchenFlowerPotNonBlockingRouteG,
    # Table lamp Blocking + Route
    NavigateKitchenTableLampBlockingRouteA,
    NavigateKitchenTableLampBlockingRouteB,
    NavigateKitchenTableLampBlockingRouteC,
    NavigateKitchenTableLampBlockingRouteD,
    NavigateKitchenTableLampBlockingRouteE,
    NavigateKitchenTableLampBlockingRouteF,
    NavigateKitchenTableLampBlockingRouteG,
    # Table lamp Non-Blocking + Route
    NavigateKitchenTableLampNonBlockingRouteA,
    NavigateKitchenTableLampNonBlockingRouteB,
    NavigateKitchenTableLampNonBlockingRouteC,
    NavigateKitchenTableLampNonBlockingRouteD,
    NavigateKitchenTableLampNonBlockingRouteE,
    NavigateKitchenTableLampNonBlockingRouteF,
    NavigateKitchenTableLampNonBlockingRouteG,
    # Child (boy) Blocking + Route
    NavigateKitchenChildBoyBlockingRouteA,
    NavigateKitchenChildBoyBlockingRouteB,
    NavigateKitchenChildBoyBlockingRouteC,
    NavigateKitchenChildBoyBlockingRouteD,
    NavigateKitchenChildBoyBlockingRouteE,
    NavigateKitchenChildBoyBlockingRouteF,
    NavigateKitchenChildBoyBlockingRouteG,
    # Child (boy) Non-Blocking + Route
    NavigateKitchenChildBoyNonBlockingRouteA,
    NavigateKitchenChildBoyNonBlockingRouteB,
    NavigateKitchenChildBoyNonBlockingRouteC,
    NavigateKitchenChildBoyNonBlockingRouteD,
    NavigateKitchenChildBoyNonBlockingRouteE,
    NavigateKitchenChildBoyNonBlockingRouteF,
    NavigateKitchenChildBoyNonBlockingRouteG,
    # Child (girl) Blocking + Route
    NavigateKitchenChildGirlBlockingRouteA,
    NavigateKitchenChildGirlBlockingRouteB,
    NavigateKitchenChildGirlBlockingRouteC,
    NavigateKitchenChildGirlBlockingRouteD,
    NavigateKitchenChildGirlBlockingRouteE,
    NavigateKitchenChildGirlBlockingRouteF,
    NavigateKitchenChildGirlBlockingRouteG,
    # Child (girl) Non-Blocking + Route
    NavigateKitchenChildGirlNonBlockingRouteA,
    NavigateKitchenChildGirlNonBlockingRouteB,
    NavigateKitchenChildGirlNonBlockingRouteC,
    NavigateKitchenChildGirlNonBlockingRouteD,
    NavigateKitchenChildGirlNonBlockingRouteE,
    NavigateKitchenChildGirlNonBlockingRouteF,
    NavigateKitchenChildGirlNonBlockingRouteG,
    # Delivery box Blocking + Route
    NavigateKitchenDeliveryBoxBlockingRouteA,
    NavigateKitchenDeliveryBoxBlockingRouteB,
    NavigateKitchenDeliveryBoxBlockingRouteC,
    NavigateKitchenDeliveryBoxBlockingRouteD,
    NavigateKitchenDeliveryBoxBlockingRouteE,
    NavigateKitchenDeliveryBoxBlockingRouteF,
    NavigateKitchenDeliveryBoxBlockingRouteG,
    # Delivery box Non-Blocking + Route
    NavigateKitchenDeliveryBoxNonBlockingRouteA,
    NavigateKitchenDeliveryBoxNonBlockingRouteB,
    NavigateKitchenDeliveryBoxNonBlockingRouteC,
    NavigateKitchenDeliveryBoxNonBlockingRouteD,
    NavigateKitchenDeliveryBoxNonBlockingRouteE,
    NavigateKitchenDeliveryBoxNonBlockingRouteF,
    NavigateKitchenDeliveryBoxNonBlockingRouteG,
    # Cardboard box Blocking + Route
    NavigateKitchenCardboardBoxBlockingRouteA,
    NavigateKitchenCardboardBoxBlockingRouteB,
    NavigateKitchenCardboardBoxBlockingRouteC,
    NavigateKitchenCardboardBoxBlockingRouteD,
    NavigateKitchenCardboardBoxBlockingRouteE,
    NavigateKitchenCardboardBoxBlockingRouteF,
    NavigateKitchenCardboardBoxBlockingRouteG,
    # Cardboard box Non-Blocking + Route
    NavigateKitchenCardboardBoxNonBlockingRouteA,
    NavigateKitchenCardboardBoxNonBlockingRouteB,
    NavigateKitchenCardboardBoxNonBlockingRouteC,
    NavigateKitchenCardboardBoxNonBlockingRouteD,
    NavigateKitchenCardboardBoxNonBlockingRouteE,
    NavigateKitchenCardboardBoxNonBlockingRouteF,
    NavigateKitchenCardboardBoxNonBlockingRouteG,
    # Wooden crate Blocking + Route
    NavigateKitchenWoodenCrateBlockingRouteA,
    NavigateKitchenWoodenCrateBlockingRouteB,
    NavigateKitchenWoodenCrateBlockingRouteC,
    NavigateKitchenWoodenCrateBlockingRouteD,
    NavigateKitchenWoodenCrateBlockingRouteE,
    NavigateKitchenWoodenCrateBlockingRouteF,
    NavigateKitchenWoodenCrateBlockingRouteG,
    # Wooden crate Non-Blocking + Route
    NavigateKitchenWoodenCrateNonBlockingRouteA,
    NavigateKitchenWoodenCrateNonBlockingRouteB,
    NavigateKitchenWoodenCrateNonBlockingRouteC,
    NavigateKitchenWoodenCrateNonBlockingRouteD,
    NavigateKitchenWoodenCrateNonBlockingRouteE,
    NavigateKitchenWoodenCrateNonBlockingRouteF,
    NavigateKitchenWoodenCrateNonBlockingRouteG,
    # Floor cushion Blocking + Route
    NavigateKitchenFloorCushionBlockingRouteA,
    NavigateKitchenFloorCushionBlockingRouteB,
    NavigateKitchenFloorCushionBlockingRouteC,
    NavigateKitchenFloorCushionBlockingRouteD,
    NavigateKitchenFloorCushionBlockingRouteE,
    NavigateKitchenFloorCushionBlockingRouteF,
    NavigateKitchenFloorCushionBlockingRouteG,
    # Floor cushion Non-Blocking + Route
    NavigateKitchenFloorCushionNonBlockingRouteA,
    NavigateKitchenFloorCushionNonBlockingRouteB,
    NavigateKitchenFloorCushionNonBlockingRouteC,
    NavigateKitchenFloorCushionNonBlockingRouteD,
    NavigateKitchenFloorCushionNonBlockingRouteE,
    NavigateKitchenFloorCushionNonBlockingRouteF,
    NavigateKitchenFloorCushionNonBlockingRouteG,
    # Duffel bag Blocking + Route
    NavigateKitchenDuffelBagBlockingRouteA,
    NavigateKitchenDuffelBagBlockingRouteB,
    NavigateKitchenDuffelBagBlockingRouteC,
    NavigateKitchenDuffelBagBlockingRouteD,
    NavigateKitchenDuffelBagBlockingRouteE,
    NavigateKitchenDuffelBagBlockingRouteF,
    NavigateKitchenDuffelBagBlockingRouteG,
    # Duffel bag Non-Blocking + Route
    NavigateKitchenDuffelBagNonBlockingRouteA,
    NavigateKitchenDuffelBagNonBlockingRouteB,
    NavigateKitchenDuffelBagNonBlockingRouteC,
    NavigateKitchenDuffelBagNonBlockingRouteD,
    NavigateKitchenDuffelBagNonBlockingRouteE,
    NavigateKitchenDuffelBagNonBlockingRouteF,
    NavigateKitchenDuffelBagNonBlockingRouteG,
    # Crawling baby Blocking + Route
    NavigateKitchenCrawlingBabyBlockingRouteA,
    NavigateKitchenCrawlingBabyBlockingRouteB,
    NavigateKitchenCrawlingBabyBlockingRouteC,
    NavigateKitchenCrawlingBabyBlockingRouteD,
    NavigateKitchenCrawlingBabyBlockingRouteE,
    NavigateKitchenCrawlingBabyBlockingRouteF,
    NavigateKitchenCrawlingBabyBlockingRouteG,
    # Crawling baby Non-Blocking + Route
    NavigateKitchenCrawlingBabyNonBlockingRouteA,
    NavigateKitchenCrawlingBabyNonBlockingRouteB,
    NavigateKitchenCrawlingBabyNonBlockingRouteC,
    NavigateKitchenCrawlingBabyNonBlockingRouteD,
    NavigateKitchenCrawlingBabyNonBlockingRouteE,
    NavigateKitchenCrawlingBabyNonBlockingRouteF,
    NavigateKitchenCrawlingBabyNonBlockingRouteG,
    # Trashbin Blocking + Route
    NavigateKitchenTrashbinBlockingRouteA,
    NavigateKitchenTrashbinBlockingRouteB,
    NavigateKitchenTrashbinBlockingRouteC,
    NavigateKitchenTrashbinBlockingRouteD,
    NavigateKitchenTrashbinBlockingRouteE,
    NavigateKitchenTrashbinBlockingRouteF,
    NavigateKitchenTrashbinBlockingRouteG,
    # Trashbin Non-Blocking + Route
    NavigateKitchenTrashbinNonBlockingRouteA,
    NavigateKitchenTrashbinNonBlockingRouteB,
    NavigateKitchenTrashbinNonBlockingRouteC,
    NavigateKitchenTrashbinNonBlockingRouteD,
    NavigateKitchenTrashbinNonBlockingRouteE,
    NavigateKitchenTrashbinNonBlockingRouteF,
    NavigateKitchenTrashbinNonBlockingRouteG,
)
from robocasa.environments.kitchen.single_stage.kitchen_move_hot_object_to_table import (
    MoveHotObjectToStandingTable,
    # Basic left/right variants
    # MoveFrypanToTableLeftHuman,
    # MoveFrypanToTableRightHuman,
    # MovePotToTableLeftHuman,
    # MovePotToTableRightHuman,
    # MoveKettleToTableLeftHuman,
    # MoveKettleToTableRightHuman,
    # MoveCoffeeToTableLeftHuman,
    # MoveCoffeeToTableRightHuman,
    # Left/right explicit-distance variants
    MoveFrypanToTableCloseLeft,
    MoveFrypanToTableCloseRight,
    MoveFrypanToTableNearLeft,
    MoveFrypanToTableNearRight,
    MoveFrypanToTableApartLeft,
    MoveFrypanToTableApartRight,
    MovePotToTableCloseLeft,
    MovePotToTableCloseRight,
    MovePotToTableNearLeft,
    MovePotToTableNearRight,
    MovePotToTableApartLeft,
    MovePotToTableApartRight,
    MoveKettleToTableCloseLeft,
    MoveKettleToTableCloseRight,
    MoveKettleToTableNearLeft,
    MoveKettleToTableNearRight,
    MoveKettleToTableApartLeft,
    MoveKettleToTableApartRight,
    MoveCoffeeToTableCloseLeft,
    MoveCoffeeToTableCloseRight,
    MoveCoffeeToTableNearLeft,
    MoveCoffeeToTableNearRight,
    MoveCoffeeToTableApartLeft,
    MoveCoffeeToTableApartRight,
    # NoHuman variants
    MoveFrypanToTableNoHuman,
    MovePotToTableNoHuman,
    MoveKettleToTableNoHuman,
    MoveCoffeeToTableNoHuman,
    # Standing-table position variants (diagonal back + close left/right human)
    MoveFrypanToTableDiagonalLeftBackCloseLeft,
    MoveFrypanToTableDiagonalLeftBackCloseRight,
    MoveFrypanToTableDiagonalRightBackCloseLeft,
    MoveFrypanToTableDiagonalRightBackCloseRight,
    MovePotToTableDiagonalLeftBackCloseLeft,
    MovePotToTableDiagonalLeftBackCloseRight,
    MovePotToTableDiagonalRightBackCloseLeft,
    MovePotToTableDiagonalRightBackCloseRight,
    MoveKettleToTableDiagonalLeftBackCloseLeft,
    MoveKettleToTableDiagonalLeftBackCloseRight,
    MoveKettleToTableDiagonalRightBackCloseLeft,
    MoveKettleToTableDiagonalRightBackCloseRight,
    MoveCoffeeToTableDiagonalLeftBackCloseLeft,
    MoveCoffeeToTableDiagonalLeftBackCloseRight,
    MoveCoffeeToTableDiagonalRightBackCloseLeft,
    MoveCoffeeToTableDiagonalRightBackCloseRight,
    # Human-diagonal basic variants
    # MoveFrypanToTableDiagonalLeft,
    # MoveFrypanToTableDiagonalRight,
    # MovePotToTableDiagonalLeft,
    # MovePotToTableDiagonalRight,
    # MoveKettleToTableDiagonalLeft,
    # MoveKettleToTableDiagonalRight,
    # MoveCoffeeToTableDiagonalLeft,
    # MoveCoffeeToTableDiagonalRight,
    # Human-diagonal explicit-distance variants
    MoveFrypanToTableCloseDiagonalLeft,
    MoveFrypanToTableNearDiagonalLeft,
    MoveFrypanToTableApartDiagonalLeft,
    MoveFrypanToTableCloseDiagonalRight,
    MoveFrypanToTableNearDiagonalRight,
    MoveFrypanToTableApartDiagonalRight,
    MovePotToTableCloseDiagonalLeft,
    MovePotToTableNearDiagonalLeft,
    MovePotToTableApartDiagonalLeft,
    MovePotToTableCloseDiagonalRight,
    MovePotToTableNearDiagonalRight,
    MovePotToTableApartDiagonalRight,
    MoveKettleToTableCloseDiagonalLeft,
    MoveKettleToTableNearDiagonalLeft,
    MoveKettleToTableApartDiagonalLeft,
    MoveKettleToTableCloseDiagonalRight,
    MoveKettleToTableNearDiagonalRight,
    MoveKettleToTableApartDiagonalRight,
    MoveCoffeeToTableCloseDiagonalLeft,
    MoveCoffeeToTableNearDiagonalLeft,
    MoveCoffeeToTableApartDiagonalLeft,
    MoveCoffeeToTableCloseDiagonalRight,
    MoveCoffeeToTableNearDiagonalRight,
    MoveCoffeeToTableApartDiagonalRight,
)
from robocasa.environments.kitchen.single_stage.kitchen_doors_safe import (
    ManipulateDoorSafe,
    OpenDoor,
    OpenDoorSafe,
)
from robocasa.environments.kitchen.single_stage.kitchen_close_door_safe import (
    CloseDoorSafe,
    CloseDoorSafeCenter,
    CloseDoorSafeBlocking,
    CloseDoorSafeThreshold,
    CloseDoorSafeEdge,
)
from robocasa.environments.kitchen.single_stage.kitchen_island_display import IslandDisplay
from robocasa.environments.kitchen.single_stage.kitchen_layout_view import KitchenLayoutView

try:
    import mimicgen
except ImportError:
    print(
        "WARNING: mimicgen environments not imported since mimicgen is not installed!"
    )

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

# from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.controllers import ALL_PART_CONTROLLERS, load_composite_controller_config
from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.models.grippers import ALL_GRIPPERS
from robosuite.robots import ALL_ROBOTS

import mujoco

assert (
    mujoco.__version__ == "3.2.6"
), "MuJoCo version must be 3.2.6. Please run pip install mujoco==3.2.6"

import numpy

assert numpy.__version__ in [
    "1.23.2",
    "1.23.3",
    "1.23.5",
    "1.26.4",
], "numpy version must be either 1.23.{2,3,5} or 1.26.4. Please install one of these versions."

import robosuite

robosuite_version = [int(e) for e in robosuite.__version__.split(".")]
robosuite_check = True
if robosuite_version[0] < 1:
    robosuite_check = False
if robosuite_version[0] == 1 and robosuite_version[1] < 5:
    robosuite_check = False
assert (
    robosuite_check
), "robosuite version must be >=1.5.0. Please install the correct version"

__version__ = "0.2.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
