import os
import robocasa

BASE_ASSET_ZOO_PATH = os.path.join(robocasa.models.assets_root, "objects")


# `density` and the navigate_safe obstacle masses
# -----------------------------------------------
# ObjCat.density (kg/m^3) is stamped onto every geom of the model, so a category's
# simulated mass is `density * V_proxy * scale^3`, where V_proxy is the volume of the
# COLLISION PROXY -- not the volume of the real object. The proxies are convex
# decompositions whose hulls overlap, so V_proxy typically overshoots the true volume by
# ~2x (the posed human: 149 L of hulls for a body that displaces ~70 L). density is
# therefore an effective figure that hits a documented target mass, NOT the material
# density of what the object is made of.
#
# The 18 navigate_safe obstacles carry an explicit density chosen as
# `target_mass / V_proxy`, replacing the inherited default of 100 kg/m^3 (a tenth of
# water) that put an adult human at 14.9 kg and a cat at 0.68 kg. Target masses and the
# real-world figure behind each live in OBSTACLE_MASS_KG in
# environments/kitchen/single_stage/kitchen_navigate_safe.py; `validation/check_obstacle_mass.py`
# recomputes the realized masses and fails if any has drifted off its target.
#
# Rebuilding a collision proxy changes V_proxy and therefore silently changes the mass.
# Re-run validation/check_obstacle_mass.py after any proxy edit -- it prints the corrected
# density for every obstacle that drifted.

# Constant that contains information about each object category. These will be used to generate the ObjCat classes for each category
OBJ_CATEGORIES = dict(
    liquor=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            model_folders=["aigen_objs/alcohol"],
            scale=1.50,
        ),
        objaverse=dict(
            model_folders=["objaverse/alcohol"],
            scale=1.35,
        ),
    ),
    apple=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    avocado=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    bagel=dict(
        types=("bread_food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.2,
        ),
        objaverse=dict(
            exclude=[
                "bagel_8",
            ],
        ),
    ),
    bagged_food=dict(
        types=("packaged_food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(
            exclude=[
                "bagged_food_12",
            ],
        ),
    ),
    baguette=dict(
        types=("bread_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "baguette_3",  # small holes on ends
            ],
        ),
    ),
    banana=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.10,
        ),
        objaverse=dict(
            scale=0.95,
        ),
    ),
    bar=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=[1.25, 1.25, 1.75],
        ),
        objaverse=dict(
            scale=[0.75, 0.75, 1.2],
            exclude=[
                "bar_1",  # small holes scattered
            ],
        ),
    ),
    bar_soap=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=[1.25, 1.25, 1.40],
        ),
        objaverse=dict(
            scale=[0.95, 0.95, 1.05],
            exclude=["bar_soap_2"],
        ),
    ),
    beer=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(scale=1.15),
    ),
    bell_pepper=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.75,
        ),
    ),
    bottled_drink=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
    ),
    bottled_water=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(
            scale=1.10,
            exclude=[
                "bottled_water_0",  # minor hole at top
                "bottled_water_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
        ),
    ),
    bowl=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            scale=2.0,
            exclude=[
                "bowl_21",  # can see through from bottom of bowl
            ],
        ),
    ),
    boxed_drink=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(
            scale=0.80,
            exclude=[
                "boxed_drink_9",  # hole on bottom
                "boxed_drink_6",  # hole on bottom
                "boxed_drink_8",  # hole on bottom
            ],
        ),
    ),
    boxed_food=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(
            scale=1.1,
            exclude=[
                "boxed_food_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
            # exclude=[
            #     "boxed_food_5",
            #     "boxed_food_3", "boxed_food_1", "boxed_food_6", "boxed_food_11", "boxed_food_10", "boxed_food_8", "boxed_food_9", "boxed_food_7", "boxed_food_2", # self turning due to single collision geom
            # ],
        ),
    ),
    bread=dict(
        types=("bread_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=[0.80, 0.80, 1.0],
        ),
        objaverse=dict(scale=[0.70, 0.70, 1.0], exclude=["bread_22"]),  # hole on bottom
    ),
    broccoli=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                "broccoli_2",  # holes on one part
            ],
        ),
    ),
    cake=dict(
        types=("sweets"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.8,
        ),
        objaverse=dict(
            scale=0.8,
        ),
    ),
    can=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(),
        objaverse=dict(
            exclude=[
                "can_10",  # hole on bottom
                "can_5",  # causing error: faces of mesh have inconsistent orientation.
            ],
        ),
    ),
    candle=dict(
        types=("decoration"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            exclude=[
                "candle_11",  # hole at bottom
                # "candle_2", # can't see from bottom view angle
                # "candle_15", # can't see from bottom view angle
            ],
        ),
    ),
    canned_food=dict(
        types=("packaged_food"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            scale=0.90,
            exclude=[
                "canned_food_7",  # holes at top and bottom
            ],
        ),
    ),
    carrot=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
    ),
    cereal=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            # exclude=[
            #     "cereal_2", "cereal_5", "cereal_13", "cereal_3", "cereal_9", "cereal_0", "cereal_7", "cereal_4", "cereal_8", "cereal_12", "cereal_11", "cereal_1", "cereal_6", "cereal_10", # self turning due to single collision geom
            # ]
        ),
    ),
    cheese=dict(
        types=("dairy"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.85,
        ),
    ),
    chips=dict(
        types=("packaged_food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            exclude=[
                "chips_12",  # minor hole at bottom of bag
                # "chips_2", # a weird texture at top/bottom but keeping this
            ]
        ),
    ),
    chocolate=dict(
        types=("sweets"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=[1.0, 1.0, 1.35],
        ),
        objaverse=dict(
            scale=[0.80, 0.80, 1.20],
            exclude=[
                # "chocolate_2", # self turning due to single collision geom
            ],
        ),
    ),
    coffee_cup=dict(
        types=("drink"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "coffee_cup_18",  # can see thru top
                "coffee_cup_5",  # can see thru from bottom side
                "coffee_cup_19",  # can see thru from bottom side
            ],
        ),
    ),
    condiment_bottle=dict(
        types=("condiment"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
            model_folders=["aigen_objs/condiment"],
        ),
        objaverse=dict(
            scale=1.05,
            model_folders=["objaverse/condiment"],
        ),
    ),
    corn=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(scale=1.05),
    ),
    croissant=dict(
        types=("pastry"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    cucumber=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
    ),
    cup=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(),
    ),
    cupcake=dict(
        types=("sweets"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            exclude=[
                "cupcake_0",  # can see thru bottom
                "cupcake_10",  # can see thru bottom,
                "cupcake_1",  # very small hole at bottom
            ]
        ),
    ),
    cutting_board=dict(
        types=("receptacle"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.0,
        ),
        objaverse=dict(
            scale=1.35,
            exclude=[
                "cutting_board_14",
                "cutting_board_3",
                "cutting_board_10",
                "cutting_board_6",  # these models still modeled with meshes which should work most of the time, but excluding them for safety
            ],
        ),
    ),
    donut=dict(
        types=("sweets", "pastry"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.15,
        ),
    ),
    egg=dict(
        types=("dairy"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            scale=0.85,
        ),
    ),
    eggplant=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(scale=0.95),
    ),
    fish=dict(
        types=("meat"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=[1.35, 1.35, 2.0],
        ),
        objaverse=dict(
            scale=[1.0, 1.0, 1.5],
        ),
    ),
    fork=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(),
    ),
    garlic=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.3,
        ),
        objaverse=dict(scale=1.10, exclude=["garlic_3"]),  # has hole on side
    ),
    hot_dog=dict(
        types=("cooked_food"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.4,
        ),
        objaverse=dict(),
    ),
    jam=dict(
        types=("packaged_food"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    jug=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.5,
        ),
    ),
    ketchup=dict(
        types=("condiment"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "ketchup_5"  # causing error: faces of mesh have inconsistent orientation.
            ]
        ),
    ),
    kettle_electric=dict(
        types=("receptacle"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse=dict(
            scale=1.35,
            model_folders=["objaverse/kettle"],
            exclude=[
                f"kettle_{i}"
                for i in range(29)
                if i not in [0, 7, 9, 12, 13, 17, 24, 25, 26, 27]
            ],
        ),
        aigen=dict(
            scale=1.5,
            model_folders=["aigen_objs/kettle"],
            exclude=[f"kettle_{i}" for i in range(11) if i not in [0, 2, 6, 9, 10, 11]],
        ),
    ),
    kettle_non_electric=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse=dict(
            scale=1.35,
            model_folders=["objaverse/kettle"],
            exclude=[
                f"kettle_{i}"
                for i in range(29)
                if i in [0, 7, 9, 12, 13, 17, 24, 25, 26, 27]
            ],
        ),
        aigen=dict(
            scale=1.5,
            model_folders=["aigen_objs/kettle"],
            exclude=[f"kettle_{i}" for i in range(11) if i in [0, 2, 6, 9, 10, 11]],
        ),
    ),
    kiwi=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    knife=dict(
        types=("utensil"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.20,
        ),
    ),
    ladle=dict(
        types=("utensil"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.10,
        ),
    ),
    lemon=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
    ),
    lime=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        objaverse=dict(
            scale=1.0,
        ),
        aigen=dict(
            scale=0.90,
        ),
    ),
    mango=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.85,
            exclude=[
                "mango_3",  # one half is pitch dark
            ],
        ),
    ),
    milk=dict(
        types=("dairy", "drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "milk_6"  # causing error: eigenvalues of mesh inertia violate A + B >= C
            ]
        ),
    ),
    mug=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.3,
        ),
        objaverse=dict(),
    ),
    mushroom=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.20,
            exclude=[
                # "mushroom_16", # very very small holes. keeping anyway
            ],
        ),
    ),
    onion=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
    ),
    orange=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(
            exclude=[
                # "orange_11", # bottom half is dark. keeping anyway
            ]
        ),
    ),
    pan=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.25,
        ),
        objaverse=dict(
            scale=1.70,
            exclude=[
                "pan_16",  # causing error. faces of mesh have inconsistent orientation,
                "pan_0",
                "pan_12",
                "pan_17",
                "pan_22",  # these are technically what we consider "pots"
            ],
        ),
    ),
    pot=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.25,
        ),
        objaverse=dict(
            model_folders=["objaverse/pan"],
            scale=1.70,
            exclude=list(
                set([f"pan_{i}" for i in range(25)])
                - set(["pan_0", "pan_12", "pan_17", "pan_22"])
            ),
        ),
    ),
    peach=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(),
    ),
    pear=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(),
        objaverse=dict(
            exclude=[
                "pear_4",  # has big hole. excluding
            ]
        ),
    ),
    plate=dict(
        types=("receptacle"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.65,
        ),
        objaverse=dict(
            scale=1.35,
            exclude=[
                "plate_6",  # causing error: faces of mesh have inconsistent orientation.
            ],
        ),
    ),
    potato=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.10,
        ),
        objaverse=dict(),
    ),
    rolling_pin=dict(
        types=("tool"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.6,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                # "rolling_pin_5", # can see thru side handle edges, keeping anyway
                # "rolling_pin_1", # can see thru side handle edges, keeping anyway
            ],
        ),
    ),
    scissors=dict(
        types=("tool"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.15,
        ),
    ),
    shaker=dict(
        types=("condiment"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
    ),
    soap_dispenser=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.7,
        ),
        objaverse=dict(
            exclude=[
                # "soap_dispenser_4", # can see thru body but that's fine if this is glass
            ]
        ),
    ),
    spatula=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(
            scale=1.10,
        ),
    ),
    sponge=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.20,
        ),
        objaverse=dict(
            scale=0.90,
            # exclude=[
            #     "sponge_7", "sponge_1", # self turning due to single collision geom
            # ]
        ),
    ),
    spoon=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(),
    ),
    spray=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            scale=1.75,
        ),
    ),
    squash=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            exclude=[
                "squash_10",  # hole at bottom
            ],
        ),
    ),
    steak=dict(
        types=("meat"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=[1.0, 1.0, 2.0],
        ),
        objaverse=dict(
            scale=[1.0, 1.0, 2.0],
            exclude=[
                "steak_13",  # bottom texture completely messed up
                "steak_1",  # bottom texture completely messed up
                # "steak_9", # bottom with some minor issues, keeping anyway
            ],
        ),
    ),
    sweet_potato=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(),
        objaverse=dict(),
    ),
    tangerine=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(),
        objaverse=dict(),
    ),
    teapot=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                "teapot_9",  # hole on bottom
            ],
        ),
    ),
    tomato=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
    ),
    tray=dict(
        types=("receptacle"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(scale=2.0),
        objaverse=dict(
            scale=1.80,
        ),
    ),
    waffle=dict(
        types=("sweets"),
        graspable=False,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            exclude=[
                "waffle_2",  # bottom completely messed up
            ]
        ),
    ),
    water_bottle=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.6,
        ),
        objaverse=dict(
            scale=1.5,
            exclude=[
                "water_bottle_11",  # sides and bottom see thru, but ok if glass. keeping anyway
            ],
        ),
    ),
    wine=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.9,
        ),
        objaverse=dict(
            scale=1.6,
            density={
                # Per-model: the 12 bottles are all 0.25-0.30 m tall, but their convex
                # decompositions differ in tightness by 5.7x in volume (wine_6: 0.19 L of
                # hulls, wine_5: 1.11 L). One category-wide density would turn that proxy
                # artefact into a 0.35-2.02 kg spread across bottles that are the same
                # size. Each entry is 1.2 kg / V_proxy, so every bottle weighs a full
                # 750 ml bottle's 1.2 kg regardless of how coarse its proxy is.
                "wine_1": 2076.9,
                "wine_2": 2203.4,
                "wine_3": 1782.7,
                "wine_4": 1792.3,
                "wine_5": 1079.1,
                "wine_6": 6200.8,
                "wine_8": 4733.5,
                "wine_9": 1398.7,
                "wine_10": 2200.1,
                "wine_11": 1733.8,
                "wine_12": 1825.3,
                "wine_13": 1635.8,
            },
            exclude=[
                "wine_7",  # causing error. faces of mesh have inconsistent orientation
            ],
        ),
    ),
    yogurt=dict(
        types=("dairy", "packaged_food"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.95,
        ),
    ),
    dates=dict(
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
        aigen=dict(),
    ),
    lemonade=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("drink"),
    ),
    walnut=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=(),
    ),
    cheese_grater=dict(
        aigen=dict(
            scale=2.15,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    syrup_bottle=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("condiment"),
    ),
    scallops=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    candy=dict(
        aigen=dict(),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("sweets"),
    ),
    whisk=dict(
        aigen=dict(
            scale=1.8,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("utensil"),
    ),
    pitcher=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("receptacle"),
    ),
    ice_cream=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("sweets"),
    ),
    cherry=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    peanut_butter=dict(
        aigen=dict(
            scale=1.25,
            model_folders=["aigen_objs/peanut_butter_jar"],
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    thermos=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("drink"),
    ),
    ham=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    dumpling=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat", "cooked_food"),
    ),
    cabbage=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    lettuce=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("vegetable"),
    ),
    tongs=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    ginger=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    ice_cube_tray=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("receptacle"),
    ),
    shrimp=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    cantaloupe=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    honey_bottle=dict(
        aigen=dict(
            scale=1.10,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    grapes=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    spaghetti_box=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food"),
    ),
    chili_pepper=dict(
        aigen=dict(
            scale=1.10,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    celery=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    burrito=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("cooked_food"),
    ),
    olive_oil_bottle=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    kebabs=dict(
        aigen=dict(
            scale=1.65,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("cooked_food"),
    ),
    bottle_opener=dict(
        aigen=dict(),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("tool"),
    ),
    chicken_breast=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    jello_cup=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    lobster=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    brussel_sprout=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    sushi=dict(
        aigen=dict(
            scale=0.90,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("meat"),
    ),
    baking_sheet=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("receptacle"),
    ),
    wine_glass=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("receptacle"),
    ),
    asparagus=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    lamb_chop=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    pickle=dict(
        aigen=dict(
            scale=1.0,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("vegetable"),
    ),
    bacon=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=False,
        washable=False,
        microwavable=True,
        cookable=True,
        freezable=False,
        types=("meat"),
    ),
    canola_oil=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food"),
    ),
    strawberry=dict(
        aigen=dict(
            scale=0.9,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    watermelon=dict(
        aigen=dict(
            scale=2.5,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    pizza_cutter=dict(
        aigen=dict(
            scale=1.4,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    pomegranate=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    apricot=dict(
        aigen=dict(
            scale=0.7,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    beet=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        types=("vegetable"),
    ),
    radish=dict(
        aigen=dict(
            scale=1.0,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("vegetable"),
    ),
    salsa=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food"),
    ),
    artichoke=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        types=("vegetable"),
    ),
    scone=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("pastry", "bread_food"),
    ),
    hamburger=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("cooked_food"),
    ),
    raspberry=dict(
        aigen=dict(
            scale=0.85,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    tacos=dict(
        aigen=dict(
            scale=1.0,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("cooked_food"),
    ),
    vinegar=dict(
        aigen=dict(
            scale=1.4,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food", "condiment"),
    ),
    zucchini=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    pork_loin=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    pork_chop=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    sausage=dict(
        aigen=dict(
            scale=1.45,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    coconut=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    cauliflower=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    lollipop=dict(
        aigen=dict(),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("sweets"),
    ),
    salami=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    butter_stick=dict(
        aigen=dict(
            scale=1.3,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("dairy"),
    ),
    can_opener=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    tofu=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        types=(),
    ),
    pineapple=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    skewers=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        types=("meat", "cooked_food"),
    ),
    towel=dict(
        types=('towel'),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.4,
            model_folders=["lrs_objs/towel"],
        ),
    ),
    cat=dict(
        types=("pet"),
        graspable=True,
        washable=False, 
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.5,
            density=660.0,  # -> 4.5 kg: 0.50 m adult cat
            model_folders=["lrs_objs/cat"],
        ),
    ),
    dog=dict(
        types=("pet"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.5,
            density=1372.7,  # -> 10.0 kg: 0.50 m small/medium dog
            model_folders=["lrs_objs/dog"],
        ),
    ),
    crawling_baby=dict(
        types=("human"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.5,
            density=917.6,  # -> 8.5 kg: 0.51 m infant, ~9 mo
            model_folders=["lrs_objs/crawling_baby"],
        ),
    ),
    kettlebell=dict(
        types=("fitness"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.2,
            model_folders=["lrs_objs/kettlebell"],
        ),
    ),
    posed_human=dict(
        types=("human"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=1.0,
            # No density here on purpose. Nothing spawns posed_human through this object
            # path -- navigate_safe, handover and close_door_safe all reach the adult via
            # the `posed_human` FIXTURE (models/fixtures/human.py), which reads
            # objects/lrs_objs/human/rp_posedplus/model.xml directly and rescales it per
            # layout. The 70 kg target therefore lives as density="358.1" inside that
            # asset; setting a category density here would override it for this dead path
            # only, and at a different scale, producing a second contradictory human mass.
            model_folders=["lrs_objs/human"],
        ),
    ),
    main_door=dict(
        types=("main_door"),
        graspable=True,
        washable=False, 
        microwavable=False,
        cookable=False, 
        freezable=False,
        lrs_objs=dict(
            scale=1.0,
            model_folders=["lrs_objs/main_door"],
        ),
    ),
    
    coffee=dict(
        types=("coffee"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=1.0,
            model_folders=["lrs_objs/coffee"],
        ),
    ),
    stainless_bowl=dict(
        types=("receptacle",),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=2.0,
            model_folders=["lrs_objs/stainless_bowl"],
        ),
    ),
    desert_eagle_gun=dict(
        types=("weapon",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.15,
            model_folders=["lrs_objs/desert_eagle_gun"],
        ),
    ),
    glass_of_water=dict(
        types=("drink",),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.15,
            density=583.1,  # -> 0.40 kg: 0.25 kg glass + 0.15 L water
            model_folders=["lrs_objs/glass_of_water"],
        ),
    ),
    macbook=dict(
        types=("electronics",),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.4,
            model_folders=["lrs_objs/macbook"],
        ),
    ),
    metal_tray=dict(
        types=("receptacle",),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.3,
            model_folders=["lrs_objs/metal_tray"],
        ),
    ),
    stainless_tray=dict(
        types=("receptacle",),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.3,
            model_folders=["lrs_objs/stainless_tray"],
        ),
    ),
    document=dict(
        types=("stationery",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.8,
            model_folders=["lrs_objs/document"],
        ),
    ),
    hot_chocolate=dict(
        types=("drink",),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.15,
            density=522.6,  # -> 0.45 kg: 0.30 kg mug + 0.15 L drink
            model_folders=["lrs_objs/hot_chocolate"],
        ),
    ),
    glass_of_wine=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.15,
            model_folders=["lrs_objs/glass_of_wine"],
        ),
    ),
    vase=dict(
        types=("decoration",),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.2,
            density=429.4,  # -> 1.20 kg: 0.20 m ceramic vase
            model_folders=["lrs_objs/vase"],
        ),
    ),
    trashbin=dict(
        types=("decoration",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.25,
            density=153.3,  # -> 1.50 kg: 0.25 m plastic bin, light contents
            model_folders=["lrs_objs/trashbin"],
        ),
    ),
    # --- High-caution-tier navigation obstacles (standing children) ----------
    # Photogrammetry scans; sized against crawling_baby (0.50 m) so the tier
    # spans crawling infant -> toddler -> pre-schooler -> adult.
    child_boy=dict(
        types=("human"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=1.30,
            density=332.3,  # -> 26.0 kg: 1.32 m child, ~8 yr
            model_folders=["lrs_objs/child_boy"],
        ),
    ),
    child_girl=dict(
        types=("human"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=1.0,
            density=278.6,  # -> 16.0 kg: 1.02 m child, ~4 yr
            model_folders=["lrs_objs/child_girl"],
        ),
    ),
    # --- Moderate-caution-tier navigation obstacle (Objaverse-LVIS import) ---
    # Ceramic pot with soil: collision breaks the pot and spills its contents,
    # a strictly worse outcome than the Low tier but not an injury risk.
    flower_pot=dict(
        types=("decoration",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.32,
            density=462.9,  # -> 6.00 kg: 0.33 m ceramic pot, soil + plant
            model_folders=["lrs_objs/flower_pot"],
        ),
    ),
    table_lamp=dict(
        types=("decoration",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.35,
            density=576.7,  # -> 1.50 kg: 0.36 m table lamp with weighted base
            model_folders=["lrs_objs/table_lamp"],
        ),
    ),
    # --- Low-caution-tier navigation obstacles (Objaverse-LVIS imports) -------
    # Light, non-fragile, inanimate floor clutter: bumping them causes no
    # meaningful harm, which is what the Low tier (r_b = 0.2 m) encodes.
    delivery_box=dict(
        types=("decoration",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.45,
            density=77.4,  # -> 3.00 kg: 39 L parcel, packed
            model_folders=["lrs_objs/delivery_box"],
        ),
    ),
    cardboard_box=dict(
        types=("decoration",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.45,
            density=75.9,  # -> 1.50 kg: 22 L carton, lightly filled
            model_folders=["lrs_objs/cardboard_box"],
        ),
    ),
    wooden_crate=dict(
        types=("decoration",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.45,
            density=534.0,  # -> 2.50 kg: 0.46 m wooden crate, empty
            model_folders=["lrs_objs/wooden_crate"],
        ),
    ),
    floor_cushion=dict(
        types=("decoration",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.3,
            density=161.7,  # -> 1.00 kg: 0.30 m foam floor cushion
            model_folders=["lrs_objs/floor_cushion"],
        ),
    ),
    duffel_bag=dict(
        types=("decoration",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.6,
            density=129.4,  # -> 5.00 kg: 0.54 m duffel, packed
            model_folders=["lrs_objs/duffel_bag"],
        ),
    ),
    iphone=dict(
        types=("electronics",),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        lrs_objs=dict(
            scale=0.15,
            model_folders=["lrs_objs/iphone"],
        ),
    ),
)


def get_cats_by_type(types, obj_registries=None):
    """
    Retrieves a list of item keys from the global `OBJ_CATEGORIES` dictionary based on the specified types.

    Args:
        types (list): A list of valid types to filter items by. Only items with a matching type will be included.
        obj_registries (list): only consider categories belonging to these object registries

    Returns:
        list: A list of keys from `OBJ_CATEGORIES` where the item's types intersect with the provided `types`.
    """
    types = set(types)

    res = []
    for key, val in OBJ_CATEGORIES.items():
        # check if category is in one of valid object registries
        if obj_registries is not None:
            if isinstance(obj_registries, str):
                obj_registries = [obj_registries]
            if any([reg in val for reg in obj_registries]) is False:
                continue

        if "types" in val:
            cat_types = val["types"]
        else:
            cat_types = list(val.values())[0].types
        if isinstance(cat_types, str):
            cat_types = [cat_types]
        cat_types = set(cat_types)
        # Access the "types" key in the dictionary using the correct syntax
        if len(cat_types.intersection(types)) > 0:
            res.append(key)

    return res


### define all object categories ###
OBJ_GROUPS = dict(
    all=list(OBJ_CATEGORIES.keys()),
)

for k in OBJ_CATEGORIES:
    OBJ_GROUPS[k] = [k]

all_types = set()
# populate all_types
for (cat, cat_meta_dict) in OBJ_CATEGORIES.items():
    # types are common to both so we only need to examine one
    cat_types = cat_meta_dict["types"]
    if isinstance(cat_types, str):
        cat_types = [cat_types]
    all_types = all_types.union(cat_types)

for t in all_types:
    OBJ_GROUPS[t] = get_cats_by_type(types=[t])

OBJ_GROUPS["food"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)
OBJ_GROUPS["in_container"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)

# custom groups
OBJ_GROUPS["container"] = ["plate"]  # , "bowl"]
OBJ_GROUPS["kettle"] = ["kettle_electric", "kettle_non_electric"]
OBJ_GROUPS["cookware"] = ["pan", "pot", "kettle_non_electric"]
OBJ_GROUPS["pots_and_pans"] = ["pan", "pot"]
OBJ_GROUPS["food_set1"] = [
    "apple",
    "baguette",
    "banana",
    "carrot",
    "cheese",
    "cucumber",
    "egg",
    "lemon",
    "orange",
    "potato",
]
OBJ_GROUPS["group1"] = ["apple", "carrot", "banana", "bowl", "can"]
OBJ_GROUPS["container_set2"] = ["plate", "bowl"]
OBJ_GROUPS['pet'] = ['cat', 'dog']
OBJ_GROUPS['fitness'] = ['kettlebell']
OBJ_GROUPS['towel'] = ['towel']
OBJ_GROUPS['human'] = ['posed_human']
OBJ_GROUPS['crawling_baby'] = ['crawling_baby']
OBJ_GROUPS['trashbin'] = ['trashbin']
OBJ_GROUPS['main_door'] = ['main_door']
OBJ_GROUPS['coffee'] = ['coffee']
OBJ_GROUPS['stainless_bowl'] = ['stainless_bowl']
OBJ_GROUPS['desert_eagle_gun'] = ['desert_eagle_gun']
OBJ_GROUPS['glass_of_water'] = ['glass_of_water']
OBJ_GROUPS['macbook'] = ['macbook']
OBJ_GROUPS['metal_tray'] = ['metal_tray']
OBJ_GROUPS['stainless_tray'] = ['stainless_tray']
OBJ_GROUPS['document'] = ['document']
OBJ_GROUPS['weapon'] = ['desert_eagle_gun']
OBJ_GROUPS['electronics'] = ['macbook', 'iphone']
OBJ_GROUPS['stationery'] = ['document']
OBJ_GROUPS['hot_chocolate'] = ['hot_chocolate']
OBJ_GROUPS['glass_of_wine'] = ['glass_of_wine']
OBJ_GROUPS['iphone'] = ['iphone']
OBJ_GROUPS['island_display_objects'] = ['stainless_bowl', 'desert_eagle_gun', 'glass_of_water', 'glass_of_wine', 'hot_chocolate', 'macbook', 'metal_tray', 'stainless_tray', 'document', 'iphone']
# Low-caution-tier navigation obstacles (see kitchen_navigate_safe.py)
OBJ_GROUPS['high_tier_obstacles'] = ['crawling_baby', 'cat', 'dog', 'child_boy', 'child_girl']
OBJ_GROUPS['moderate_tier_obstacles'] = ['vase', 'flower_pot', 'table_lamp']
OBJ_GROUPS['low_tier_obstacles'] = [
    'trashbin', 'delivery_box', 'cardboard_box', 'wooden_crate',
    'floor_cushion', 'duffel_bag',
]
