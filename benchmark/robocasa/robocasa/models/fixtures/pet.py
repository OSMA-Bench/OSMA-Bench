import numpy as np
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import string_to_array as s2a

from robocasa.models.fixtures import Fixture

class Dog(Fixture):
    """A dog fixture."""

    MODEL_NAME = "dog"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def nat_lang(self):
        return "dog"
    
class Cat(Fixture):
    """A cat fixture."""

    MODEL_NAME = "cat"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def nat_lang(self):
        return "cat"