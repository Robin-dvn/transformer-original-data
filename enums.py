"""
This module defines the main enumerations used in the project, including observation types, zones, fruit states, and leaf states.
"""

from enum import Enum, IntEnum


class Observation(str, Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    SYLLEPTIC_SMALL = "SYLLEPTIC_SMALL"
    SYLLEPTIC_MEDIUM = "SYLLEPTIC_MEDIUM"
    SYLLEPTIC_LARGE = "SYLLEPTIC_LARGE"
    DORMANT = "DORMANT"
    FLORAL = "FLORAL"
    TRUNK = "TRUNK"
    NEW_SHOOT = "NEW_SHOOT"


