from enum import Enum


class Direction(Enum):
    POSITIVE = 1
    NEGATIVE = -1


class Dithering(Enum):
    """
    Enum class to represent
    the different type of dithering.
    """

    FSdithering = "FS"  # Floyd-Steinberg
    JJN = "JJN"  # Jarvis, Judice, Ninke
