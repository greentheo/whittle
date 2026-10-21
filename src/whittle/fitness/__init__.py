"""Objectives. Lower is always better."""

from .aero import DragArea, DragCoefficient
from .base import Candidate, Objective, WeightedSum
from .mass import MaterialFraction, MeshVolume
from .shape import KeepOut, NoFolding, StayInside, VolumeBudget
from .structural import AnchorConnectivity, Compliance

__all__ = [
    "AnchorConnectivity",
    "Candidate",
    "Compliance",
    "DragArea",
    "DragCoefficient",
    "KeepOut",
    "MaterialFraction",
    "MeshVolume",
    "NoFolding",
    "Objective",
    "StayInside",
    "VolumeBudget",
    "WeightedSum",
]
