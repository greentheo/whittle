"""Objectives. Lower is always better."""

from .aero import DragCoefficient
from .base import Candidate, Objective, WeightedSum
from .mass import MaterialFraction, MeshVolume
from .structural import AnchorConnectivity, Compliance

__all__ = [
    "AnchorConnectivity",
    "Candidate",
    "Compliance",
    "DragCoefficient",
    "MaterialFraction",
    "MeshVolume",
    "Objective",
    "WeightedSum",
]
