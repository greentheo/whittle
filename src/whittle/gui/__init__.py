"""A local web GUI: load a part, click to place loads or pick the wind, slide.

    python -m whittle.gui

See :mod:`whittle.gui.server`.
"""

from .server import App, main, serve

__all__ = ["App", "main", "serve"]
