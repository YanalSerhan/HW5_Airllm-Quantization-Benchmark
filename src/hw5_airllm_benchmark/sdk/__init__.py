"""
Public interface for the sdk sub-package.
"""

from ..shared.version import __version__
from .sdk import HW5SDK

__all__ = ["HW5SDK", "__version__"]
