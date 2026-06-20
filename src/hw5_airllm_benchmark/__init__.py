"""
Top-level package for hw5_airllm_benchmark. (GUIDE §14.2)

Exports the public API and defines the package version so that any
importer can do:
    from hw5_airllm_benchmark import HW5SDK, __version__
"""

from .sdk.sdk import HW5SDK
from .shared.version import __version__

__all__ = ["HW5SDK", "__version__"]
