"""Environment checks.

On some Windows machines Application Control blocks pandas' compiled libraries, and SHAP cannot load without pandas.
The screening models themselves only need numpy and scikit-learn, so the app asks these helpers what is available and
degrades gracefully (for example by explaining a result with feature occlusion instead of SHAP).
"""

from __future__ import annotations

import importlib.abc
import sys
from functools import lru_cache


class _PandasBlocker(importlib.abc.MetaPathFinder):
    """Makes ``import pandas`` fail immediately and cleanly, instead of re-trying a blocked DLL each time."""

    def find_spec(self, name, path, target=None):
        if name == "pandas" or name.startswith("pandas."):
            raise ModuleNotFoundError("pandas is not usable on this machine (blocked by Windows Application Control)",
                                      name=name)
        return None


@lru_cache(maxsize=1)
def pandas_available() -> bool:
    try:
        import pandas  # noqa: F401
        return True
    except Exception:  # ImportError, or an OSError raised while loading a blocked DLL
        for name in [m for m in sys.modules if m == "pandas" or m.startswith("pandas.")]:
            del sys.modules[name]  # scikit-learn indexes sys.modules["pandas"] directly, so no half-loaded entry may stay
        if not any(isinstance(f, _PandasBlocker) for f in sys.meta_path):
            sys.meta_path.insert(0, _PandasBlocker())
        return False


@lru_cache(maxsize=1)
def shap_available() -> bool:
    if not pandas_available():
        return False
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False
