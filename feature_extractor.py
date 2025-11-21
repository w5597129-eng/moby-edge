"""
Compatibility wrapper for the project's feature extractor.

Some scripts or external tooling may expect `feature_extractor` to be
importable from the repository root. This wrapper re-exports the main
functions from `src/feature_extractor.py` so both `import feature_extractor`
and `from src.feature_extractor import ...` usages are supported.

If `src` cannot be imported as a package, the wrapper will attempt to load
the implementation file by path as a fallback.
"""
from __future__ import annotations

import os
import importlib
import importlib.util
from types import ModuleType
from typing import Optional


def _load_impl() -> Optional[ModuleType]:
    # Prefer package-style import if available
    try:
        return importlib.import_module("src.feature_extractor")
    except Exception:
        # Fallback: load by file path relative to repo root
        impl_path = os.path.join(os.path.dirname(__file__), "src", "feature_extractor.py")
        if not os.path.exists(impl_path):
            return None
        spec = importlib.util.spec_from_file_location("feature_extractor_impl", impl_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_impl = _load_impl()

if _impl is not None:
    # Re-export commonly used functions if present
    for attr in (
        "extract_features_v17",
        "process_multi_sensor_files_v17",
        "extract_features",
    ):
        if hasattr(_impl, attr):
            globals()[attr] = getattr(_impl, attr)

__all__ = [n for n in ("extract_features_v17", "process_multi_sensor_files_v17", "extract_features") if n in globals()]
