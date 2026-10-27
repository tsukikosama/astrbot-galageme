"""Allow source tests to run even when a Git checkout has a different folder name."""

import importlib.util
import sys
from pathlib import Path


def ensure_package():
    name = "astrbot_plugin_relationship_memory"
    if name in sys.modules:
        return
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
