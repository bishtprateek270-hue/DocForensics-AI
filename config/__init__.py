"""
DocForensics AI — Configuration Package & Academic Rules Registry
Re-exports root hardware/path configurations and provides institutional rules loading.
"""

import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

# 1. Re-export all root config.py symbols (BASE_DIR, CHECKPOINT_DIR, cfg, get_target_device, etc.)
_root_config_path = Path(__file__).resolve().parent.parent / "config.py"
if _root_config_path.exists():
    _spec = importlib.util.spec_from_file_location("_root_config", str(_root_config_path))
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        for _attr in dir(_mod):
            if not _attr.startswith("__"):
                globals()[_attr] = getattr(_mod, _attr)

# 2. Institutional Academic Rules Registry
RULES_DIR = Path(__file__).resolve().parent / "academic_rules"
if not RULES_DIR.exists():
    RULES_DIR = Path(__file__).resolve().parent

_LOADED_RULES: Dict[str, Dict[str, Any]] = {}


def load_academic_rule(scheme_id: str = "standard_ugc_10point") -> Optional[Dict[str, Any]]:
    """Loads an academic grading and calculation rule definition."""
    if scheme_id in _LOADED_RULES:
        return _LOADED_RULES[scheme_id]

    file_path = RULES_DIR / f"{scheme_id}.json"
    if not file_path.exists():
        file_path = RULES_DIR / "standard_ugc_10point.json"
        if not file_path.exists():
            return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _LOADED_RULES[scheme_id] = data
            return data
    except Exception:
        return None


def list_available_rules() -> Dict[str, str]:
    """Lists all available academic rule schemes."""
    schemes = {}
    for f in RULES_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                schemes[data.get("scheme_id", f.stem)] = data.get("name", f.stem)
        except Exception:
            continue
    return schemes

