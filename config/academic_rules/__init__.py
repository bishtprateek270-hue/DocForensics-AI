"""
DocForensics AI — Academic Rules Registry & Loader
Loads institution-specific grading, marks, and SGPA calculation rules.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

RULES_DIR = Path(__file__).resolve().parent

_LOADED_RULES: Dict[str, Dict[str, Any]] = {}


def load_academic_rule(scheme_id: str = "standard_ugc_10point") -> Optional[Dict[str, Any]]:
    """Loads an academic grading and calculation rule definition."""
    if scheme_id in _LOADED_RULES:
        return _LOADED_RULES[scheme_id]

    file_path = RULES_DIR / f"{scheme_id}.json"
    if not file_path.exists():
        # Fallback to standard
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
