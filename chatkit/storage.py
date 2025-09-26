import json
from typing import List, Dict, Any
from pathlib import Path

def save_history(path: str, history: List[Dict[str, str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict) and "role" in m and "content" in m]
    return []

def save_user_data(path: str, user_data: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)

def load_user_data(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
