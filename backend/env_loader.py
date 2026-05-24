from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_dotenv() -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        base_dir / ".env",
        base_dir.parent / ".env",
    ]

    loaded: list[Path] = []
    for path in dict.fromkeys(candidates):
        if not path.exists() or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            parsed = _parse_env_line(line)
            if not parsed:
                continue
            key, value = parsed
            if not os.environ.get(key):
                os.environ[key] = value
        loaded.append(path)
    return loaded
