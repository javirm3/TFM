from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_imports() -> None:
    notebooks_dir = Path(__file__).resolve().parent
    code_dir = notebooks_dir.parent
    glmhmmt_src = code_dir / "glmhmmt" / "src"

    for candidate in (notebooks_dir, glmhmmt_src):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


ensure_repo_imports()
