"""Utilities shared by research experiments."""

import subprocess
import uuid

from datetime import datetime, timezone
from pathlib import Path


def experiment_output(experiment_id: str) -> Path:
    return (
        Path("results")
        / experiment_id
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    )


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
