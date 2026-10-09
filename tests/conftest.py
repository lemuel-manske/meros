from pathlib import Path

import pytest

from tests.support import Scenario, create_scenario


@pytest.fixture
def scenario(tmp_path: Path) -> Scenario:
    return create_scenario(tmp_path / "data")
