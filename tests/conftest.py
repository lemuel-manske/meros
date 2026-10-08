from pathlib import Path

import pytest

from tests.support import ProjectFixture, create_project


@pytest.fixture
def project_fixture(tmp_path: Path) -> ProjectFixture:
    return create_project(tmp_path / "data")


@pytest.fixture
def prepared_project(project_fixture: ProjectFixture) -> ProjectFixture:
    project_fixture.prepare()

    return project_fixture
