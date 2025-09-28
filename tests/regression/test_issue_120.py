import pytest
from click.testing import CliRunner

from openapi_python_generator.common import HTTPLibrary, library_config_dict
from openapi_python_generator.generate_data import generate_data
from tests.conftest import test_data_folder
from tests.conftest import test_result_path


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.mark.parametrize(
    "library",
    [HTTPLibrary.httpx, HTTPLibrary.aiohttp, HTTPLibrary.requests],
)
def test_issue_120(runner: CliRunner, model_data_with_cleanup, library) -> None:
    """
    https://github.com/MarcoMuellner/openapi-python-generator/issues/120
    """
    generate_data(test_data_folder / "issue_120.json", test_result_path, library)

    if library_config_dict[library].include_sync:
        assert (test_result_path / "services" / "default_service.py").exists()
        assert (test_result_path / "services" / "default_service.py").read_text().find(
            "language: Optional[str] = None"
        ) != -1

    if library_config_dict[library].include_async:
        assert (test_result_path / "services" / "async_default_service.py").exists()
        assert (
            test_result_path / "services" / "async_default_service.py"
        ).read_text().find("language: Optional[str] = None") != -1
