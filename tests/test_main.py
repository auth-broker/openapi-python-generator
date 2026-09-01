"""Test cases for the __main__ module."""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pydantic_openapi_generator.__main__ import main
from pydantic_openapi_generator.common import HTTPLibrary
from tests.conftest import test_data_path
from tests.conftest import test_result_path
from tests.test_client_generator_contracts import CONTRACT_SPEC


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.mark.parametrize(
    "library",
    [HTTPLibrary.httpx, HTTPLibrary.requests, HTTPLibrary.aiohttp],
)
def test_main_succeeds(runner: CliRunner, model_data_with_cleanup, library) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(
        main,
        [str(test_data_path), str(test_result_path), "--library", library.value],
    )
    assert result.exit_code == 0


def test_main_accepts_config_path(runner: CliRunner, tmp_path) -> None:
    spec_path = tmp_path / "openapi.json"
    output_path = tmp_path / "generated"
    config_path = tmp_path / "generator.yaml"
    spec_path.write_text(json.dumps(CONTRACT_SPEC))
    config_path.write_text("""
parameters:
  - name: X-Test-Header
    type: ClassVar
    code_name: test_header
""")

    result = runner.invoke(
        main,
        [
            str(spec_path),
            str(output_path),
            "--config-path",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert (output_path / "clients" / "sync_client.py").exists()


def test_main_accepts_multiple_overlays_in_order(runner: CliRunner, tmp_path) -> None:
    source_path = tmp_path / "openapi.json"
    output_path = tmp_path / "generated"
    first_overlay_path = tmp_path / "first.yaml"
    second_overlay_path = tmp_path / "second.json"
    source_path.write_text(json.dumps(CONTRACT_SPEC))
    first_overlay_path.write_text("components: {}")
    second_overlay_path.write_text("{}")

    with patch("pydantic_openapi_generator.__main__.generate_data") as generate_data_mock:
        result = runner.invoke(
            main,
            [
                str(source_path),
                str(output_path),
                "--overlay",
                str(first_overlay_path),
                "--overlay",
                str(second_overlay_path),
            ],
        )

    assert result.exit_code == 0
    assert generate_data_mock.call_args.args[-1] == (str(first_overlay_path), str(second_overlay_path))
