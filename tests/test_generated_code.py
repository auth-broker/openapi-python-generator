import asyncio
import importlib
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from pydantic_openapi_generator.common import HTTPLibrary
from pydantic_openapi_generator.generate_data import generate_data

from .conftest import test_data_path
from .conftest import test_result_path


def test_sync_client_access_token(model_data_with_cleanup):
    generate_data(test_data_path, test_result_path)

    from .test_result.clients.sync_client import SyncClient

    client = SyncClient()
    assert client.get_access_token() is None
    client.set_access_token("foo_bar")
    assert client.get_access_token() == "foo_bar"


def test_async_client_access_token(model_data_with_cleanup):
    generate_data(test_data_path, test_result_path)

    from .test_result.clients.async_client import AsyncClient

    async def check_client():
        client = AsyncClient()
        assert await client.get_access_token() is None
        await client.set_access_token("foo_bar")
        assert await client.get_access_token() == "foo_bar"

    asyncio.run(check_client())


@pytest.mark.parametrize(
    "library, use_orjson, openapi_version",
    [
        (HTTPLibrary.httpx, False, "3.0"),
        (HTTPLibrary.httpx, True, "3.0"),
        (HTTPLibrary.requests, False, "3.0"),
        (HTTPLibrary.aiohttp, False, "3.0"),
        (HTTPLibrary.httpx, False, "3.1"),
        (HTTPLibrary.httpx, True, "3.1"),
        (HTTPLibrary.requests, False, "3.1"),
        (HTTPLibrary.aiohttp, False, "3.1"),
    ],
)
def test_generate_code_structure(library, use_orjson, openapi_version):
    temp_dir, package_name = _generate_temp_package(
        library=library,
        use_orjson=use_orjson,
        openapi_version=openapi_version,
    )

    try:
        assert (temp_dir / "__init__.py").exists()
        assert (temp_dir / "models").exists()
        assert (temp_dir / "clients").exists()
        assert (temp_dir / "exceptions").exists()

        clients_dir = temp_dir / "clients"
        assert (clients_dir / "__init__.py").exists()
        assert (clients_dir / "sync_client.py").exists()
        assert (clients_dir / "async_client.py").exists()

        assert importlib.import_module(f"{package_name}.clients.sync_client").SyncClient
        assert importlib.import_module(f"{package_name}.clients.async_client").AsyncClient
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.respx(assert_all_called=False, assert_all_mocked=False)
@pytest.mark.parametrize("async_client", [False, True])
def test_httpx_client_e2e(async_client, respx_mock):
    temp_dir, package_name = _generate_temp_package(
        library=HTTPLibrary.httpx,
        use_orjson=False,
        openapi_version="3.0",
    )

    try:
        models_module = importlib.import_module(f"{package_name}.models")
        base_url = "http://localhost:5000"
        _setup_httpx_mocks(respx_mock, base_url)

        if async_client:
            client_module = importlib.import_module(f"{package_name}.clients.async_client")
            client = client_module.AsyncClient(base_url=base_url)
            asyncio.run(_run_async_client_tests(client, models_module))
        else:
            client_module = importlib.import_module(f"{package_name}.clients.sync_client")
            client = client_module.SyncClient(base_url=base_url)
            _run_sync_client_tests(client, models_module)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _generate_temp_package(
    *,
    library: HTTPLibrary,
    use_orjson: bool,
    openapi_version: str,
) -> tuple[Path, str]:
    test_data_folder = Path(__file__).parent / "test_data"
    spec_31 = test_data_folder / "test_api_31.json"
    spec_file = spec_31 if openapi_version == "3.1" and spec_31.exists() else test_data_path

    package_name = f"test_result_{library.value}_{use_orjson}_{openapi_version}".replace(".", "_")
    temp_dir = Path(tempfile.gettempdir()) / package_name

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    generate_data(spec_file, temp_dir, library, use_orjson=use_orjson)
    sys.path.insert(0, str(temp_dir.parent))

    return temp_dir, package_name


def _setup_httpx_mocks(respx_mock, base_url):
    respx_mock.get(f"{base_url}/").mock(
        return_value=httpx.Response(200, json={"message": "Hello World"})
    )

    respx_mock.get(f"{base_url}/users").mock(
        return_value=httpx.Response(
            200,
            json=[
                dict(
                    id=1,
                    username="user1",
                    email="x@y.com",
                    password="123456",
                    is_active=True,
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
                dict(
                    id=2,
                    username="user2",
                    email="x@y.com",
                    password="123456",
                    is_active=True,
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
            ],
        )
    )

    respx_mock.get(f"{base_url}/teams").mock(
        return_value=httpx.Response(
            200,
            json=[
                dict(
                    id=1,
                    name="team1",
                    description="team1",
                    is_active=True,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                ),
                dict(
                    id=2,
                    name="team2",
                    description="team2",
                    is_active=True,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                ),
            ],
        )
    )


def _run_sync_client_tests(client, models_module):
    root = client.root__get()
    assert isinstance(root, models_module.RootResponse)

    users = client.get_users_users_get()
    assert isinstance(users, list)
    assert isinstance(users[0], models_module.User)
    assert isinstance(users[1], models_module.User)

    teams = client.get_teams_teams_get()
    assert isinstance(teams, list)


async def _run_async_client_tests(client, models_module):
    root = await client.root__get()
    assert isinstance(root, models_module.RootResponse)

    users = await client.get_users_users_get()
    assert isinstance(users, list)
    assert isinstance(users[0], models_module.User)
    assert isinstance(users[1], models_module.User)

    teams = await client.get_teams_teams_get()
    assert isinstance(teams, list)
