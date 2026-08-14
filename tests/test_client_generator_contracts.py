import asyncio
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from pydantic_openapi_generator.common import HTTPLibrary
from pydantic_openapi_generator.generate_data import generate_data

CONTRACT_SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Contract Test API", "version": "1.0.0"},
    "servers": [{"url": "http://testserver"}],
    "paths": {
        "/things": {
            "get": {
                "operationId": "getThing",
                "parameters": [
                    {
                        "name": "X-Test-Header",
                        "in": "header",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "X-Optional-Header",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string", "default": "AU"},
                    },
                    {
                        "name": "brand",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string", "default": "NRMA"},
                    },
                ],
                "responses": {
                    "200": {
                        "description": "Thing",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ThingResponse"}
                            }
                        },
                    },
                    "206": {
                        "description": "Validation",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ValidationResponse"
                                }
                            }
                        },
                    },
                },
            }
        },
        "/upload": {
            "post": {
                "operationId": "uploadDocument",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string", "format": "binary"},
                                    "requestMetadata": {
                                        "type": "string",
                                        "format": "binary",
                                    },
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Uploaded",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/UploadResponse"
                                }
                            }
                        },
                    },
                    "202": {
                        "description": "Accepted",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/UploadResponse"
                                }
                            }
                        },
                    },
                },
            }
        },
        "/download": {
            "get": {
                "operationId": "downloadDocument",
                "responses": {
                    "200": {
                        "description": "PDF",
                        "content": {
                            "application/pdf": {
                                "schema": {"type": "string", "format": "binary"}
                            }
                        },
                    }
                },
            }
        },
        "/empty": {
            "delete": {
                "operationId": "deleteThing",
                "responses": {"204": {"description": "Deleted"}},
            }
        },
    },
    "components": {
        "schemas": {
            "ThingResponse": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string"}},
            },
            "ValidationResponse": {
                "type": "object",
                "required": ["message"],
                "properties": {"message": {"type": "string"}},
            },
            "UploadResponse": {
                "type": "object",
                "required": ["documentId"],
                "properties": {"documentId": {"type": "string"}},
            },
        }
    },
}


@pytest.fixture
def generated_contract_package():
    package_name = f"contract_result_{uuid4().hex}"
    temp_dir = Path(tempfile.gettempdir()) / package_name
    spec_path = temp_dir.parent / f"{package_name}.json"
    temp_dir.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(CONTRACT_SPEC))
    sys.path.insert(0, str(temp_dir.parent))

    try:
        generate_data(spec_path, temp_dir, HTTPLibrary.httpx)
        yield package_name
    finally:
        if str(temp_dir.parent) in sys.path:
            sys.path.remove(str(temp_dir.parent))
        shutil.rmtree(temp_dir, ignore_errors=True)
        spec_path.unlink(missing_ok=True)
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(
                f"{package_name}."
            ):
                sys.modules.pop(module_name, None)


@pytest.mark.respx(assert_all_called=False, assert_all_mocked=True)
@pytest.mark.parametrize("async_client", [False, True])
def test_generated_clients_honor_openapi_request_response_contracts(
    generated_contract_package,
    respx_mock,
    async_client,
):
    requests: list[httpx.Request] = []

    def thing_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(206, json={"message": "partial"})

    def upload_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"documentId": "doc-123"})

    def download_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=b"%PDF-1.4",
            headers={"content-type": "application/pdf"},
        )

    respx_mock.get("http://testserver/things").mock(side_effect=thing_handler)
    respx_mock.post("http://testserver/upload").mock(side_effect=upload_handler)
    respx_mock.get("http://testserver/download").mock(side_effect=download_handler)
    respx_mock.delete("http://testserver/empty").mock(return_value=httpx.Response(204))

    models = importlib.import_module(f"{generated_contract_package}.models")

    if async_client:
        client_module = importlib.import_module(
            f"{generated_contract_package}.clients.async_client"
        )
        client = client_module.AsyncClient()
        thing, upload, download, empty = asyncio.run(_run_async_contract(client))
    else:
        client_module = importlib.import_module(
            f"{generated_contract_package}.clients.sync_client"
        )
        client = client_module.SyncClient()
        thing = client.getThing(X_Test_Header="required")
        upload = client.uploadDocument(
            data={"file": ("doc.txt", b"hello", "text/plain")}
        )
        download = client.downloadDocument()
        empty = client.deleteThing()

    assert isinstance(thing, models.ValidationResponse)
    assert upload.documentId == "doc-123"
    assert download == b"%PDF-1.4"
    assert empty is None

    thing_request = requests[0]
    assert thing_request.headers["X-Test-Header"] == "required"
    assert thing_request.headers["X-Optional-Header"] == "AU"
    assert thing_request.url.params["brand"] == "NRMA"

    upload_request = requests[1]
    assert upload_request.headers["content-type"].startswith("multipart/form-data")
    assert b'form-data; name="file"; filename="doc.txt"' in upload_request.content
    assert b"hello" in upload_request.content

    download_request = requests[2]
    assert download_request.headers["accept"] == "application/pdf"


async def _run_async_contract(client):
    thing = await client.getThing(X_Test_Header="required")
    upload = await client.uploadDocument(
        data={"file": ("doc.txt", b"hello", "text/plain")}
    )
    download = await client.downloadDocument()
    empty = await client.deleteThing()
    return thing, upload, download, empty
