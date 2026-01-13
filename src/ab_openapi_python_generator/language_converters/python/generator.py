from typing import Optional, Union

from openapi_pydantic.v3.v3_0 import OpenAPI as OpenAPI30
from openapi_pydantic.v3.v3_1 import OpenAPI as OpenAPI31

from ab_openapi_python_generator.common import PydanticVersion
from ab_openapi_python_generator.language_converters.python import common
from ab_openapi_python_generator.language_converters.python.model_generator import (
    generate_models,
)
from ab_openapi_python_generator.language_converters.python.service_generator import (
    generate_clients,
)
from ab_openapi_python_generator.models import ConversionResult, LibraryConfig

# Type alias for both OpenAPI versions
OpenAPISpec = Union[OpenAPI30, OpenAPI31]


def generator(
    data: OpenAPISpec,
    library_config: LibraryConfig,
    env_token_name: Optional[str] = None,
    use_orjson: bool = False,
    custom_template_path: Optional[str] = None,
    pydantic_version: PydanticVersion = PydanticVersion.V2,
) -> ConversionResult:
    """
    Generate Python code from an OpenAPI 3.0+ specification.
    """

    common.set_use_orjson(use_orjson)
    common.set_custom_template_path(custom_template_path)

    if data.components is not None:
        models = generate_models(data.components, pydantic_version)
    else:
        models = []

    if data.paths is not None:
        clients = generate_clients(data, data.paths, library_config, env_token_name, pydantic_version)
    else:
        clients = []

    return ConversionResult(
        models=models,
        services=[],          # keep field if your writer expects it
        api_config=None,      # keep field if your writer expects it
        clients=clients,
    )
