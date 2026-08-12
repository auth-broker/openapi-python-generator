from __future__ import annotations

from pydantic_openapi_generator.language_converters.python.jinja_config import (
    HTTP_EXCEPTION_TEMPLATE,
    create_jinja_env,
)
from pydantic_openapi_generator.models import Model


def generate_exceptions() -> list[Model]:
    """
    Generate shared exception modules (package-local support code).
    """
    jinja_env = create_jinja_env()

    http_exception = Model(
        file_name="http_exception",
        content=jinja_env.get_template(HTTP_EXCEPTION_TEMPLATE).render(),
        openapi_object=None,  # Model.openapi_object is optional now
        properties=[],
    )

    return [http_exception]
