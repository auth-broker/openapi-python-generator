from pathlib import Path
from typing import Optional

from pydantic_di.loaders import ObjectLoaderYaml

from pydantic_openapi_generator.config.generator_config import (
    PydanticOpenAPIGeneratorConfig,
)


def load_config(config_path: Optional[str | Path]) -> PydanticOpenAPIGeneratorConfig:
    if config_path is None:
        return PydanticOpenAPIGeneratorConfig()

    return ObjectLoaderYaml[PydanticOpenAPIGeneratorConfig](
        path=Path(config_path),
    ).load()
