from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_di.loaders import ObjectLoaderYaml


class BaseArgumentConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    code_name: Optional[str] = None


class KwargArgumentConfiguration(BaseArgumentConfiguration):
    type: Literal["Kwarg"] = "Kwarg"


class ClassVarArgumentConfiguration(BaseArgumentConfiguration):
    type: Literal["ClassVar"] = "ClassVar"
    is_secret: bool = False


class FunctionArgumentConfiguration(BaseArgumentConfiguration):
    type: Literal["Function"] = "Function"


ArgumentConfiguration = Annotated[
    KwargArgumentConfiguration
    | ClassVarArgumentConfiguration
    | FunctionArgumentConfiguration,
    Field(discriminator="type"),
]


class PydanticOpenAPIGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: list[ArgumentConfiguration] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_parameter_names(self) -> "PydanticOpenAPIGeneratorConfig":
        names: set[str] = set()
        duplicates: set[str] = set()
        for parameter in self.parameters:
            if parameter.name in names:
                duplicates.add(parameter.name)
            names.add(parameter.name)

        if duplicates:
            duplicate_list = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate parameter configuration for: {duplicate_list}")

        return self


def load_config(config_path: Optional[str | Path]) -> PydanticOpenAPIGeneratorConfig:
    if config_path is None:
        return PydanticOpenAPIGeneratorConfig()

    return ObjectLoaderYaml[PydanticOpenAPIGeneratorConfig](
        path=Path(config_path),
    ).load()
