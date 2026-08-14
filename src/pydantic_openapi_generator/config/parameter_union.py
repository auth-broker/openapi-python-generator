from typing import Annotated

from pydantic import Field

from pydantic_openapi_generator.config.class_var_parameter import (
    ClassVarArgumentConfiguration,
)
from pydantic_openapi_generator.config.function_parameter import (
    FunctionArgumentConfiguration,
)
from pydantic_openapi_generator.config.kwarg_parameter import KwargArgumentConfiguration

ArgumentConfiguration = Annotated[
    KwargArgumentConfiguration | ClassVarArgumentConfiguration | FunctionArgumentConfiguration,
    Field(discriminator="type"),
]
