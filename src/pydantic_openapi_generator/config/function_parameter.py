from typing import Literal, Optional, override

from pydantic_openapi_generator.config.base_parameter import (
    BaseArgumentConfiguration,
)
from pydantic_openapi_generator.config.parameter_source import ParameterSource
from pydantic_openapi_generator.config.utils import (
    Parameter,
    lower_code_name,
    optional_override_type_and_default,
)


class FunctionArgumentConfiguration(BaseArgumentConfiguration):
    type: Literal[ParameterSource.FUNCTION] = ParameterSource.FUNCTION

    @override
    def default_code_name(self, param_name: str) -> str:
        return lower_code_name(param_name)

    @override
    def method_type_and_default(self, param: Parameter) -> tuple[str, Optional[str]]:
        return optional_override_type_and_default(param)

    @override
    def field_type_hint(self, param: Parameter) -> Optional[str]:
        return None

    @override
    def field_default(self, param: Parameter) -> Optional[str]:
        return None

    @override
    def getter_name(self, code_name: str) -> str:
        return f"get_{code_name}"

    @override
    def local_name(self, code_name: str) -> str:
        return f"_{code_name}"

    @override
    def value_expression(self, code_name: str) -> str:
        return self.local_name(code_name)

    @override
    def is_secret_parameter(self) -> bool:
        return False
