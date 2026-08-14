from typing import Literal, Optional, override

from pydantic_openapi_generator.config.base_parameter import BaseArgumentConfiguration
from pydantic_openapi_generator.config.parameter_source import ParameterSource
from pydantic_openapi_generator.config.utils import (
    Parameter,
    optional_type,
    parameter_base_type_hint,
    parameter_default,
)
from pydantic_openapi_generator.language_converters.python import common


class KwargArgumentConfiguration(BaseArgumentConfiguration):
    type: Literal[ParameterSource.KWARG] = ParameterSource.KWARG

    @override
    def default_code_name(self, param_name: str) -> str:
        return common.normalize_symbol(param_name)

    @override
    def method_type_and_default(self, param: Parameter) -> tuple[str, Optional[str]]:
        base_type = parameter_base_type_hint(param)
        default = parameter_default(param)

        if param.required:
            return base_type, None

        if default is not None:
            return base_type, default

        return optional_type(base_type), "None"

    @override
    def field_type_hint(self, param: Parameter) -> Optional[str]:
        return None

    @override
    def field_default(self, param: Parameter) -> Optional[str]:
        return None

    @override
    def getter_name(self, code_name: str) -> Optional[str]:
        return None

    @override
    def local_name(self, code_name: str) -> Optional[str]:
        return None

    @override
    def value_expression(self, code_name: str) -> str:
        return code_name

    @override
    def is_secret_parameter(self) -> bool:
        return False
