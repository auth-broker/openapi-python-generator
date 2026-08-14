from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, ConfigDict

from pydantic_openapi_generator.config.utils import Parameter, parameter_base_type_hint
from pydantic_openapi_generator.language_converters.python import common
from pydantic_openapi_generator.models import GeneratedParameter


class BaseArgumentConfiguration(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    name: str
    code_name: Optional[str] = None

    def resolve_parameter(self, param: Parameter) -> GeneratedParameter:
        code_name = self.resolved_code_name(param)
        type_hint, default = self.method_type_and_default(param)

        return GeneratedParameter(
            wire_name=param.name,
            code_name=code_name,
            location=param.param_in,  # type: ignore[arg-type]
            type_hint=type_hint,
            base_type_hint=parameter_base_type_hint(param),
            required=param.required,
            default=default,
            source=self.type,  # type: ignore[attr-defined]
            is_secret=self.is_secret_parameter(),
            field_type_hint=self.field_type_hint(param),
            field_default=self.field_default(param),
            getter_name=self.getter_name(code_name),
            local_name=self.local_name(code_name),
            value_expression=self.value_expression(code_name),
        )

    def resolved_code_name(self, param: Parameter) -> str:
        if self.code_name:
            return common.normalize_symbol(self.code_name)
        return self.default_code_name(param.name)

    @abstractmethod
    def default_code_name(self, param_name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def method_type_and_default(self, param: Parameter) -> tuple[str, Optional[str]]:
        raise NotImplementedError

    @abstractmethod
    def field_type_hint(self, param: Parameter) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def field_default(self, param: Parameter) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def getter_name(self, code_name: str) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def local_name(self, code_name: str) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def value_expression(self, code_name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_secret_parameter(self) -> bool:
        raise NotImplementedError
