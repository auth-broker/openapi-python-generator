from enum import StrEnum


class ParameterSource(StrEnum):
    KWARG = "Kwarg"
    CLASS_VAR = "ClassVar"
    FUNCTION = "Function"
