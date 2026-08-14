from typing import List, Literal, Optional, Union

from openapi_pydantic.v3.v3_0 import (
    Operation as Operation30,
)
from openapi_pydantic.v3.v3_0 import (
    PathItem as PathItem30,
)
from openapi_pydantic.v3.v3_0 import (
    Schema as Schema30,
)
from openapi_pydantic.v3.v3_1 import (
    Operation as Operation31,
)
from openapi_pydantic.v3.v3_1 import (
    PathItem as PathItem31,
)
from openapi_pydantic.v3.v3_1 import (
    Schema as Schema31,
)
from pydantic import BaseModel, Field

# Type unions for compatibility with both OpenAPI 3.0 and 3.1
Operation = Union[Operation30, Operation31]
PathItem = Union[PathItem30, PathItem31]
Schema = Union[Schema30, Schema31]


class LibraryConfig(BaseModel):
    name: str
    library_name: str
    template_name: str
    include_async: bool
    include_sync: bool


class TypeConversion(BaseModel):
    original_type: str
    converted_type: str
    import_types: Optional[List[str]] = None


class OpReturnType(BaseModel):
    type: Optional[TypeConversion] = None
    status_code: int
    complex_type: bool = False
    list_type: Optional[str] = None
    variants: List["ResponseVariant"] = Field(default_factory=list)
    accepted_status_codes: List[int] = Field(default_factory=list)
    accept_content_types: List[str] = Field(default_factory=list)
    return_type_hint: Optional[str] = None


class ResponseVariant(BaseModel):
    status_code: int
    content_type: Optional[str] = None
    type: Optional[TypeConversion] = None
    complex_type: bool = False
    list_type: Optional[str] = None
    body_kind: Literal["empty", "json", "text", "binary"] = "empty"


class RequestBodyDefinition(BaseModel):
    content_type: Optional[str] = None
    encoding: Literal["json", "form", "multipart", "binary", "text"]
    expression: str


class ServiceOperation(BaseModel):
    params: str
    operation_id: str
    query_params: List[str]
    header_params: List[str]
    return_type: OpReturnType
    operation: Operation
    pathItem: PathItem
    content: str
    async_client: Optional[bool] = False
    tag: Optional[str] = None
    path_name: str
    body_param: Optional[str] = None
    request_body: Optional[RequestBodyDefinition] = None
    method: str
    is_sse: bool = False
    use_orjson: bool = False


class Property(BaseModel):
    name: str
    type: TypeConversion
    required: bool
    default: Optional[str]
    import_type: Optional[List[str]] = None


class Model(BaseModel):
    file_name: str
    content: str
    openapi_object: Optional[Schema] = None
    properties: List[Property] = []


class ConversionResult(BaseModel):
    models: List[Model] = Field(default_factory=list)
    clients: List[Model] = Field(default_factory=list)
    exceptions: List[Model] = Field(default_factory=list)
