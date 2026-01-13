from __future__ import annotations

import itertools
import re
from typing import Dict, List, Optional, Set, Tuple, Union

import click
from openapi_pydantic.v3.v3_0 import (
    Components as Components30,
)
from openapi_pydantic.v3.v3_0 import (
    Reference as Reference30,
)
from openapi_pydantic.v3.v3_0 import (
    Schema as Schema30,
)
from openapi_pydantic.v3.v3_1 import (
    Components as Components31,
)
from openapi_pydantic.v3.v3_1 import (
    Reference as Reference31,
)
from openapi_pydantic.v3.v3_1 import (
    Schema as Schema31,
)

from openapi_python_generator.common import PydanticVersion
from openapi_python_generator.language_converters.python import common
from openapi_python_generator.language_converters.python.jinja_config import (
    ENUM_TEMPLATE,
    MODELS_TEMPLATE,
    MODELS_TEMPLATE_PYDANTIC_V2,
    ALIAS_UNION_TEMPLATE,
    create_jinja_env,
)
from openapi_python_generator.models import Model, Property, TypeConversion

# Type aliases for compatibility
Schema = Union[Schema30, Schema31]
Reference = Union[Reference30, Reference31]
Components = Union[Components30, Components31]


def _get_discriminator_key(schema: Schema) -> Optional[str]:
    """
    Return discriminator property name if present on the schema.
    openapi-pydantic v3.x uses `schema.discriminator.propertyName` (common),
    but we defensively check a couple of variants.
    """
    disc = getattr(schema, "discriminator", None)
    if disc is None:
        return None

    # Most common: propertyName
    key = getattr(disc, "propertyName", None)
    if key:
        return str(key)

    # Defensive fallbacks
    key = getattr(disc, "property_name", None)
    if key:
        return str(key)

    return None


def _schema_is_union(schema: Schema) -> bool:
    used = schema.oneOf if schema.oneOf is not None else schema.anyOf
    return used is not None and len(used) > 0


def _alias_name_for_property(prop_name: str) -> str:
    # token_issuer -> TokenIssuer, foo-bar -> FooBar, etc.
    parts = re.split(r"[^a-zA-Z0-9]+", prop_name.strip())
    parts = [p for p in parts if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _dedupe_imports(imports: Optional[List[str]]) -> List[str]:
    if not imports:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for imp in imports:
        if imp and imp not in seen:
            seen.add(imp)
            out.append(imp)
    return out


def _render_union_alias_module(
    *,
    jinja_env,
    alias_name: str,
    union_type: str,
    discriminator_key: Optional[str],
    member_imports: List[str],
) -> str:
    return jinja_env.get_template(ALIAS_UNION_TEMPLATE).render(
        alias_name=alias_name,
        union_type=union_type,
        discriminator_key=discriminator_key,
        member_imports=_dedupe_imports(member_imports),
    )


def type_converter(  # noqa: C901
    schema: Union[Schema, Reference],
    required: bool = False,
    model_name: Optional[str] = None,
) -> TypeConversion:
    """
    Converts an OpenAPI type to a Python type.
    :param schema: Schema or Reference containing the type to be converted
    :param model_name: Name of the original model on which the type is defined
    :param required: Flag indicating if the type is required by the class
    :return: The converted type
    """
    # Handle Reference objects by converting them to type references
    if isinstance(schema, Reference30) or isinstance(schema, Reference31):
        import_type = common.normalize_symbol(schema.ref.split("/")[-1])
        if required:
            converted_type = import_type
        else:
            converted_type = f"Optional[{import_type}]"

        return TypeConversion(
            original_type=schema.ref,
            converted_type=converted_type,
            import_types=(
                [f"from .{import_type} import {import_type}"]
                if import_type != model_name
                else None
            ),
        )

    if required:
        pre_type = ""
        post_type = ""
    else:
        pre_type = "Optional["
        post_type = "]"

    original_type = (
        schema.type.value
        if hasattr(schema.type, "value") and schema.type is not None
        else str(schema.type) if schema.type is not None else "object"
    )
    import_types: Optional[List[str]] = None

    if schema.allOf is not None:
        conversions = []
        for sub_schema in schema.allOf:
            if isinstance(sub_schema, Schema30) or isinstance(sub_schema, Schema31):
                conversions.append(type_converter(sub_schema, True))
            else:
                import_type = common.normalize_symbol(sub_schema.ref.split("/")[-1])
                if import_type == model_name and model_name is not None:
                    conversions.append(
                        TypeConversion(
                            original_type=sub_schema.ref,
                            converted_type='"' + model_name + '"',
                            import_types=None,
                        )
                    )
                else:
                    import_types = [f"from .{import_type} import {import_type}"]
                    conversions.append(
                        TypeConversion(
                            original_type=sub_schema.ref,
                            converted_type=import_type,
                            import_types=import_types,
                        )
                    )

        original_type = (
            "tuple<" + ",".join([i.original_type for i in conversions]) + ">"
        )
        if len(conversions) == 1:
            converted_type = conversions[0].converted_type
        else:
            converted_type = (
                "Tuple[" + ",".join([i.converted_type for i in conversions]) + "]"
            )

        converted_type = pre_type + converted_type + post_type
        # Collect first import from referenced sub-schemas only (skip empty lists)
        import_types = [
            i.import_types[0]
            for i in conversions
            if i.import_types is not None and len(i.import_types) > 0
        ] or None

    elif schema.oneOf is not None or schema.anyOf is not None:
        used = schema.oneOf if schema.oneOf is not None else schema.anyOf
        used = used if used is not None else []
        conversions = []
        for sub_schema in used:
            if isinstance(sub_schema, Schema30) or isinstance(sub_schema, Schema31):
                conversions.append(type_converter(sub_schema, True))
            else:
                import_type = common.normalize_symbol(sub_schema.ref.split("/")[-1])
                import_types = [f"from .{import_type} import {import_type}"]
                conversions.append(
                    TypeConversion(
                        original_type=sub_schema.ref,
                        converted_type=import_type,
                        import_types=import_types,
                    )
                )
        original_type = (
            "union<" + ",".join([i.original_type for i in conversions]) + ">"
        )

        if len(conversions) == 1:
            converted_type = conversions[0].converted_type
        else:
            converted_type = (
                "Union[" + ",".join([i.converted_type for i in conversions]) + "]"
            )

        converted_type = pre_type + converted_type + post_type
        import_types = list(
            itertools.chain(
                *[i.import_types for i in conversions if i.import_types is not None]
            )
        )
    # We only want to auto convert to datetime if orjson is used throghout the code, otherwise we can not
    # serialize it to JSON.
    elif (schema.type == "string" or str(schema.type) == "DataType.STRING") and (
        schema.schema_format is None or not common.get_use_orjson()
    ):
        converted_type = pre_type + "str" + post_type
    elif (
        (schema.type == "string" or str(schema.type) == "DataType.STRING")
        and schema.schema_format is not None
        and schema.schema_format.startswith("uuid")
        and common.get_use_orjson()
    ):
        if len(schema.schema_format) > 4 and schema.schema_format[4].isnumeric():
            uuid_type = schema.schema_format.upper()
            converted_type = pre_type + uuid_type + post_type
            import_types = ["from pydantic import " + uuid_type]
        else:
            converted_type = pre_type + "UUID" + post_type
            import_types = ["from uuid import UUID"]
    elif (
        schema.type == "string" or str(schema.type) == "DataType.STRING"
    ) and schema.schema_format == "date-time":
        converted_type = pre_type + "datetime" + post_type
        import_types = ["from datetime import datetime"]
    elif schema.type == "integer" or str(schema.type) == "DataType.INTEGER":
        converted_type = pre_type + "int" + post_type
    elif schema.type == "number" or str(schema.type) == "DataType.NUMBER":
        converted_type = pre_type + "float" + post_type
    elif schema.type == "boolean" or str(schema.type) == "DataType.BOOLEAN":
        converted_type = pre_type + "bool" + post_type
    elif schema.type == "array" or str(schema.type) == "DataType.ARRAY":
        retVal = pre_type + "List["
        if isinstance(schema.items, Reference30) or isinstance(
            schema.items, Reference31
        ):
            converted_reference = _generate_property_from_reference(
                model_name or "", "", schema.items, schema, required
            )
            import_types = converted_reference.type.import_types
            original_type = "array<" + converted_reference.type.original_type + ">"
            retVal += converted_reference.type.converted_type
        elif isinstance(schema.items, Schema30) or isinstance(schema.items, Schema31):
            type_str = schema.items.type
            if hasattr(type_str, "value"):
                type_value = str(type_str.value) if type_str is not None else "unknown"
            else:
                type_value = str(type_str) if type_str is not None else "unknown"
            original_type = "array<" + type_value + ">"
            retVal += type_converter(schema.items, True).converted_type
        else:
            original_type = "array<unknown>"
            retVal += "Any"

        converted_type = retVal + "]" + post_type
    elif schema.type == "object" or str(schema.type) == "DataType.OBJECT":
        converted_type = pre_type + "Dict[str, Any]" + post_type
    elif schema.type == "null" or str(schema.type) == "DataType.NULL":
        converted_type = pre_type + "None" + post_type
    elif schema.type is None:
        converted_type = pre_type + "Any" + post_type
    else:
        # Handle DataType enum types as strings
        if hasattr(schema.type, "value"):
            # Single DataType enum
            if schema.type.value == "string":
                # Check for UUID format first
                if (
                    schema.schema_format is not None
                    and schema.schema_format.startswith("uuid")
                    and common.get_use_orjson()
                ):
                    if (
                        len(schema.schema_format) > 4
                        and schema.schema_format[4].isnumeric()
                    ):
                        uuid_type = schema.schema_format.upper()
                        converted_type = pre_type + uuid_type + post_type
                        import_types = ["from pydantic import " + uuid_type]
                    else:
                        converted_type = pre_type + "UUID" + post_type
                        import_types = ["from uuid import UUID"]
                # Check for date-time format
                elif schema.schema_format == "date-time":
                    converted_type = pre_type + "datetime" + post_type
                    import_types = ["from datetime import datetime"]
                else:
                    converted_type = pre_type + "str" + post_type
            elif schema.type.value == "integer":
                converted_type = pre_type + "int" + post_type
            elif schema.type.value == "number":
                converted_type = pre_type + "float" + post_type
            elif schema.type.value == "boolean":
                converted_type = pre_type + "bool" + post_type
            elif schema.type.value == "array":
                converted_type = pre_type + "List[Any]" + post_type
            elif schema.type.value == "object":
                converted_type = pre_type + "Dict[str, Any]" + post_type
            elif schema.type.value == "null":
                converted_type = pre_type + "None" + post_type
            else:
                converted_type = pre_type + "str" + post_type  # Default fallback
        elif isinstance(schema.type, list) and len(schema.type) > 0:
            # List of DataType enums - use first one
            first_type = schema.type[0]
            if hasattr(first_type, "value"):
                if first_type.value == "string":
                    # Check for UUID format first
                    if (
                        schema.schema_format is not None
                        and schema.schema_format.startswith("uuid")
                        and common.get_use_orjson()
                    ):
                        if (
                            len(schema.schema_format) > 4
                            and schema.schema_format[4].isnumeric()
                        ):
                            uuid_type = schema.schema_format.upper()
                            converted_type = pre_type + uuid_type + post_type
                            import_types = ["from pydantic import " + uuid_type]
                        else:
                            converted_type = pre_type + "UUID" + post_type
                            import_types = ["from uuid import UUID"]
                    # Check for date-time format
                    elif schema.schema_format == "date-time":
                        converted_type = pre_type + "datetime" + post_type
                        import_types = ["from datetime import datetime"]
                    else:
                        converted_type = pre_type + "str" + post_type
                elif first_type.value == "integer":
                    converted_type = pre_type + "int" + post_type
                elif first_type.value == "number":
                    converted_type = pre_type + "float" + post_type
                elif first_type.value == "boolean":
                    converted_type = pre_type + "bool" + post_type
                elif first_type.value == "array":
                    converted_type = pre_type + "List[Any]" + post_type
                elif first_type.value == "object":
                    converted_type = pre_type + "Dict[str, Any]" + post_type
                elif first_type.value == "null":
                    converted_type = pre_type + "None" + post_type
                else:
                    converted_type = pre_type + "str" + post_type  # Default fallback
            else:
                converted_type = pre_type + "str" + post_type  # Default fallback
        else:
            converted_type = pre_type + "str" + post_type  # Default fallback

    return TypeConversion(
        original_type=original_type,
        converted_type=converted_type,
        import_types=import_types,
    )


def _generate_property_from_schema(
    model_name: str, name: str, schema: Schema, parent_schema: Optional[Schema] = None
) -> Property:
    """
    Generates a property from a schema. It takes the type of the schema and converts it to a python type, and then
    creates the according property.
    :param model_name: Name of the model this property belongs to
    :param name: Name of the schema
    :param schema: schema to be converted
    :param parent_schema: Component this belongs to
    :return: Property
    """
    required = (
        parent_schema is not None
        and parent_schema.required is not None
        and name in parent_schema.required
    )

    import_type = None
    if required:
        import_type = [] if name == model_name else [name]

    return Property(
        name=name,
        type=type_converter(schema, required, model_name),
        required=required,
        default=None if required else "None",
        import_type=import_type,
    )


def _generate_property_from_reference(
    model_name: str,
    name: str,
    reference: Reference,
    parent_schema: Optional[Schema] = None,
    force_required: bool = False,
) -> Property:
    """
    Generates a property from a reference. It takes the name of the reference as the type, and then
    returns a property type
    :param name: Name of the schema
    :param reference: reference to be converted
    :param parent_schema: Component this belongs to
    :param force_required: Force the property to be required
    :return: Property and model to be imported by the file
    """
    required = (
        parent_schema is not None
        and parent_schema.required is not None
        and name in parent_schema.required
    ) or force_required
    import_model = common.normalize_symbol(reference.ref.split("/")[-1])

    if import_model == model_name:
        type_conv = TypeConversion(
            original_type=reference.ref,
            converted_type=(
                import_model if required else 'Optional["' + import_model + '"]'
            ),
            import_types=None,
        )
    else:
        type_conv = TypeConversion(
            original_type=reference.ref,
            converted_type=(
                import_model if required else "Optional[" + import_model + "]"
            ),
            import_types=[f"from .{import_model} import {import_model}"],
        )
    return Property(
        name=name,
        type=type_conv,
        required=required,
        default=None if required else "None",
        import_type=[import_model],
    )


def generate_models(
    components: Components, pydantic_version: PydanticVersion = PydanticVersion.V2
) -> List[Model]:
    """
    Receives components from an OpenAPI 3.0+ specification and generates the models from it.
    Additionally:
      - Detects unions / discriminated unions in property schemas (oneOf/anyOf)
      - Emits a named alias module (e.g. TokenIssuer.py)
      - Rewrites the property type to use that alias (instead of Union[...])
    """
    models: List[Model] = []

    if components.schemas is None:
        return models

    jinja_env = create_jinja_env()

    # Track alias modules so we only create each once
    alias_models_by_name: Dict[str, Model] = {}

    for schema_name, schema_or_reference in components.schemas.items():
        name = common.normalize_symbol(schema_name)

        # --------------------------
        # Enums
        # --------------------------
        if schema_or_reference.enum is not None:
            value_dict = schema_or_reference.model_dump()
            value_dict["enum"] = [
                (common.normalize_symbol(str(i)).upper(), i) for i in value_dict["enum"]
            ]
            m = Model(
                file_name=name,
                content=jinja_env.get_template(ENUM_TEMPLATE).render(
                    name=name, **value_dict
                ),
                openapi_object=schema_or_reference,
                properties=[],
            )
            try:
                compile(m.content, "<string>", "exec")
                models.append(m)
            except SyntaxError as e:  # pragma: no cover
                click.echo(f"Error in model {name}: {e}")

            continue  # pragma: no cover

        # --------------------------
        # Normal models
        # --------------------------
        properties: List[Property] = []
        property_iterator = (
            schema_or_reference.properties.items()
            if schema_or_reference.properties is not None
            else {}
        )

        for prop_name, prop_schema in property_iterator:
            # Reference property
            if isinstance(prop_schema, Reference30) or isinstance(prop_schema, Reference31):
                conv_property = _generate_property_from_reference(
                    name, prop_name, prop_schema, schema_or_reference
                )
                properties.append(conv_property)
                continue

            # Schema property
            conv_property = _generate_property_from_schema(
                name, prop_name, prop_schema, schema_or_reference
            )

            # -----------------------------------------
            # NEW: union / discriminated union factoring
            # -----------------------------------------
            if isinstance(prop_schema, (Schema30, Schema31)) and _schema_is_union(prop_schema):
                alias_name = _alias_name_for_property(prop_name)
                discriminator_key = _get_discriminator_key(prop_schema)

                # Build the union type and gather imports from members.
                # Important: we want a NON-optional union for the alias definition.
                union_conv = type_converter(prop_schema, required=True, model_name=name)
                union_type_str = union_conv.converted_type  # e.g. Union[A,B,C]
                member_imports = union_conv.import_types or []

                # Create alias module once
                if alias_name not in alias_models_by_name:
                    alias_content = _render_union_alias_module(
                        jinja_env=jinja_env,
                        alias_name=alias_name,
                        union_type=union_type_str,
                        discriminator_key=discriminator_key,
                        member_imports=member_imports,
                    )

                    # Validate alias module compiles
                    try:
                        compile(alias_content, "<string>", "exec")
                    except SyntaxError as e:  # pragma: no cover
                        click.echo(f"Error in union alias {alias_name}: {e}")  # pragma: no cover

                    alias_models_by_name[alias_name] = Model(
                        file_name=alias_name,
                        content=alias_content,
                        openapi_object=prop_schema,
                        properties=[],
                    )

                # Rewrite property type to use alias
                rewritten_type = alias_name if conv_property.required else f"Optional[{alias_name}]"
                conv_property.type = TypeConversion(
                    original_type=conv_property.type.original_type,
                    converted_type=rewritten_type,
                    import_types=[f"from .{alias_name} import {alias_name}"],
                )

            properties.append(conv_property)

        template_name = (
            MODELS_TEMPLATE_PYDANTIC_V2
            if pydantic_version == PydanticVersion.V2
            else MODELS_TEMPLATE
        )

        generated_content = jinja_env.get_template(template_name).render(
            schema_name=name, schema=schema_or_reference, properties=properties
        )

        try:
            compile(generated_content, "<string>", "exec")
        except SyntaxError as e:  # pragma: no cover
            click.echo(f"Error in model {name}: {e}")  # pragma: no cover

        models.append(
            Model(
                file_name=name,
                content=generated_content,
                openapi_object=schema_or_reference,
                properties=properties,
            )
        )

    # Ensure alias modules are included in output
    models.extend(alias_models_by_name.values())

    return models
