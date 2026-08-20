"""Project executable tool contracts into provider-native function schemas."""

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from auto_harness.providers.protocols.schemas import canonical_json_hash


_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


class ToolSchemaProjectionError(ValueError):
    """A registry contract cannot be safely exposed to a provider."""


class ToolArgumentsValidationError(ValueError):
    """Native arguments violate the exact schema shown to the provider."""


@dataclass(frozen=True)
class ToolSchemaProjection:
    tools: List[Dict[str, Any]]
    schema_hash: str
    tool_names: List[str]


def _strict_object_schemas(value: Any) -> Any:
    """Copy a JSON schema and close every explicitly declared object."""
    if isinstance(value, list):
        return [_strict_object_schemas(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _strict_object_schemas(item) for key, item in value.items()}
    if result.get("type") == "object" or "properties" in result:
        result.setdefault("type", "object")
        result["additionalProperties"] = False
    return result


def project_provider_tools(
    registry: Any,
    *,
    stage: str,
    agent_mode: str,
    allowed_categories: Sequence[str] = ("read_only",),
    allowed_tool_names: Optional[Iterable[str]] = None,
) -> ToolSchemaProjection:
    """Return the minimal, deterministic tool schema visible to an LLM.

    Only tools already executable for the current stage/mode can be exposed.
    Executor, approval, policy and side-effect metadata intentionally remain
    local and are never placed in the provider request.
    """
    allowed_categories_set = set(allowed_categories)
    allowed_names_set = set(allowed_tool_names) if allowed_tool_names is not None else None
    provider_tools: List[Dict[str, Any]] = []
    names: List[str] = []
    executable = registry.executable_for_stage(stage, agent_mode=agent_mode)
    for contract in sorted(executable, key=lambda item: str(item.get("name", ""))):
        name = str(contract.get("name", ""))
        if contract.get("category") not in allowed_categories_set:
            continue
        if allowed_names_set is not None and name not in allowed_names_set:
            continue
        if not _TOOL_NAME.fullmatch(name):
            raise ToolSchemaProjectionError("invalid provider tool name: %s" % name)
        parameters = _strict_object_schemas(copy.deepcopy(contract.get("input_schema") or {}))
        if not isinstance(parameters, dict):
            raise ToolSchemaProjectionError("tool input schema must be an object: %s" % name)
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters["additionalProperties"] = False
        provider_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(contract.get("success_signal") or "Execute %s." % name)[:500],
                "parameters": parameters,
            },
        })
        names.append(name)
    return ToolSchemaProjection(
        tools=provider_tools,
        schema_hash=canonical_json_hash(provider_tools),
        tool_names=names,
    )


def validate_tool_arguments(schema: Dict[str, Any], value: Any, path: str = "arguments") -> None:
    """Validate the bounded JSON-schema subset used by registered tools."""
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise ToolArgumentsValidationError("%s must be an object" % path)
        properties = schema.get("properties") or {}
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise ToolArgumentsValidationError(
                "%s is missing required fields: %s" % (path, ", ".join(sorted(missing)))
            )
        unknown = set(value) - set(properties)
        if schema.get("additionalProperties") is False and unknown:
            raise ToolArgumentsValidationError(
                "%s contains unknown fields: %s" % (path, ", ".join(sorted(unknown)))
            )
        for name, item in value.items():
            if name in properties:
                validate_tool_arguments(properties[name], item, "%s.%s" % (path, name))
    elif expected == "array":
        if not isinstance(value, list):
            raise ToolArgumentsValidationError("%s must be an array" % path)
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ToolArgumentsValidationError("%s has too few items" % path)
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ToolArgumentsValidationError("%s has too many items" % path)
        for index, item in enumerate(value):
            validate_tool_arguments(schema.get("items") or {}, item, "%s[%d]" % (path, index))
    elif expected == "string":
        if not isinstance(value, str):
            raise ToolArgumentsValidationError("%s must be a string" % path)
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ToolArgumentsValidationError("%s is too short" % path)
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ToolArgumentsValidationError("%s is too long" % path)
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolArgumentsValidationError("%s must be an integer" % path)
        if "minimum" in schema and value < int(schema["minimum"]):
            raise ToolArgumentsValidationError("%s is below minimum" % path)
        if "maximum" in schema and value > int(schema["maximum"]):
            raise ToolArgumentsValidationError("%s is above maximum" % path)
    elif expected == "boolean" and not isinstance(value, bool):
        raise ToolArgumentsValidationError("%s must be a boolean" % path)
    if "enum" in schema and value not in schema["enum"]:
        raise ToolArgumentsValidationError("%s is not an allowed value" % path)


__all__ = [
    "ToolSchemaProjection",
    "ToolSchemaProjectionError",
    "ToolArgumentsValidationError",
    "project_provider_tools",
    "validate_tool_arguments",
]
