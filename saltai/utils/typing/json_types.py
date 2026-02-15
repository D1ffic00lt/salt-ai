from __future__ import annotations

from typing import Any, TypeAlias

__all__ = (
    "JSONScalar",
    "JSONValue",
    "JSONObject",
    "JSON",
    "PathLike",
    "Tags",
    "Context",
)

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
JSON: TypeAlias = JSONValue

PathLike: TypeAlias = str
Tags: TypeAlias = dict[str, str]
Context: TypeAlias = dict[str, Any]