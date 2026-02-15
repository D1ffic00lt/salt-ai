from __future__ import annotations

from typing import Callable, TypeVar

from .base import InternalError, SaltAIError
from .codes import EC
from ..typing.json_types import Context

T = TypeVar("T")


def ensure(
        cond: bool,
        *,
        code: str,
        message: str,
        hint: str | None = None,
        context: Context | None = None,
        exc_type: type[SaltAIError] = SaltAIError,
) -> None:
    if not cond:
        raise exc_type(code, message, hint=hint, context=context)


def wrap_unknown(
        e: BaseException,
        *,
        message: str = "Unhandled internal error",
        context: Context | None = None,
) -> SaltAIError | InternalError:
    if isinstance(e, SaltAIError):
        return e
    return InternalError(
        EC.INTERNAL,
        message,
        hint="Report a bug and attach run manifest + logs",
        context=context,
        cause=e,
    )


def guard(stage: str, fn: Callable[[], T], *, context: Context | None = None) -> T:
    try:
        return fn()
    except BaseException as e:
        err = wrap_unknown(e, context={**(context or {}), "stage": stage})
        raise err from e
