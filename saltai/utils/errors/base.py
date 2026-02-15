from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..typing.json_types import Context


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    code: str
    message: str
    hint: str | None
    context: Context


class SaltAIError(Exception):
    def __init__(
            self,
            code: str,
            message: str,
            *,
            hint: str | None = None,
            context: Context | None = None,
            cause: BaseException | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.context = context or {}
        self.cause = cause

    def to_info(self) -> ErrorInfo:
        return ErrorInfo(
            code=self.code,
            message=self.message,
            hint=self.hint,
            context=dict(self.context),
        )

    def with_context(self, **ctx: Any) -> "SaltAIError":
        merged = dict(self.context)
        merged.update(ctx)
        return self.__class__(
            self.code,
            self.message,
            hint=self.hint,
            context=merged,
            cause=self.cause,
        )

    def __str__(self) -> str:
        base = f"[{self.code}] {self.message}"
        if self.hint:
            base += f" | hint: {self.hint}"
        return base


class ConfigError(SaltAIError): ...


class ManifestError(SaltAIError): ...


class DataError(SaltAIError): ...


class EngineError(SaltAIError): ...


class ArtifactError(SaltAIError): ...


class CheckpointError(SaltAIError): ...


class MetricError(SaltAIError): ...


class LoggerError(SaltAIError): ...


class ReproducibilityError(SaltAIError): ...


class InternalError(SaltAIError): ...
