from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from saltai.integrations.cloud.client import CloudClient
from saltai.logging.utils.jsonable import to_jsonable

__all__ = (
    "CloudRunLogger",
    "CloudRunLoggerError",
)


class CloudRunLoggerError(RuntimeError):
    pass


class CloudRunLogger(object):
    def __init__(
            self,
            *,
            client: CloudClient,
            run_id: str,
            raise_on_error: bool = False,
            log_lifecycle_events: bool = True,
            log_artifacts: bool = True,
    ):
        if not run_id:
            raise ValueError("run_id is required")

        self.client = client
        self.run_id = str(run_id)
        self.raise_on_error = bool(raise_on_error)
        self.log_lifecycle_events = bool(log_lifecycle_events)
        self.log_artifacts = bool(log_artifacts)

        self.errors: list[BaseException] = []
        self._closed = False
        self._failed = False
        self._finished = False

    def log(self, event: object) -> None:
        if self._closed:
            raise CloudRunLoggerError("CloudRunLogger is already closed")

        try:
            self._send(event)
        except BaseException as e:
            self.errors.append(e)
            if self.raise_on_error:
                raise

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True

    def _send(self, event: object) -> None:
        event_type = str(getattr(event, "type", type(event).__name__))
        timestamp = _event_datetime(event)
        payload = _event_payload(event)

        if event_type == "metric":
            point = getattr(event, "point", None)
            if point is None:
                self._log_event(event_type, "warning", "Metric event has no point", payload, timestamp)
                return

            self.client.log_metric(
                self.run_id,
                str(getattr(point, "name")),
                float(getattr(point, "value")),
                step=getattr(point, "step", None),
                payload={
                    "epoch": getattr(point, "epoch", None),
                    "split": getattr(point, "split", None),
                    "extra": getattr(point, "extra", {}),
                    "event": payload,
                },
                timestamp=timestamp,
            )
            return

        if event_type == "run_failed":
            self._failed = True
            self._log_event(event_type, "error", "Run failed", payload, timestamp)
            self.client.fail_run(self.run_id)
            return

        if event_type == "run_finished":
            status = _event_data(event).get("status")
            level = "error" if status == "failed" else "info"
            self._log_event(event_type, level, "Run finished", payload, timestamp)

            if status != "failed" and not self._failed and not self._finished:
                self.client.finish_run(self.run_id)
                self._finished = True

            return

        if event_type in {"artifact_saved", "checkpoint_saved"}:
            if self.log_artifacts:
                self._register_artifact_event(event, event_type, timestamp, payload)
            return

        if self.log_lifecycle_events:
            self._log_event(event_type, "info", None, payload, timestamp)

    def _register_artifact_event(
            self,
            event: object,
            event_type: str,
            timestamp: datetime | None,
            payload: dict[str, Any],
    ) -> None:
        ref = getattr(event, "ref", None)
        if ref is None:
            self._log_event(event_type, "warning", "Artifact event has no ref", payload, timestamp)
            return

        ref_payload = _json_dict(ref)

        artifact = self.client.create_artifact(
            self.run_id,
            str(ref_payload.get("name") or ref_payload.get("id") or event_type),
            kind=str(ref_payload.get("kind") or "other"),
            size_bytes=ref_payload.get("size_bytes"),
            content_type=None,
            hash=ref_payload.get("sha256"),
            meta={
                "event_type": event_type,
                "local_ref": ref_payload,
                "event": payload,
            },
        )

        storage_uri = ref_payload.get("uri")
        if storage_uri:
            self.client.complete_artifact(
                str(artifact["id"]),
                storage_uri=str(storage_uri),
                size_bytes=ref_payload.get("size_bytes"),
                content_type=None,
                hash=ref_payload.get("sha256"),
                meta={
                    "event_type": event_type,
                    "local_ref": ref_payload,
                },
            )

        self._log_event(event_type, "info", "Artifact registered", payload, timestamp)

    def _log_event(
            self,
            event_type: str,
            level: str,
            message: str | None,
            payload: dict[str, Any],
            timestamp: datetime | None,
    ) -> None:
        self.client.log_event(
            self.run_id,
            event_type,
            level=level,
            message=message,
            payload=payload,
            timestamp=timestamp,
        )


def _event_datetime(event: object) -> datetime | None:
    ts = getattr(event, "ts", None)
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def _event_data(event: object) -> dict[str, Any]:
    data = getattr(event, "data", None)
    if isinstance(data, dict):
        return data
    return {}


def _event_payload(event: object) -> dict[str, Any]:
    payload = _json_dict(event)
    payload["local_run_id"] = str(getattr(event, "run_id", ""))
    return payload


def _json_dict(value: object) -> dict[str, Any]:
    if is_dataclass(value):
        raw = asdict(value)
    else:
        raw = to_jsonable(value)

    raw = to_jsonable(raw)

    if isinstance(raw, dict):
        return raw

    return {"value": raw}
