from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from saltai.logging.utils.jsonable import to_jsonable


class ClearMLNotInstalledError(ImportError):
    pass


def _load_task_class() -> Any:
    try:
        from clearml import Task
    except ImportError as e:
        raise ClearMLNotInstalledError(
            "ClearML is not installed. Install it with `pip install salt-ai[clearml]` "
            "or `poetry install -E clearml`."
        ) from e
    return Task


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _artifact_object_from_uri(uri: Any) -> Any:
    if not isinstance(uri, str):
        return uri

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme == "":
        return Path(uri)
    return uri


class ClearMLLogger(object):
    def __init__(
            self,
            *,
            project_name: str | None = None,
            task_name: str | None = None,
            task: Any | None = None,
            task_type: str | None = None,
            tags: list[str] | tuple[str, ...] | None = None,
            reuse_last_task_id: bool = False,
            auto_connect_arg_parser: bool = False,
            auto_connect_frameworks: bool | Mapping[str, bool] = False,
            auto_connect_streams: bool = False,
            init_kwargs: Mapping[str, Any] | None = None,
            upload_artifacts: bool = True,
            report_events: bool = False,
            close_task: bool = True,
    ):
        if task is None:
            if project_name is None or task_name is None:
                raise ValueError("project_name and task_name are required when task is not provided")

            Task = _load_task_class()
            kwargs = {
                "project_name": project_name,
                "task_name": task_name,
                "reuse_last_task_id": reuse_last_task_id,
                "auto_connect_arg_parser": auto_connect_arg_parser,
                "auto_connect_frameworks": auto_connect_frameworks,
                "auto_connect_streams": auto_connect_streams,
            }
            if task_type is not None:
                kwargs["task_type"] = task_type
            if tags is not None:
                kwargs["tags"] = list(tags)
            if init_kwargs is not None:
                kwargs.update(dict(init_kwargs))
            task = Task.init(**kwargs)

        self.task = task
        self.upload_artifacts = bool(upload_artifacts)
        self.report_events = bool(report_events)
        self.close_task = bool(close_task)
        self._logger = task.get_logger() if hasattr(task, "get_logger") else None

    def connect(self, obj: Any, *, name: str | None = None) -> Any:
        if not hasattr(self.task, "connect"):
            return obj
        if name is None:
            return self.task.connect(obj)
        return self.task.connect(obj, name=name)

    def log(self, event: object) -> None:
        payload = to_jsonable(event)
        if not isinstance(payload, Mapping):
            if self.report_events:
                self._report_text(str(payload))
            return

        event_type = payload.get("type")
        if event_type == "metric":
            self._log_metric(payload)
            return

        if event_type in {"artifact_saved", "checkpoint_saved"}:
            if self.upload_artifacts:
                self._upload_artifact(payload)
            return

        if self.report_events:
            self._report_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def flush(self) -> None:
        if self._logger is not None and hasattr(self._logger, "flush"):
            self._logger.flush()
        elif hasattr(self.task, "flush"):
            self.task.flush()

    def close(self) -> None:
        self.flush()
        if self.close_task and hasattr(self.task, "close"):
            self.task.close()

    def _log_metric(self, payload: Mapping[str, Any]) -> None:
        if self._logger is None or not hasattr(self._logger, "report_scalar"):
            return

        point = payload.get("point")
        if not isinstance(point, Mapping):
            return

        name = point.get("name")
        value = point.get("value")
        if name is None or not _is_scalar(value):
            return

        split = point.get("split") or "metrics"
        step = point.get("step")
        epoch = point.get("epoch")
        iteration = step if step is not None else epoch if epoch is not None else 0

        self._logger.report_scalar(
            title=str(split),
            series=str(name),
            value=float(value),
            iteration=int(iteration),
        )

    def _upload_artifact(self, payload: Mapping[str, Any]) -> None:
        if not hasattr(self.task, "upload_artifact"):
            return

        ref = payload.get("ref")
        if not isinstance(ref, Mapping):
            return

        event_type = payload.get("type")
        name = str(ref.get("name") or ref.get("id") or "artifact")
        if event_type == "checkpoint_saved":
            tag = payload.get("tag")
            name = f"checkpoint/{tag or name}"

        uri = ref.get("uri")
        artifact_object = _artifact_object_from_uri(uri) if uri is not None else dict(ref)
        metadata = ref.get("meta") if isinstance(ref.get("meta"), Mapping) else None

        try:
            self.task.upload_artifact(name=name, artifact_object=artifact_object, metadata=metadata)
        except TypeError:
            self.task.upload_artifact(name=name, artifact_object=artifact_object)

    def _report_text(self, text: str) -> None:
        if self._logger is not None and hasattr(self._logger, "report_text"):
            self._logger.report_text(text)
