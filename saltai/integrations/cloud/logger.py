from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from saltai.logging.utils.jsonable import to_jsonable


class CloudLoggerError(RuntimeError):
    pass


class CloudLogger(object):
    def __init__(
            self,
            *,
            endpoint_url: str,
            api_key: str | None = None,
            headers: Mapping[str, str] | None = None,
            timeout: float = 5.0,
            batch_size: int = 1,
            raise_on_error: bool = False,
    ):
        if not endpoint_url:
            raise ValueError("endpoint_url is required")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout = float(timeout)
        self.batch_size = int(batch_size)
        self.raise_on_error = bool(raise_on_error)

        self._buffer: list[dict[str, Any]] = []
        self._closed = False
        self.errors: list[BaseException] = []

    def log(self, event: object) -> None:
        if self._closed:
            raise CloudLoggerError("CloudLogger is already closed")

        payload = to_jsonable(event)
        if not isinstance(payload, dict):
            payload = {
                "type": "event",
                "payload": payload,
            }

        self._buffer.append(payload)

        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        events = self._buffer
        self._buffer = []

        try:
            self._post_events(events)
        except BaseException as e:
            self.errors.append(e)
            if self.raise_on_error:
                raise

    def close(self) -> None:
        if self._closed:
            return

        self.flush()
        self._closed = True

    def _post_events(self, events: list[dict[str, Any]]) -> None:
        body = json.dumps(
            {
                "events": events,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.headers,
        }

        if self.api_key is not None:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")

        request = Request(
            self.endpoint_url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                if status < 200 or status >= 300:
                    raise CloudLoggerError(f"Cloud logger request failed with status {status}")
        except HTTPError as e:
            raise CloudLoggerError(f"Cloud logger request failed with status {e.code}") from e
        except URLError as e:
            raise CloudLoggerError(f"Cloud logger request failed: {e.reason}") from e
