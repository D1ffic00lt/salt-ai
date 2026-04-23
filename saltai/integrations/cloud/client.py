from __future__ import annotations

import json
import mimetypes
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from saltai.logging.utils.jsonable import to_jsonable

__all__ = (
    "CloudClient",
    "CloudClientError",
    "CloudApiError",
)


class CloudClientError(RuntimeError):
    pass


class CloudApiError(CloudClientError):
    def __init__(self, status_code: int, detail: Any, body: Any | None = None):
        self.status_code = int(status_code)
        self.detail = detail
        self.body = body
        super().__init__(f"SaltAI Cloud API error {status_code}: {detail}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _quote_id(value: object) -> str:
    return quote(str(value), safe="")


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _multipart_file_body(
        *,
        boundary: str,
        field_name: str,
        filename: str,
        content_type: str,
        data: bytes,
) -> bytes:
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n"
        "\r\n"
    ).encode("utf-8")

    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")

    return head + data + tail


class CloudClient:
    def __init__(
            self,
            *,
            base_url: str,
            api_token: str,
            api_prefix: str = "/api/v1",
            timeout: float = 30.0,
            headers: Mapping[str, str] | None = None,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        if not api_token:
            raise ValueError("api_token is required")

        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.api_prefix = "/" + api_prefix.strip("/") if api_prefix else ""
        self.timeout = float(timeout)
        self.headers = dict(headers or {})

    def auth_me(self) -> dict[str, Any]:
        return self._request("GET", "/auth/me")

    def create_run(
            self,
            project_id: str,
            *,
            name: str | None = None,
            config: dict[str, Any] | None = None,
            manifest: dict[str, Any] | None = None,
            tags: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/projects/{_quote_id(project_id)}/runs",
            {
                "name": name,
                "config": config or {},
                "manifest": manifest or {},
                "tags": tags or [],
            },
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/runs/{_quote_id(run_id)}")

    def update_run(
            self,
            run_id: str,
            *,
            name: str | None = None,
            config: dict[str, Any] | None = None,
            manifest: dict[str, Any] | None = None,
            tags: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = _drop_none(
            {
                "name": name,
                "config": config,
                "manifest": manifest,
                "tags": tags,
            }
        )

        return self._request("PATCH", f"/runs/{_quote_id(run_id)}", payload)

    def finish_run(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/runs/{_quote_id(run_id)}/finish")

    def fail_run(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/runs/{_quote_id(run_id)}/fail")

    def log_metric(
            self,
            run_id: str,
            key: str,
            value: float,
            *,
            step: int | None = None,
            payload: dict[str, Any] | None = None,
            timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runs/{_quote_id(run_id)}/metrics",
            {
                "key": key,
                "value": float(value),
                "step": step,
                "payload": payload or {},
                "timestamp": timestamp,
            },
        )

    def list_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{_quote_id(run_id)}/metrics")

    def log_event(
            self,
            run_id: str,
            type: str,
            *,
            level: str = "info",
            message: str | None = None,
            payload: dict[str, Any] | None = None,
            timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runs/{_quote_id(run_id)}/events",
            {
                "type": type,
                "level": level,
                "message": message,
                "payload": payload or {},
                "timestamp": timestamp,
            },
        )

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{_quote_id(run_id)}/events")

    def create_artifact(
            self,
            run_id: str,
            name: str,
            *,
            kind: str = "other",
            size_bytes: int | None = None,
            content_type: str | None = None,
            hash: str | None = None,
            meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runs/{_quote_id(run_id)}/artifacts",
            {
                "name": name,
                "kind": kind,
                "size_bytes": size_bytes,
                "content_type": content_type,
                "hash": hash,
                "meta": meta or {},
            },
        )

    def complete_artifact(
            self,
            artifact_id: str,
            *,
            storage_uri: str | None = None,
            size_bytes: int | None = None,
            content_type: str | None = None,
            hash: str | None = None,
            meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/artifacts/{_quote_id(artifact_id)}/complete",
            {
                "storage_uri": storage_uri,
                "size_bytes": size_bytes,
                "content_type": content_type,
                "hash": hash,
                "meta": meta,
            },
        )

    def upload_artifact_file(
            self,
            artifact_id: str,
            local_path: str | Path,
            *,
            field_name: str = "file",
    ) -> dict[str, Any]:
        path = Path(local_path).expanduser().resolve()

        if not path.exists():
            raise CloudClientError(f"Artifact file not found: {path}")
        if not path.is_file():
            raise CloudClientError(f"Artifact path is not a file: {path}")

        boundary = f"saltai-{uuid.uuid4().hex}"
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

        body = _multipart_file_body(
            boundary=boundary,
            field_name=field_name,
            filename=path.name,
            content_type=content_type,
            data=path.read_bytes(),
        )

        return self._request_bytes(
            "POST",
            f"/artifacts/{_quote_id(artifact_id)}/upload",
            body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{_quote_id(run_id)}/artifacts")

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._request("GET", f"/artifacts/{_quote_id(artifact_id)}")

    def get_artifact_download_reference(self, artifact_id: str) -> dict[str, Any]:
        return self._request("GET", f"/artifacts/{_quote_id(artifact_id)}/download")

    def download_artifact_content(
            self,
            artifact_id: str,
            dst_path: str | Path,
    ) -> str:
        dst = Path(dst_path).expanduser().resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        data = self._request_raw("GET", f"/artifacts/{_quote_id(artifact_id)}/content")
        dst.write_bytes(data)

        return str(dst)

    def _request(
            self,
            method: str,
            path: str,
            payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None

        if payload is not None:
            body = json.dumps(
                to_jsonable(payload),
                ensure_ascii=False,
                default=_json_default,
                separators=(",", ":"),
            ).encode("utf-8")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_token}",
            **self.headers,
        }

        if body is not None:
            headers.setdefault("Content-Type", "application/json")

        request = Request(
            self._url(path),
            data=body,
            headers=headers,
            method=method.upper(),
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                data = response.read()

                if status < 200 or status >= 300:
                    parsed = self._parse_response(data)
                    raise CloudApiError(status, self._extract_detail(parsed), parsed)

                if not data:
                    return {}

                return self._parse_response(data)

        except HTTPError as e:
            data = e.read()
            parsed = self._parse_response(data)
            raise CloudApiError(e.code, self._extract_detail(parsed), parsed) from e

        except URLError as e:
            raise CloudClientError(f"SaltAI Cloud request failed: {e.reason}") from e

    def _request_bytes(
            self,
            method: str,
            path: str,
            body: bytes,
            *,
            content_type: str,
    ) -> Any:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": content_type,
            **self.headers,
        }

        request = Request(
            self._url(path),
            data=body,
            headers=headers,
            method=method.upper(),
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                data = response.read()

                if status < 200 or status >= 300:
                    parsed = self._parse_response(data)
                    raise CloudApiError(status, self._extract_detail(parsed), parsed)

                if not data:
                    return {}

                return self._parse_response(data)

        except HTTPError as e:
            data = e.read()
            parsed = self._parse_response(data)
            raise CloudApiError(e.code, self._extract_detail(parsed), parsed) from e

        except URLError as e:
            raise CloudClientError(f"SaltAI Cloud request failed: {e.reason}") from e

    def _request_raw(
            self,
            method: str,
            path: str,
    ) -> bytes:
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            **self.headers,
        }

        request = Request(
            self._url(path),
            headers=headers,
            method=method.upper(),
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                data = response.read()

                if status < 200 or status >= 300:
                    parsed = self._parse_response(data)
                    raise CloudApiError(status, self._extract_detail(parsed), parsed)

                return data

        except HTTPError as e:
            data = e.read()
            parsed = self._parse_response(data)
            raise CloudApiError(e.code, self._extract_detail(parsed), parsed) from e

        except URLError as e:
            raise CloudClientError(f"SaltAI Cloud request failed: {e.reason}") from e

    def _url(self, path: str) -> str:
        return f"{self.base_url}{self.api_prefix}/{path.lstrip('/')}"

    @staticmethod
    def _parse_response(data: bytes) -> Any:
        if not data:
            return {}

        text = data.decode("utf-8", errors="replace")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    @staticmethod
    def _extract_detail(body: Any) -> Any:
        if isinstance(body, dict):
            return body.get("detail", body)
        return body
