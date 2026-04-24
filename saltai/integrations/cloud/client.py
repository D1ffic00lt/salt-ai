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

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{_quote_id(run_id)}/artifacts")

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._request("GET", f"/artifacts/{_quote_id(artifact_id)}")

    def get_artifact_download_reference(self, artifact_id: str) -> dict[str, Any]:
        return self._request("GET", f"/artifacts/{_quote_id(artifact_id)}/download")

    def upload_artifact_file(
            self,
            artifact_id: str,
            local_path: str | Path,
    ) -> dict[str, Any]:
        path = Path(local_path).expanduser().resolve()

        if not path.exists():
            raise CloudClientError(f"Artifact file not found: {path}")
        if not path.is_file():
            raise CloudClientError(f"Artifact path is not a file: {path}")

        body, content_type = self._multipart_file_body(path)

        return self._request_raw(
            "POST",
            f"/artifacts/{_quote_id(artifact_id)}/upload",
            body=body,
            content_type=content_type,
            accept="application/json",
            parse_json=True,
        )

    def download_artifact_content(self, artifact_id: str) -> bytes:
        return self._request_bytes("GET", f"/artifacts/{_quote_id(artifact_id)}/content")

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

        content_type = "application/json" if body is not None else None

        return self._request_raw(
            method,
            path,
            body=body,
            content_type=content_type,
            accept="application/json",
            parse_json=True,
        )

    def _request_bytes(
            self,
            method: str,
            path: str,
            body: bytes | None = None,
            content_type: str | None = None,
    ) -> bytes:
        return self._request_raw(
            method,
            path,
            body=body,
            content_type=content_type,
            accept="application/octet-stream",
            parse_json=False,
        )

    def _request_raw(
            self,
            method: str,
            path: str,
            *,
            body: bytes | None = None,
            content_type: str | None = None,
            accept: str = "application/json",
            parse_json: bool = True,
    ) -> Any:
        headers = {
            "Accept": accept,
            "Authorization": f"Bearer {self.api_token}",
            **self.headers,
        }

        if content_type is not None:
            headers["Content-Type"] = content_type

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

                if not parse_json:
                    return data

                if not data:
                    return {}

                return self._parse_response(data)

        except HTTPError as e:
            data = e.read()
            parsed = self._parse_response(data)
            raise CloudApiError(e.code, self._extract_detail(parsed), parsed) from e

        except URLError as e:
            raise CloudClientError(f"SaltAI Cloud request failed: {e.reason}") from e

    def _multipart_file_body(self, path: Path) -> tuple[bytes, str]:
        boundary = f"----saltai-cloud-{uuid.uuid4().hex}"
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        filename = path.name

        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n"
            "\r\n"
        ).encode("utf-8")

        footer = f"\r\n--{boundary}--\r\n".encode("utf-8")

        return header + path.read_bytes() + footer, f"multipart/form-data; boundary={boundary}"

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
