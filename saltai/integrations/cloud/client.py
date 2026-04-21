from __future__ import annotations

import hashlib
import json
import mimetypes
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from saltai.logging.utils.jsonable import to_jsonable


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        self.api_prefix = "/" + api_prefix.strip("/")
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
            f"/projects/{quote(str(project_id))}/runs",
            {
                "name": name,
                "config": config or {},
                "manifest": manifest or {},
                "tags": tags or [],
            },
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/runs/{quote(str(run_id))}")

    def update_run(
            self,
            run_id: str,
            *,
            name: str | None = None,
            config: dict[str, Any] | None = None,
            manifest: dict[str, Any] | None = None,
            tags: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/runs/{quote(str(run_id))}",
            {
                "name": name,
                "config": config,
                "manifest": manifest,
                "tags": tags,
            },
        )

    def finish_run(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/runs/{quote(str(run_id))}/finish")

    def fail_run(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/runs/{quote(str(run_id))}/fail")

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
            f"/runs/{quote(str(run_id))}/metrics",
            {
                "key": key,
                "value": float(value),
                "step": step,
                "payload": payload or {},
                "timestamp": timestamp,
            },
        )

    def list_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{quote(str(run_id))}/metrics")

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
            f"/runs/{quote(str(run_id))}/events",
            {
                "type": type,
                "level": level,
                "message": message,
                "payload": payload or {},
                "timestamp": timestamp,
            },
        )

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{quote(str(run_id))}/events")

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
            f"/runs/{quote(str(run_id))}/artifacts",
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
            f"/artifacts/{quote(str(artifact_id))}/complete",
            {
                "storage_uri": storage_uri,
                "size_bytes": size_bytes,
                "content_type": content_type,
                "hash": hash,
                "meta": meta,
            },
        )

    def create_local_artifact(
            self,
            run_id: str,
            path: str | Path,
            *,
            name: str | None = None,
            kind: str = "other",
            content_type: str | None = None,
            meta: dict[str, Any] | None = None,
            complete: bool = True,
    ) -> dict[str, Any]:
        local_path = Path(path)
        if not local_path.is_file():
            raise CloudClientError(f"Artifact file not found: {local_path}")

        artifact_name = name or local_path.name
        artifact_hash = _sha256_file(local_path)
        artifact_size = local_path.stat().st_size
        artifact_content_type = content_type or mimetypes.guess_type(str(local_path))[0]

        artifact = self.create_artifact(
            run_id,
            artifact_name,
            kind=kind,
            size_bytes=artifact_size,
            content_type=artifact_content_type,
            hash=artifact_hash,
            meta=meta,
        )

        if not complete:
            return artifact

        artifact_id = artifact["id"]
        return self.complete_artifact(
            artifact_id,
            storage_uri=f"file://{local_path.resolve()}",
            size_bytes=artifact_size,
            content_type=artifact_content_type,
            hash=artifact_hash,
            meta=meta,
        )

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{quote(str(run_id))}/artifacts")

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._request("GET", f"/artifacts/{quote(str(artifact_id))}")

    def get_artifact_download_reference(self, artifact_id: str) -> dict[str, Any]:
        return self._request("GET", f"/artifacts/{quote(str(artifact_id))}/download")

    def _request(
            self,
            method: str,
            path: str,
            payload: dict[str, Any] | None = None,
    ) -> Any:
        url = self._url(path)
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
            headers["Content-Type"] = "application/json"

        request = Request(
            url,
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