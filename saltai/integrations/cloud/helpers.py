from __future__ import annotations

from typing import Any

from saltai.engine.event_bus.bus import EventBus
from saltai.integrations.cloud.artifact_store import CloudArtifactStore, StorageUriBuilder
from saltai.integrations.cloud.client import CloudClient
from saltai.integrations.cloud.run_logger import CloudRunLogger

__all__ = (
    "create_cloud_event_bus",
    "create_cloud_run_context",
)


def create_cloud_event_bus(
        *,
        base_url: str,
        api_token: str,
        project_id: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        manifest: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        raise_on_error: bool = False,
        log_lifecycle_events: bool = True,
        log_artifacts: bool = True,
) -> tuple[CloudClient, dict[str, Any], EventBus]:
    client = CloudClient(
        base_url=base_url,
        api_token=api_token,
    )

    cloud_run = client.create_run(
        project_id=project_id,
        name=run_name,
        config=config,
        manifest=manifest,
        tags=tags,
    )

    cloud_logger = CloudRunLogger(
        client=client,
        run_id=str(cloud_run["id"]),
        raise_on_error=raise_on_error,
        log_lifecycle_events=log_lifecycle_events,
        log_artifacts=log_artifacts,
    )

    event_bus = EventBus([cloud_logger], fail_fast=raise_on_error)

    return client, cloud_run, event_bus


def create_cloud_run_context(
        *,
        base_url: str,
        api_token: str,
        project_id: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        manifest: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        raise_on_error: bool = False,
        log_lifecycle_events: bool = True,
        log_artifacts: bool = True,
        storage_uri_builder: StorageUriBuilder | None = None,
) -> tuple[CloudClient, dict[str, Any], EventBus, CloudArtifactStore]:
    client, cloud_run, event_bus = create_cloud_event_bus(
        base_url=base_url,
        api_token=api_token,
        project_id=project_id,
        run_name=run_name,
        config=config,
        manifest=manifest,
        tags=tags,
        raise_on_error=raise_on_error,
        log_lifecycle_events=log_lifecycle_events,
        log_artifacts=log_artifacts,
    )

    artifact_store = CloudArtifactStore(
        client=client,
        run_id=str(cloud_run["id"]),
        storage_uri_builder=storage_uri_builder,
    )

    return client, cloud_run, event_bus, artifact_store
