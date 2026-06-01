from __future__ import annotations

import time

from collections.abc import Sequence
from typing import Any

from saltai.engine.event_bus.bus import EventBus
from saltai.utils.typing.core import DataModule, Metric, MetricPoint, MetricSummary, ModelAdapter, RunId
from saltai.utils.typing.events import EpochFinished, EpochStarted, MetricLogged, StepFinished, StepStarted


class Trainer(object):
    def __init__(self, *, event_bus: EventBus | None = None, run_id: RunId | str = RunId("local")):
        self._bus = event_bus or EventBus([])
        self._run_id = RunId(str(run_id))

    def _publish(self, ev: object) -> None:
        self._bus.publish(ev, context={"run_id": str(self._run_id)})

    @staticmethod
    def _emit_metric_events(
            *,
            bus_publish,
            run_id: RunId,
            split: str,
            epoch: int,
            step: int,
            metrics: Sequence[Metric[Any, Any]],
    ) -> dict[str, float | int]:
        values: dict[str, float | int] = {}

        for metric in metrics:
            raw = metric.compute()
            if isinstance(raw, dict):
                for sub_name, v in raw.items():
                    key = f"{metric.name}/{sub_name}"
                    values[key] = v
                    bus_publish(
                        MetricLogged(
                            type="metric",
                            run_id=run_id,
                            ts=time.time(),
                            data={},
                            point=MetricPoint(
                                name=key,
                                value=v,
                                step=step,
                                epoch=epoch,
                                split=split,
                                extra={},
                            ),
                        )
                    )
            else:
                key = metric.name
                values[key] = raw
                bus_publish(
                    MetricLogged(
                        type="metric",
                        run_id=run_id,
                        ts=time.time(),
                        data={},
                        point=MetricPoint(
                            name=key,
                            value=raw,
                            step=step,
                            epoch=epoch,
                            split=split,
                            extra={},
                        ),
                    )
                )
        return values

    def fit(
            self,
            model: ModelAdapter[Any, Any, Any],
            datamodule: DataModule[Any],
            *,
            metrics: Sequence[Metric[Any, Any]] | None = None,
            epochs: int = 1,
            start_step: int = 0,
    ) -> MetricSummary:
        ms = list(metrics or [])
        datamodule.prepare()

        step = int(start_step)
        last_values: dict[str, float | int] = {}

        for epoch in range(int(epochs)):
            for m in ms:
                m.reset()

            self._publish(
                EpochStarted(
                    type="epoch_started",
                    run_id=self._run_id,
                    ts=time.time(),
                    data={},
                    epoch=epoch,
                    split="train",
                )
            )

            for batch in datamodule.train_iter():
                step += 1
                self._publish(
                    StepStarted(
                        type="step_started",
                        run_id=self._run_id,
                        ts=time.time(),
                        data={},
                        step=step,
                        epoch=epoch,
                        split="train",
                    )
                )

                pred = model.forward(batch)
                loss = model.loss(pred, batch)
                model.zero_grad()
                model.backward(loss)
                model.step()

                for m in ms:
                    m.update(batch, pred)

                self._publish(
                    StepFinished(
                        type="step_finished",
                        run_id=self._run_id,
                        ts=time.time(),
                        data={},
                        step=step,
                        epoch=epoch,
                        split="train",
                    )
                )

            last_values = self._emit_metric_events(
                bus_publish=self._publish,
                run_id=self._run_id,
                split="train",
                epoch=epoch,
                step=step,
                metrics=ms,
            )

            self._publish(
                EpochFinished(
                    type="epoch_finished",
                    run_id=self._run_id,
                    ts=time.time(),
                    data={},
                    epoch=epoch,
                    split="train",
                )
            )

        return MetricSummary(values=last_values, extra={"epochs": int(epochs), "steps": int(step - start_step)})

    def evaluate(
            self,
            model: ModelAdapter[Any, Any, Any],
            datamodule: DataModule[Any],
            *,
            metrics: Sequence[Metric[Any, Any]] | None = None,
            split: str = "val",
    ) -> MetricSummary:
        ms = list(metrics or [])
        datamodule.prepare()

        for m in ms:
            m.reset()

        self._publish(
            EpochStarted(
                type="epoch_started",
                run_id=self._run_id,
                ts=time.time(),
                data={},
                epoch=0,
                split=split,
            )
        )

        step = 0
        it = datamodule.val_iter() if split == "val" else datamodule.test_iter()
        for batch in it or []:
            step += 1
            self._publish(
                StepStarted(
                    type="step_started",
                    run_id=self._run_id,
                    ts=time.time(),
                    data={},
                    step=step,
                    epoch=0,
                    split=split,
                )
            )

            pred = model.forward(batch)
            for m in ms:
                m.update(batch, pred)

            self._publish(
                StepFinished(
                    type="step_finished",
                    run_id=self._run_id,
                    ts=time.time(),
                    data={},
                    step=step,
                    epoch=0,
                    split=split,
                )
            )

        values = self._emit_metric_events(
            bus_publish=self._publish,
            run_id=self._run_id,
            split=split,
            epoch=0,
            step=step,
            metrics=ms,
        )

        self._publish(
            EpochFinished(
                type="epoch_finished",
                run_id=self._run_id,
                ts=time.time(),
                data={},
                epoch=0,
                split=split,
            )
        )

        return MetricSummary(values=values, extra={"split": split, "steps": step})
