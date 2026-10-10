"""Bounded, fixed-arrival scheduling for acceptance drivers.

This module is deliberately transport-free.  A caller reserves a worker before
submitting its one durable original request and completes it only when the POST
returns.  Due slots are consumed once: a full worker pool or a driver stall
records an unsent slot instead of building a delayed executor queue.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Iterable, Mapping, Any, NamedTuple


class ArrivalDecision(NamedTuple):
    index: int
    scheduled_at: float
    decided_at: float
    action: str
    lag_seconds: float
    active_workers: int


def _percentiles(values: Iterable[float]) -> dict[str, float | int] | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    def rank(percent: float) -> float:
        return ordered[max(0, math.ceil(len(ordered) * percent) - 1)]
    return {"count": len(ordered), "p50": rank(.50), "p95": rank(.95), "max": ordered[-1]}


def segmented_timing(items: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float | int] | None]:
    """Summarize only timestamp pairs actually present as numeric values."""
    segments = {
        "arrival": ("scheduled_arrival_monotonic", "actual_dispatch_monotonic"),
        "admission": ("actual_dispatch_monotonic", "admitted_monotonic"),
        "dispatch_to_upstream_send": ("actual_dispatch_monotonic", "upstream_send_monotonic_derived_from_wall"),
        "upstream_send_to_finished": ("upstream_send_monotonic_derived_from_wall", "upstream_finished_monotonic_derived_from_wall"),
        "upstream_send_to_success_observed": ("upstream_send_monotonic_derived_from_wall", "provider_success_observed_monotonic"),
        "dispatch_to_result_save": ("actual_dispatch_monotonic", "saved_monotonic"),
        "dispatch_to_archive": ("actual_dispatch_monotonic", "archive_confirmed_monotonic"),
    }
    summary: dict[str, dict[str, float | int] | None] = {}
    entries = list(items)
    for name, (began, ended) in segments.items():
        values = []
        for item in entries:
            start, finish = item.get(began), item.get(ended)
            if isinstance(start, (int, float)) and isinstance(finish, (int, float)) and finish >= start:
                values.append(float(finish) - float(start))
        summary[name] = _percentiles(values)
    return summary


class FixedArrivalScheduler:
    """Consumes a finite schedule without delayed work or catch-up bursts."""

    def __init__(self, count: int, interval: float, workers: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("arrival interval must be positive")
        if workers < 1:
            raise ValueError("submission workers must be positive")
        self.count, self.interval, self.workers, self.clock = count, interval, workers, clock
        self.started_at: float | None = None
        self._next = 0
        self._active: set[int] = set()
        self._records: dict[int, dict[str, Any]] = {}
        self.stop_reason: str | None = None

    def start(self, started_at: float | None = None) -> float:
        if self.started_at is not None:
            raise RuntimeError("schedule already started")
        self.started_at = self.clock() if started_at is None else float(started_at)
        return self.started_at

    @property
    def active_workers(self) -> int:
        return len(self._active)

    @property
    def complete(self) -> bool:
        return self._next == self.count and not self._active

    def _scheduled_at(self, index: int) -> float:
        assert self.started_at is not None
        return self.started_at + (index * self.interval)

    def due(self, now: float | None = None) -> list[ArrivalDecision]:
        """Consume all due slots, reserving at most one worker this tick.

        A stalled driver can observe several overdue slots.  Older slots are
        marked ``unsent_lag`` and only the newest one may reserve capacity, so
        restarting the loop cannot produce a catch-up burst.
        """
        if self.started_at is None:
            raise RuntimeError("schedule not started")
        now = self.clock() if now is None else float(now)
        if self.stop_reason is not None:
            return []
        last_due = self._next - 1
        while last_due + 1 < self.count and self._scheduled_at(last_due + 1) <= now:
            last_due += 1
        if last_due < self._next:
            return []
        decisions: list[ArrivalDecision] = []
        while self._next < last_due:
            decisions.append(self._record(self._next, now, "unsent_lag"))
            self._next += 1
        index = self._next
        action = "dispatch" if len(self._active) < self.workers else "unsent_capacity"
        decision = self._record(index, now, action)
        if action == "dispatch":
            self._active.add(index)  # reserve before handing work to an executor
            self._records[index]["reserved_for_submission"] = True
        self._next += 1
        decisions.append(decision)
        return decisions

    def _record(self, index: int, now: float, action: str) -> ArrivalDecision:
        scheduled_at = self._scheduled_at(index)
        decision = ArrivalDecision(index, scheduled_at, now, action, max(0.0, now - scheduled_at), len(self._active))
        self._records[index] = decision._asdict()
        return decision

    def dispatched(self, index: int, actual_at: float | None = None) -> None:
        if index not in self._active:
            raise RuntimeError("dispatch was not reserved")
        record = self._records[index]
        if "actual_dispatch_monotonic" in record:
            raise RuntimeError("dispatch was already recorded")
        actual = self.clock() if actual_at is None else float(actual_at)
        if actual < record["decided_at"] or actual < record["scheduled_at"]:
            raise ValueError("actual dispatch predates schedule decision")
        record["actual_dispatch_monotonic"] = actual

    def finished(self, index: int, finished_at: float | None = None) -> None:
        if index not in self._active:
            raise RuntimeError("worker was not active")
        if "actual_dispatch_monotonic" not in self._records[index]:
            raise RuntimeError("worker finished before dispatch started")
        finished = self.clock() if finished_at is None else float(finished_at)
        if finished < self._records[index]["actual_dispatch_monotonic"]:
            raise ValueError("worker finish predates dispatch")
        self._active.remove(index)
        self._records[index]["submit_finished_monotonic"] = finished

    def submission_failed(self, index: int, reason: str, failed_at: float | None = None) -> None:
        """Release a reservation that never entered an executor worker."""
        if index not in self._active:
            raise RuntimeError("worker was not active")
        record = self._records[index]
        if "actual_dispatch_monotonic" in record:
            raise RuntimeError("started work cannot be reclassified as executor failure")
        self._active.remove(index)
        record.update(action="not_sent_after_stop", executor_error=str(reason),
                      failed_at_monotonic=self.clock() if failed_at is None else float(failed_at))

    def stop(self, reason: str, now: float | None = None) -> None:
        if self.started_at is None:
            raise RuntimeError("schedule not started")
        if self.stop_reason is not None:
            return
        when = self.clock() if now is None else float(now)
        self.stop_reason = str(reason)
        while self._next < self.count:
            self._records[self._next] = {
                "index": self._next, "scheduled_at": self._scheduled_at(self._next),
                "decided_at": when, "action": "not_sent_after_stop",
                "lag_seconds": max(0.0, when - self._scheduled_at(self._next)),
                "active_workers": len(self._active),
            }
            self._next += 1

    def records(self) -> list[dict[str, Any]]:
        return [dict(self._records[index]) for index in sorted(self._records)]

    def summary(self) -> dict[str, Any]:
        records = self.records()
        counts = {name: sum(record["action"] == name for record in records)
                  for name in ("dispatch", "unsent_capacity", "unsent_lag", "not_sent_after_stop")}
        counts["reserved_for_submission"] = sum(bool(record.get("reserved_for_submission")) for record in records)
        counts["actual_started"] = sum("actual_dispatch_monotonic" in record for record in records)
        counts["planned"] = self.count
        counts["accounted"] = sum(counts[name] for name in ("dispatch", "unsent_capacity", "unsent_lag", "not_sent_after_stop"))
        return {"started_at_monotonic": self.started_at, "interval_seconds": self.interval,
                "submission_workers": self.workers, "stop_reason": self.stop_reason,
                "counts": counts,
                "dispatch_lag_seconds": _percentiles(record["lag_seconds"] for record in records if record["action"] == "dispatch")}
