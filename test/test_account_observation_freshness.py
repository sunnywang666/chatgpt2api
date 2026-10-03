"""Stored quota evidence must not masquerade as a fresh observation.

Pure projection regressions; no account refresh, model request or scheduler run.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from services import owned_accounts


NOW = datetime(2026, 9, 30, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr(owned_accounts, "datetime", FrozenDateTime)


def snapshot(remaining=7, observed_at=None):
    return {
        "limits_progress": [{"feature_name": "image_gen", "remaining": remaining, "reset_after": 60}],
        "capacity_observed_at": NOW.isoformat() if observed_at is None else observed_at,
        "capacity_used_since_observation": False,
        "capacity_read_failed_at": None,
        "managed_disabled": False,
        "quota": remaining,
    }


@pytest.mark.parametrize("age,expected", [(0, True), (299, True), (300, True), (301, False), (-1, False)])
def test_freshness_is_bounded_on_both_sides(age, expected):
    value = (NOW - timedelta(seconds=age)).isoformat()
    assert owned_accounts.observation_is_fresh(value) is expected


@pytest.mark.parametrize("value", [None, "", "not-a-date", "0001-01-01T00:00:00+14:00", "9999-12-31T23:59:59-14:00"])
def test_bad_or_unrepresentable_timestamps_are_not_fresh(value):
    assert not owned_accounts.observation_is_fresh(value)


@pytest.mark.parametrize("value", ["2026-09-30T08:00:00Z", "2026-09-30 08:00:00", "2026-09-30T16:00:00+08:00"])
def test_existing_timestamp_formats_represent_the_same_observation(value):
    assert owned_accounts.observation_is_fresh(value)


@pytest.mark.parametrize("remaining", [0, 7, 7.0])
def test_fresh_observed_values_include_real_zero(remaining):
    result = owned_accounts.observed_capacity(snapshot(remaining))
    assert result["state"] == "observed"
    assert result["remaining"] == int(remaining)


@pytest.mark.parametrize("observed_at", ["", "bad-time", "2026-09-30T07:54:59Z", "2026-09-30T08:00:01Z"])
def test_old_missing_or_future_time_keeps_value_but_is_stale(observed_at):
    account = snapshot(0, observed_at)
    before = deepcopy(account)
    result = owned_accounts.observed_capacity(account)
    assert result["state"] == "stale"
    assert result["remaining"] == 0
    assert result["observed_at"] == observed_at
    assert result["failed_at"] is None
    assert account == before


@pytest.mark.parametrize("remaining", [None, True, -1, 1.5, "7", float("nan"), float("inf")])
def test_unknown_is_not_fabricated_zero(remaining):
    result = owned_accounts.observed_capacity(snapshot(remaining))
    assert result["state"] == "unknown" and result["remaining"] is None


def test_used_and_failed_observations_keep_their_existing_priority():
    account = snapshot()
    account["capacity_used_since_observation"] = True
    assert owned_accounts.observed_capacity(account)["state"] == "stale"
    account["capacity_read_failed_at"] = NOW.isoformat()
    result = owned_accounts.observed_capacity(account)
    assert result["state"] == "read_failed" and result["remaining"] == 7


def test_new_success_can_restore_observed_projection_without_mutating_account():
    account = snapshot(0, "2026-09-30T07:54:00Z")
    assert owned_accounts.observed_capacity(account)["state"] == "stale"
    # Simulate the already existing refresh writer saving a new observation.
    # This does not test or claim actual quota refresh or automatic dispatch.
    account["limits_progress"][0]["remaining"] = 7
    account["capacity_observed_at"] = NOW.isoformat()
    before = deepcopy(account)
    assert owned_accounts.observed_capacity(account)["state"] == "observed"
    assert account == before
