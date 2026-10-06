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


@pytest.mark.parametrize("route", ["pool", "remote", "remote_after_401"])
@pytest.mark.parametrize("success", [True, False])
def test_late_capacity_observation_cannot_erase_consumption(tmp_path, monkeypatch, route, success):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    from services.openai_backend_api import InvalidAccessTokenError
    from unittest.mock import Mock

    account_id = "12345678-1234-5678-9234-567812345678"
    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{
        "access_token": "fixture-old", "account_id": account_id, "user_id": "fixture-user",
        "managed_owner": "fixture-owner", "source_type": "web", "status": "正常", "type": "Plus",
        **snapshot(1),
    }])
    token = ["fixture-old"]
    def refresh(value, **kw):
        if kw.get("force"):
            token[0] = service._apply_refreshed_tokens(value, {"access_token": "fixture-new"}, "fixture")
        return token[0]
    monkeypatch.setattr(service, "refresh_access_token", refresh)
    remote = {"quota": 1, "status": "正常", "account_id": account_id, "user_id": "fixture-user",
              "limits_progress": [{"feature_name": "image_gen", "remaining": 1}]}
    applied = [False]
    def read():
        if not applied[0]:
            applied[0] = True
            service.mark_image_result(token[0], success)
        return deepcopy(remote)
    def backend(value):
        obj = Mock()
        if route == "remote_after_401" and value == "fixture-old":
            obj.get_user_info.side_effect = InvalidAccessTokenError("fixture unauthorized")
        else:
            obj.get_user_info.side_effect = read
        return obj
    monkeypatch.setattr("services.openai_backend_api.OpenAIBackendAPI", backend)
    # Preserve the validator normally exposed by the real constructor.
    backend._validated_account_id = lambda value: value
    monkeypatch.setattr(service, "_verified_chat_info", lambda value: (("fixture-user", account_id), read()))
    ref = service.list_pool_accounts()[0]["account_ref"]
    def observe():
        if route == "pool":
            service._refresh_pool_chat(ref)
        else:
            service.fetch_remote_info(token[0])
    observe()
    saved = AccountService(JSONStorageBackend(tmp_path / "accounts.json")).get_account(token[0])
    assert saved["capacity_used_since_observation"] is True
    assert saved["quota"] == (0 if success else 1)
    assert saved["status"] == ("限流" if success else "正常")
    assert (saved["success"], saved["fail"]) == ((1, 0) if success else (0, 1))
    assert owned_accounts.observed_capacity(saved)["state"] == "stale"
    # The following real observation begins after consumption and can restore
    # eligibility. Rejecting the stale one must not permanently disable it.
    observe()
    latest = service.get_account(token[0])
    assert latest["capacity_used_since_observation"] is False
    assert latest["quota"] == 1 and latest["status"] == "正常"
    assert owned_accounts.observed_capacity(latest)["state"] == "observed"


@pytest.mark.parametrize("route", ["pool", "remote"])
def test_late_positive_observation_keeps_newer_zero_observation(tmp_path, monkeypatch, route):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    from unittest.mock import Mock

    account_id = "12345678-1234-5678-9234-567812345678"
    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{
        "access_token": "fixture", "account_id": account_id, "user_id": "fixture-user",
        "managed_owner": "fixture-owner", "source_type": "web", "status": "正常", "type": "Plus",
        **snapshot(1, (NOW - timedelta(seconds=1)).isoformat()),
    }])
    monkeypatch.setattr(service, "refresh_access_token", lambda value, **kw: value)
    def read():
        service.update_account("fixture", {**snapshot(0), "status": "限流"}, quiet=True)
        return {"quota": 1, "status": "正常", "account_id": account_id, "user_id": "fixture-user",
                "limits_progress": [{"feature_name": "image_gen", "remaining": 1}]}
    backend = Mock()
    backend.return_value.get_user_info.side_effect = read
    backend._validated_account_id.side_effect = lambda value: value
    monkeypatch.setattr("services.openai_backend_api.OpenAIBackendAPI", backend)
    monkeypatch.setattr(service, "_verified_chat_info", lambda value: (("fixture-user", account_id), read()))
    if route == "pool":
        service._refresh_pool_chat(service.list_pool_accounts()[0]["account_ref"])
    else:
        service.fetch_remote_info("fixture")
    saved = AccountService(JSONStorageBackend(tmp_path / "accounts.json")).get_account("fixture")
    assert saved["quota"] == 0 and saved["status"] == "限流"
    assert saved["limits_progress"][0]["remaining"] == 0
    assert saved["capacity_observed_at"] == NOW.isoformat()
    assert saved["capacity_used_since_observation"] is False


@pytest.mark.parametrize("route", ["pool", "remote"])
def test_newer_zero_observation_survives_older_positive_completing_first(tmp_path, monkeypatch, route):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread
    from unittest.mock import Mock
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend

    account_id = "12345678-1234-5678-9234-567812345678"
    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{
        "access_token": "fixture", "account_id": account_id, "user_id": "fixture-user",
        "managed_owner": "fixture-owner", "source_type": "web", "status": "正常", "type": "Plus",
        **snapshot(1, (NOW - timedelta(seconds=1)).isoformat()),
    }])
    monkeypatch.setattr(service, "refresh_access_token", lambda value, **kw: value)
    first_started, second_started, first_saved = Event(), Event(), Event()
    times = {}
    monkeypatch.setattr(owned_accounts, "utc_now", lambda: times[current_thread().name])
    def read():
        if times[current_thread().name] == NOW.isoformat():
            first_started.set()
            assert second_started.wait(5)
            return {**snapshot(1), "status": "正常", "account_id": account_id, "user_id": "fixture-user"}
        second_started.set()
        assert first_saved.wait(5)
        return {**snapshot(0), "status": "限流", "account_id": account_id, "user_id": "fixture-user"}
    backend = Mock()
    backend.return_value.get_user_info.side_effect = read
    backend._validated_account_id.side_effect = lambda value: value
    monkeypatch.setattr("services.openai_backend_api.OpenAIBackendAPI", backend)
    monkeypatch.setattr(service, "_verified_chat_info", lambda value: (("fixture-user", account_id), read()))
    ref = service.list_pool_accounts()[0]["account_ref"]
    def observe(second):
        times[current_thread().name] = (NOW + timedelta(seconds=int(second))).isoformat()
        try:
            if route == "pool":
                service._refresh_pool_chat(ref)
            else:
                service.fetch_remote_info("fixture")
        finally:
            if not second:
                first_saved.set()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(observe, False)
        assert first_started.wait(5)
        second = executor.submit(observe, True)
        first.result(timeout=10)
        second.result(timeout=10)
    saved = AccountService(JSONStorageBackend(tmp_path / "accounts.json")).get_account("fixture")
    assert saved["quota"] == 0 and saved["limits_progress"][0]["remaining"] == 0
    assert saved["capacity_observed_at"] == (NOW + timedelta(seconds=1)).isoformat()
    assert saved["capacity_used_since_observation"] is False


def test_reimport_old_export_does_not_rewind_consumption_or_reenable_late_observation(tmp_path):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend

    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{
        "access_token": "fixture", "managed_owner": "fixture-owner", "source_type": "web",
        "status": "正常", "type": "Plus", **snapshot(1),
    }])
    old = service.get_account("fixture")
    expected = service._capacity_observation_revision(old)
    service.mark_image_result("fixture", True)
    result = service.add_account_items([{**old, "refresh_token": "fixture-replacement"}])
    assert result["added"] == 0 and result["skipped"] == 1
    service.update_account("fixture", {**snapshot(1), "status": "正常"},
                           expected_credentials=("fixture", str(old.get("account_id") or "")),
                           expected_capacity_observation=expected)
    saved = AccountService(JSONStorageBackend(tmp_path / "accounts.json")).get_account("fixture")
    assert saved["quota"] == 0 and saved["status"] == "限流"
    assert saved["success"] == 1 and saved["fail"] == 0
    assert saved["capacity_used_since_observation"] is True
    assert saved["refresh_token"] == "fixture-replacement"


@pytest.mark.parametrize("download_success", [True, False])
@pytest.mark.parametrize("remaining", [0, 4])
@pytest.mark.parametrize("settlement_first", [False, True])
def test_download_settlement_keeps_observation_after_asset_consumption(
        tmp_path, download_success, remaining, settlement_first):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend

    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{"access_token": "fixture", "managed_owner": "fixture-owner",
        "source_type": "web", "status": "正常", "type": "Plus", **snapshot(5)}])
    before = service._capacity_observation_revision(service.get_account("fixture"))
    service.mark_image_capacity_consumed("fixture")
    consumed = service.get_account("fixture")
    assert consumed["capacity_used_since_observation"] is True
    assert consumed["quota"] == 4
    assert (consumed["success"], consumed["fail"]) == (0, 0)
    # A metadata read that started before the asset was produced must fail CAS,
    # even though download outcome counters have not changed yet.
    service.update_account("fixture", snapshot(5), quiet=True, expected_capacity_observation=before)
    assert service.get_account("fixture")["capacity_used_since_observation"] is True
    current = service._capacity_observation_revision(service.get_account("fixture"))
    # The metadata read started after the generation consumed capacity. Merely
    # settling its download while that read is in flight cannot make it stale.
    if settlement_first:
        service.mark_image_result("fixture", download_success, release_slot=False, capacity_consumed=True)
    service.update_account("fixture", {**snapshot(remaining), "status": "限流" if remaining == 0 else "正常"},
        quiet=True, expected_capacity_observation=current)
    fresh = service.get_account("fixture")
    assert fresh["capacity_used_since_observation"] is False
    # A late save/failure must neither subtract the same generation again nor
    # invalidate the newer real observation (including a real zero).
    if not settlement_first:
        service.mark_image_result("fixture", download_success, release_slot=False, capacity_consumed=True)
    saved = AccountService(JSONStorageBackend(tmp_path / "accounts.json")).get_account("fixture")
    assert saved["quota"] == remaining and saved["status"] == fresh["status"]
    assert saved["capacity_used_since_observation"] is False
    assert saved["last_used_at"] == consumed["last_used_at"]
    assert (saved["success"], saved["fail"]) == ((1, 0) if download_success else (0, 1))


def test_two_asset_consumptions_before_download_invalidate_interleaved_observation(tmp_path):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{"access_token": "fixture", "managed_owner": "fixture-owner",
        "source_type": "web", "status": "正常", "type": "Plus", **snapshot(5)}])
    service.mark_image_capacity_consumed("fixture")
    between = service._capacity_observation_revision(service.get_account("fixture"))
    service.mark_image_capacity_consumed("fixture")
    service.update_account("fixture", snapshot(4), quiet=True, expected_capacity_observation=between)
    saved = service.get_account("fixture")
    assert saved["quota"] == 3 and saved["capacity_used_since_observation"] is True


def test_final_image_recheck_reports_consumed_observation_without_exposing_identity(tmp_path):
    from services.account_service import AccountService
    from services.request_context import AdmissionLost
    from services.storage.json_storage import JSONStorageBackend
    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{"access_token": "private-fixture", "managed_owner": "fixture-owner",
        "source_type": "web", "status": "正常", "type": "Plus", **snapshot(5)}])
    service.require_image_account("private-fixture", "gpt-image-2")
    service.mark_image_capacity_consumed("private-fixture")
    with pytest.raises(AdmissionLost) as rejected:
        service.require_image_account("private-fixture", "gpt-image-2")
    assert rejected.value.reason == "image_capability_unavailable"
    assert rejected.value.capability_reason == "stale_consumed"
    assert "private-fixture" not in str(rejected.value)


@pytest.mark.parametrize("after_image_consumption", [False, True])
def test_text_usage_does_not_invalidate_image_metadata_read(tmp_path, after_image_consumption):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{"access_token": "fixture", "managed_owner": "fixture-owner",
        "source_type": "web", "status": "正常", "type": "Plus", **snapshot(5)}])
    if after_image_consumption:
        service.mark_image_capacity_consumed("fixture")
    expected = service._capacity_observation_revision(service.get_account("fixture"))
    service.mark_text_used("fixture")
    service.update_account("fixture", snapshot(3), quiet=True, expected_capacity_observation=expected)
    saved = service.get_account("fixture")
    assert saved["quota"] == 3
    assert saved["capacity_used_since_observation"] is False


def test_old_account_import_cannot_rewind_asset_consumption_counter(tmp_path):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{"access_token": "fixture", "managed_owner": "fixture-owner",
        "source_type": "web", "status": "正常", "type": "Plus", **snapshot(5)}])
    service.mark_image_capacity_consumed("fixture")
    service.add_account_items([{"access_token": "fixture", "capacity_consumption_count": 0, **snapshot(5)}])
    saved = AccountService(JSONStorageBackend(tmp_path / "accounts.json")).get_account("fixture")
    assert saved["capacity_consumption_count"] == 1
    assert saved["capacity_used_since_observation"] is True
    assert saved["quota"] == 4


def test_refresh_rotation_and_generic_update_preserve_consumption_counter(tmp_path):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    service.add_account_items([{"access_token": "fixture-old", "managed_owner": "fixture-owner",
        "source_type": "web", "status": "正常", "type": "Plus", **snapshot(5)}])
    service.mark_image_capacity_consumed("fixture-old")
    service.update_account("fixture-old", {"capacity_consumption_count": 0}, quiet=True)
    token = service._apply_refreshed_tokens("fixture-old", {"access_token": "fixture-new"}, "fixture")
    saved = service.get_account(token)
    assert token == "fixture-new"
    assert saved["capacity_consumption_count"] == 1 and saved["quota"] == 4
    assert saved["capacity_used_since_observation"] is True


@pytest.mark.parametrize("value", [None, -1, True, "invalid", 1.5])
def test_invalid_consumption_counter_normalizes_without_breaking_account(tmp_path, value):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    service = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    saved = service._normalize_account({"access_token": "fixture", "capacity_consumption_count": value})
    assert saved["capacity_consumption_count"] == 0
