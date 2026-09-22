"""Reserve admission uses original turns and receipts; all transport is fake."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import time
from types import SimpleNamespace

import pytest

from services.codex_service import CodexService, CodexServiceError, _project_limits
from services.durable_forward import envelope
from services.request_context import AdmissionLost
from test.test_codex_service import account, observation, FakeAccounts, FakeSession, FakeResponse, SessionFactory
from test.test_pool_admission_identity import runtime, legacy

LUNA = "gpt-5.6-luna"
RESERVE = "gpt-reserve"
SOL = "gpt-5.6-sol"


def reserve_row(token="reserve", *, main_used=100, allowed=True, mapping=LUNA):
    return legacy(token, codex_observation=observation(
        state="limited" if main_used == 100 else "observed",
        models=[{"id": m, "reasoning_efforts": ["low", "high"]} for m in (LUNA, SOL, RESERVE)],
        limits=[{"id": "codex", "windows": [{"used_percent": main_used}], "allowed": main_used < 100},
                {"id": RESERVE, "allowed": allowed, "normal_model_slug": mapping,
                 "windows": [{"used_percent": 0, "resets_at": int(time.time()) + 3600}]}]))


def test_projection_preserves_flags_mapping_and_drops_other_fields():
    limits, limited = _project_limits({
        "rate_limit": {"allowed": False, "primary_window": {"used_percent": 100}},
        "additional_rate_limits": [{"limit_name": RESERVE, "normal_model_slug": LUNA,
            "private_data": "do-not-project", "rate_limit": {"allowed": True, "limit_reached": False,
                "primary_window": {"used_percent": 0}}}]})
    assert limited
    assert limits[1]["allowed"] is True and limits[1]["normal_model_slug"] == LUNA
    assert "private_data" not in json.dumps(limits)
    projected = CodexService.account_projection(reserve_row())
    assert projected["limits"][1]["allowed"] is True
    assert projected["limits"][1]["normal_model_slug"] == LUNA


@pytest.mark.parametrize("mutation,state,reason", [
    (lambda r: r.update(managed_disabled=True), "unavailable", "disabled"),
    (lambda r: r["codex_observation"].update(state="auth_required"), "unavailable", "auth_required"),
    (lambda r: r["codex_observation"].update(state="read_failed"), "unknown", "read_failed"),
    (lambda r: r["codex_observation"].update(observed_at=(datetime.now(timezone.utc)-timedelta(minutes=10)).isoformat()), "unknown", "stale"),
    (lambda r: r["codex_observation"].update(observed_at=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat()), "unknown", "stale"),
    (lambda r: r["codex_observation"]["limits"][1].update(allowed=False), "unavailable", "reserve_not_allowed"),
    (lambda r: r["codex_observation"]["limits"][1].pop("allowed"), "unknown", "reserve_not_allowed"),
    (lambda r: r["codex_observation"]["limits"][1].update(normal_model_slug=SOL), "unavailable", "reserve_mapping_unavailable"),
    (lambda r: r["codex_observation"]["limits"][1]["windows"][0].update(used_percent=100), "unavailable", "reserve_limited"),
    (lambda r: r["codex_observation"]["limits"][1]["windows"][0].update(resets_at=1), "unknown", "stale"),
    (lambda r: r["codex_observation"].update(models=[{"id": LUNA}]), "unavailable", "reserve_mapping_unavailable"),
])
def test_reserve_requires_current_positive_evidence(mutation, state, reason):
    row = reserve_row()
    mutation(row)
    service = CodexService(FakeAccounts([row]), SessionFactory([]))
    decision = service.quota_decision(row, LUNA)
    assert (decision["state"], decision["reason"]) == (state, reason)
    assert service._eligible_account(row, LUNA, allow_probe=False) is None


def test_main_and_reserve_are_model_specific_and_generic_excludes_reserve():
    row = reserve_row()
    service = CodexService(FakeAccounts([row]), SessionFactory([]))
    for model in (LUNA, RESERVE):
        assert service.quota_decision(row, model)["quota_bucket"] == RESERVE
    for model in ("", SOL, "auto", "gpt-6-astra"):
        assert service.quota_decision(row, model)["state"] != "available"
    recovered = reserve_row(main_used=20)
    assert service.quota_decision(recovered, LUNA)["quota_bucket"] == "codex"
    assert service.quota_decision(recovered, RESERVE)["state"] == "unavailable"


@pytest.mark.parametrize("main_used,upstream", [(100, RESERVE), (20, LUNA)])
def test_wire_mapping_preserves_original_payload_high_and_headers(main_used, upstream):
    row = reserve_row(main_used=main_used)
    session = FakeSession(post_response=FakeResponse(payload={"id": "response-test", "output": []}))
    service = CodexService(FakeAccounts([row]), SessionFactory([session]))
    payload = {"model": LUNA, "input": [], "reasoning": {"effort": "high"}, "tools": []}
    original = deepcopy(payload)
    service.submit({"id": "test", "role": "user"}, payload, {})
    assert payload == original
    assert len(session.calls) == 1
    wire = json.loads(session.calls[0][2]["data"])
    assert wire == {**payload, "model": upstream}
    assert session.calls[0][2]["headers"]["x-openai-codex-luna-reserve"] == "1"


def test_unknown_session_never_moves_to_another_account():
    rows = [reserve_row("r"), reserve_row("n", main_used=20)]
    service = CodexService(FakeAccounts(rows), SessionFactory([]))
    identity, headers, payload = {"id": "test"}, {"session-id": "session"}, {"model": LUNA}
    affinity = service._affinity_key(identity, headers, payload)
    service.accounts.accounts["r"]["codex_affinities"] = {affinity: {"state": "unknown"}}
    with pytest.raises(CodexServiceError) as error:
        service._select_account(identity, headers, payload)
    assert error.value.code == "codex_session_outcome_unknown"


def submit(rt, name, owner="source-a", model=LUNA, session=None):
    payload = {"model": model, "input": [], "reasoning": {"effort": "high"}}
    req = SimpleNamespace(headers={"x-client-request-id": name, "session-id": session or name})
    body = envelope({"id": owner, "role": "user"}, payload, req, "codex")
    rt.service.submit(owner, body, source=owner)
    return body


def test_pool_prefers_dispatchable_reserve_then_normal_with_shared_turns(runtime):
    rt = runtime([reserve_row("r"), reserve_row("n", main_used=20)])
    submit(rt, "first")
    first = rt.admission.claim_next()
    assert first.selected_account()["access_token"] == "r"
    first.before_send()
    submit(rt, "second", owner="source-b")
    second = rt.admission.claim_next()
    assert second.selected_account()["access_token"] == "n"
    second.before_send()
    submit(rt, "third")
    assert rt.admission.claim_next() is None
    metrics = rt.admission.model_resources([LUNA])[LUNA]
    assert metrics["eligible_accounts"] == 2
    assert metrics["occupied"] == 2 and metrics["dispatchable_now"] == 0
    assert {row["dispatch_reason"] for row in metrics["accounts"]} == {"account_busy"}


def test_reserve_preference_does_not_preempt_other_source(runtime):
    rt = runtime([reserve_row("r"), reserve_row("n", main_used=20)])
    submit(rt, "luna", owner="source-b")
    submit(rt, "sol", owner="source-a", model=SOL)
    first = rt.admission.claim_next()
    assert first.owner == "source-a" and first.selected_account()["access_token"] == "n"


def test_server_limit_caps_reserve_and_main_together(runtime):
    rt = runtime([reserve_row("r"), reserve_row("n", main_used=20)])
    rt.admission.settings = lambda: {"image_account_concurrency": 4, "codex_max_concurrency": 1}
    submit(rt, "first")
    rt.admission.claim_next().before_send()
    submit(rt, "second", owner="source-b")
    assert rt.admission.claim_next() is None
    metrics = rt.admission.model_resources([LUNA])[LUNA]
    assert metrics["eligible_accounts"] == 2 and metrics["dispatchable_now"] == 0


def test_before_send_rechecks_reserve_mode_without_marking_sent(runtime):
    rt = runtime([reserve_row()])
    submit(rt, "first")
    claim = rt.admission.claim_next()
    claim.expected_codex_quota_bucket = RESERVE
    selected = claim.selected_account()
    rt.accounts.update_account(selected["access_token"], {"codex_observation": reserve_row(main_used=10)["codex_observation"]})
    with pytest.raises(AdmissionLost):
        claim.before_send()
    with rt.store.connect() as db:
        saved = rt.store.read_receipt(db, "text", claim.owner, claim.request_id)
    assert saved["_submission_started"] is False


def test_model_projection_deduplicates_physical_account_and_exposes_dispatch(runtime):
    rt = runtime([reserve_row()])
    rt.native.admission = rt.admission
    catalog = {m["id"]: m for m in rt.native.management_models()["items"]}
    assert catalog[LUNA]["available_accounts"] == 1
    assert catalog[LUNA]["eligible_accounts"] == 1 and catalog[LUNA]["dispatchable_now"] == 1
    assert catalog[SOL]["available_accounts"] == 0
    assert catalog[LUNA]["accounts"][0]["quota_bucket"] == RESERVE
    rows = [reserve_row("a"), reserve_row("b")]
    rows[1]["account_id"] = rows[0]["account_id"]
    service = CodexService(FakeAccounts(rows), SessionFactory([]))
    models = {m["id"]: m for m in service.management_models()["items"]}
    assert models[LUNA]["supported_accounts"] == 1


def test_durable_reserve_wire_and_existing_binding_stay_on_original_account(runtime):
    # Start on normal quota, then add a preferred Reserve sibling. Continuation
    # still consumes the original physical account, including its Reserve.
    rt = runtime([reserve_row("original", main_used=20)])
    submit(rt, "first", session="conversation")
    normal = FakeSession(post_response=FakeResponse(payload={"id": "response-first", "output": []}))
    rt.factory.sessions.append(normal)
    first = rt.admission.claim_next()
    rt.admission.execute(first)
    assert rt.service.read("source-a", "first")["status"] == "succeeded"
    assert json.loads(normal.calls[0][2]["data"])["model"] == LUNA
    rt.accounts.add_account_items([reserve_row("sibling")])
    rt.accounts.update_account("original", {"codex_observation": reserve_row()["codex_observation"]}, quiet=True)
    submit(rt, "continuation", session="conversation")
    reserve = FakeSession(post_response=FakeResponse(payload={"id": "response-second", "output": []}))
    rt.factory.sessions.append(reserve)
    continuation = rt.admission.claim_next()
    assert continuation.selected_account()["access_token"] == "original"
    rt.admission.execute(continuation)
    receipt = rt.service.read("source-a", "continuation")
    assert receipt["status"] == "succeeded"
    assert receipt["provider_account_identity"] == first.receipt()["provider_account_identity"]
    assert len(reserve.calls) == 1
    wire = json.loads(reserve.calls[0][2]["data"])
    assert wire["model"] == RESERVE and wire["reasoning"] == {"effort": "high"}
    assert rt.admission.model_resources([LUNA])[LUNA]["occupied"] == 0


def test_model_occupancy_keeps_running_turn_when_quota_later_exhausts(runtime):
    rt = runtime([reserve_row("original", main_used=20)])
    submit(rt, "held", model=SOL)
    rt.admission.claim_next().before_send()
    rt.accounts.update_account("original", {"codex_observation": reserve_row()["codex_observation"]}, quiet=True)
    metrics = rt.admission.model_resources([SOL])[SOL]
    assert metrics["eligible_accounts"] == 0
    assert metrics["occupied"] == 1 and metrics["dispatchable_now"] == 0
