"""Real account JSON + native executor; no production accounts or requests."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from services.account_service import AccountService
from services.codex_service import CodexService
from services.durable_forward import envelope, raw_receipt
from services.log_service import LogService
from services.storage.json_storage import JSONStorageBackend
from services.text_task_service import TextTaskService
from test.test_codex_service import account, FakeResponse, FakeSession, SessionFactory
from test.test_pool_admission import build, Clock


def legacy(token="fixture-codex-a", *, persisted_ref=True, **updates):
    row = account(token, **{"source_type": "codex", **updates})
    if persisted_ref:
        row["managed_pool_account_ref"] = AccountService.pool_account_ref(row)
    return row


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    clock = Clock()
    logs = LogService(tmp_path / "calls.jsonl")
    monkeypatch.setattr("services.account_service.log_service", logs)
    monkeypatch.setattr("services.log_service.log_service", logs)
    monkeypatch.setattr(AccountService, "_get_cumulative_file", lambda self: tmp_path / "total")

    def open_runtime(rows=None):
        path = tmp_path / "accounts.json"
        if rows is not None:
            path.write_text(json.dumps(rows))
        accounts, store, admission = build(tmp_path, clock)
        factory = SessionFactory([])
        native = CodexService(accounts, factory)
        admission.codex = native
        monkeypatch.setattr("services.codex_service.codex_service", native)
        monkeypatch.setattr(accounts, "refresh_codex_access_token", lambda token, **kw: accounts.get_account(token)["access_token"])
        service = TextTaskService(store.path, admission=admission, clock=clock)
        admission.register("text", lambda ctx, payload: service._run(ctx.owner, ctx.request_id, payload))
        return SimpleNamespace(accounts=accounts, store=store, admission=admission, service=service,
                               clock=clock, native=native, factory=factory, path=path)
    return open_runtime


def submit(rt, name="original", *, session="original-session", previous=None):
    identity = {"id": "ordinary-key", "role": "user"}
    payload = {"model": "gpt-5.6-codex", "input": [], "reasoning": {"effort": "high"}}
    if previous:
        payload["previous_response_id"] = previous
    request = SimpleNamespace(headers={"x-client-request-id": name, "session-id": session})
    body = envelope(identity, payload, request, "codex")
    rt.service.submit(identity["id"], body)
    return body, payload, identity


def response(rt, name="response-original"):
    session = FakeSession(post_response=FakeResponse(payload={"id": name, "output": []}))
    rt.factory.sessions.append(session)
    return session


@pytest.mark.parametrize("persisted_ref", [True, False])
def test_legacy_codex_without_image_identity_is_read_only_and_executes(runtime, persisted_ref):
    row = legacy(persisted_ref=persisted_ref)
    rt = runtime([row])
    original_bytes = rt.path.read_bytes()
    assert rt.native._eligible_account(rt.accounts.list_accounts()[0], "gpt-5.6-codex", allow_probe=False)
    for _ in range(2):
        snapshot = rt.admission.resource_snapshot()
        assert snapshot["codex"]["slots_total"] == 1
        assert snapshot["accounts"][0]["codex"]["dispatchable_now"] == 1
        assert snapshot["image"]["slots_total"] == 0
    assert rt.path.read_bytes() == original_bytes
    assert "provider_account_identity" not in rt.accounts.list_accounts()[0]
    projected_identity = snapshot["accounts"][0]["provider_account_identity"]
    assert projected_identity and row["access_token"] not in projected_identity
    _, payload, _ = submit(rt)
    sent = response(rt)
    context = rt.admission.claim_next()
    assert context is not None
    assert context.selected_account()["access_token"] == row["access_token"]
    rt.admission.execute(context)
    receipt = rt.service.read("ordinary-key", "original")
    assert receipt["status"] == "succeeded"
    assert receipt["provider_account_identity"] == projected_identity
    assert len(sent.calls) == 1
    assert json.loads(sent.calls[0][2]["data"]) == payload


def test_new_codex_only_account_automatically_serves_existing_full_pool_wait(runtime):
    rt = runtime([legacy()])
    submit(rt, "held")
    held = rt.admission.claim_next()
    assert held is not None
    held.before_send()
    submit(rt, "waiting", session="other-session")
    assert rt.admission.claim_next() is None
    sent = response(rt)
    rt.admission.start()
    try:
        # Normal account import by another worker; no new image binding,
        # request resubmission, queue wake command or consumer restart.
        writer = AccountService(JSONStorageBackend(rt.path))
        assert writer.add_account_items([legacy("fixture-codex-b")])["added"] == 1
        deadline = time.monotonic() + 4
        while rt.service.read("ordinary-key", "waiting")["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(.02)
        while rt.service.read("ordinary-key", "waiting")["status"] == "running" and time.monotonic() < deadline:
            time.sleep(.02)
        receipt = rt.service.read("ordinary-key", "waiting")
        assert receipt["status"] == "succeeded"
        assert receipt["provider_account_identity"] != held.receipt()["provider_account_identity"]
        assert rt.admission.resource_snapshot()["codex"]["slots_total"] == 2
        assert len(sent.calls) == 1
        assert held.receipt()["_claim_id"] == held.claim
    finally:
        rt.admission.stop()


def test_restart_and_other_worker_token_rotation_keep_response_and_session_owner(runtime):
    rt = runtime([legacy()])
    body, _, identity = submit(rt)
    response(rt)
    first = rt.admission.claim_next()
    assert first is not None
    original_identity = first.receipt()["provider_account_identity"]
    original_ref = first.selected_account()["managed_pool_account_ref"]
    rt.admission.execute(first)
    writer = AccountService(JSONStorageBackend(rt.path))
    writer._apply_refreshed_tokens("fixture-codex-a", {"access_token": "fixture-rotated"}, "isolated-test")
    writer.add_account_items([legacy("fixture-codex-b")])
    # The already-running worker follows rotation even without an image ID.
    assert rt.accounts.get_account("fixture-codex-a")["access_token"] == "fixture-rotated"
    rotated = rt.accounts.get_account("fixture-rotated")
    assert rotated["managed_pool_account_ref"] == original_ref
    assert body["client_conversation_id"] in rotated["codex_affinities"]
    assert hashlib.sha256(b"response-original").hexdigest() in rotated["codex_response_ids"]
    restarted = runtime()
    assert restarted.native._response_owner("response-original", identity) == "fixture-rotated"
    submit(restarted, "next", previous="response-original")
    writer.update_account("fixture-rotated", {"managed_disabled": True}, quiet=True)
    assert restarted.admission.claim_next() is None  # Available sibling cannot steal the continuation.
    writer.update_account("fixture-rotated", {"managed_disabled": False, "status": "正常"}, quiet=True)
    context = restarted.admission.claim_next()
    assert context is not None
    assert context.selected_account()["access_token"] == "fixture-rotated"
    assert context.receipt()["provider_account_identity"] == original_identity
    sent = response(restarted, "response-next")
    restarted.admission.execute(context)
    assert restarted.service.read("ordinary-key", "next")["status"] == "succeeded"
    assert len(sent.calls) == 1


@pytest.mark.parametrize("sent", [False, True])
def test_rotated_legacy_claim_restart_recovers_only_unsent_original(runtime, sent):
    rt = runtime([legacy()])
    submit(rt)
    context = rt.admission.claim_next()
    assert context is not None
    identity = context.receipt()["provider_account_identity"]
    if sent:
        context.before_send()
    writer = AccountService(JSONStorageBackend(rt.path))
    writer._apply_refreshed_tokens("fixture-codex-a", {"access_token": "fixture-rotated"}, "isolated-test")
    writer.add_account_items([legacy("fixture-codex-b")])
    rt.clock.now += rt.admission.CLAIM_SECONDS + 1
    restarted = runtime()
    recovered = restarted.admission.claim_next()
    receipt = raw_receipt(restarted.service, "ordinary-key", "original")
    assert receipt["provider_account_identity"] == identity
    if sent:
        assert recovered is None
        assert receipt["status"] == "unknown"
        assert restarted.factory.kwargs == []
    else:
        assert recovered is not None and recovered.request_id == "original"
        assert recovered.selected_account()["access_token"] == "fixture-rotated"
        actual = response(restarted)
        restarted.admission.execute(recovered)
        assert len(actual.calls) == 1
        assert restarted.service.read("ordinary-key", "original")["status"] == "succeeded"


def test_two_workers_claim_legacy_codex_only_account_once(runtime):
    first = runtime([legacy()])
    submit(first)
    second = runtime()
    barrier = Barrier(2)
    def claim(rt):
        barrier.wait(timeout=3)
        return rt.admission.claim_next()
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(claim, [first, second]))
    assert sum(ctx is not None for ctx in results) == 1
    assert first.admission.resource_snapshot()["codex"]["inflight"] == 1
    assert second.admission.resource_snapshot()["codex"]["inflight"] == 1
    assert first.factory.kwargs == second.factory.kwargs == []


@pytest.mark.parametrize("identity", [None, "account_previously_issued"])
def test_first_chat_binding_and_rotation_keep_projected_or_existing_identity(runtime, identity):
    rt = runtime([legacy(source_type="web", type="Plus", provider_account_identity=identity)])
    before = rt.path.read_bytes()
    projected = rt.admission.resource_snapshot()["accounts"][0]["provider_account_identity"]
    assert rt.path.read_bytes() == before
    if identity:
        assert projected == identity
    binding = rt.accounts.admission_binding(projected)
    assert rt.accounts.get_bound_account_identity(binding) == projected
    writer = AccountService(JSONStorageBackend(rt.path))
    writer._apply_refreshed_tokens("fixture-codex-a", {"access_token": "fixture-rotated"}, "isolated-test")
    restarted = runtime()
    assert restarted.accounts.get_bound_account_identity(binding) == projected
    assert restarted.admission.resource_snapshot()["accounts"][0]["provider_account_identity"] == projected
