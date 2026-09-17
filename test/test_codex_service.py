from __future__ import annotations

from datetime import datetime, timedelta, timezone
import base64
import hashlib
import json
import threading
import unittest
from unittest.mock import patch

from services.codex_service import (
    CODEX_COMPACT_URL,
    CODEX_MODELS_URL,
    CODEX_RESPONSES_URL,
    CODEX_USAGE_URL,
    CodexHTTPResponse,
    CodexService,
    CodexServiceError,
)


def observation(*, state="observed", minutes_ago=0, models=None, limits=None):
    return {
        "state": state,
        "observed_at": (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat(),
        "failed_at": None,
        "models": models or [{"id": "gpt-5.6-codex", "label": "GPT-5.6 Codex", "reasoning_efforts": ["medium"]}],
        "limits": limits or [{"id": "codex", "label": "Codex", "windows": [{"used_percent": 10, "window_seconds": 18000, "resets_at": 1770000000}]}],
        "error_code": None,
    }


class FakeAccounts:
    def __init__(self, accounts):
        self.accounts = {item["access_token"]: dict(item) for item in accounts}
        self.updates = []

    def list_accounts(self):
        return [dict(item) for item in self.accounts.values()]

    def get_account(self, token):
        item = self.accounts.get(token)
        return dict(item) if item else None

    def update_account(self, token, updates, quiet=False, *, expected_credentials=None):
        self.updates.append((token, updates, quiet))
        if token not in self.accounts:
            return None
        current = self.accounts[token]
        if expected_credentials is not None and expected_credentials != (
            current.get("access_token", ""), current.get("account_id", "")
        ):
            return dict(current)
        self.accounts[token].update(updates)
        return dict(self.accounts[token])

    def refresh_access_token(self, token, event=""):
        return token


class FakeResponse:
    def __init__(self, status=200, *, payload=None, chunks=None, content_type="application/json", raw_error=""):
        self.status_code = status
        self._payload = payload
        self._chunks = list(chunks) if chunks is not None else None
        self.headers = {"content-type": content_type, "set-cookie": "must-not-leak"}
        if raw_error:
            self.headers["x-secret-error"] = raw_error
        self.closed = False

    @property
    def content(self):
        if self._chunks is not None:
            return b"".join(self._chunks)
        return json.dumps(self._payload).encode()

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=None):
        if self._chunks is not None:
            yield from self._chunks
        else:
            yield self.content

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, *, gets=None, post_response=None, post_error=None):
        self.gets = list(gets or [])
        self.post_response = post_response
        self.post_error = post_error
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.gets.pop(0)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if self.post_error:
            raise self.post_error
        return self.post_response

    def close(self):
        self.closed = True


class SessionFactory:
    def __init__(self, sessions):
        self.sessions = list(sessions)
        self.kwargs = []

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        return self.sessions.pop(0)


def account(token="token-a", **updates):
    value = {
        "access_token": token,
        "account_id": f"account-{token}",
        "status": "正常",
        "quota": 0,
        "type": "free",
        "codex_observation": observation(),
    }
    value.update(updates)
    return value


class CodexObservationTests(unittest.TestCase):
    def test_refresh_projects_safe_models_and_general_usage(self):
        accounts = FakeAccounts([account()])
        session = FakeSession(gets=[
            FakeResponse(payload={"models": [{
                "slug": "gpt-5.6-codex", "display_name": "Codex", "supported_reasoning_efforts": [
                    {"reasoning_effort": "low"}, {"reasoning_effort": "high"}
                ], "internal_secret": "hidden",
            }]}),
            FakeResponse(payload={"rate_limits": [{
                "limit_id": "codex", "limit_name": "Weekly", "primary": {
                    "used_percent": 42, "window_minutes": 300, "reset_at": 1770000000,
                }, "raw_private": "hidden",
            }]}),
        ])
        service = CodexService(accounts, SessionFactory([session]))

        result = service.refresh_account("token-a")

        self.assertEqual(result["state"], "observed")
        self.assertEqual(result["models"], [{"id": "gpt-5.6-codex", "label": "Codex", "reasoning_efforts": ["low", "high"]}])
        self.assertEqual(result["limits"][0]["windows"][0]["window_seconds"], 18000)
        self.assertNotIn("internal_secret", json.dumps(result))
        self.assertNotIn("raw_private", json.dumps(result))
        self.assertEqual([call[1] for call in session.calls], [CODEX_MODELS_URL, CODEX_USAGE_URL])
        self.assertFalse(session.calls[0][2]["allow_redirects"])
        self.assertTrue(service._session.__self__ is service)  # sanity: real proxy/session path is used

    def test_model_specific_zero_does_not_mark_entire_route_limited(self):
        accounts = FakeAccounts([account()])
        session = FakeSession(gets=[
            FakeResponse(payload={"models": [{"id": "gpt-5.6-codex"}]}),
            FakeResponse(payload={"rate_limits": [{
                "limit_id": "codex-spark", "primary": {"used_percent": 100, "window_minutes": 10080}
            }, {
                "limit_id": "codex", "primary": {"used_percent": 12, "window_minutes": 300}
            }]}),
        ])
        service = CodexService(accounts, SessionFactory([session]))
        self.assertEqual(service.refresh_account("token-a")["state"], "observed")

    def test_current_wham_and_model_shapes_are_projected(self):
        accounts = FakeAccounts([account()])
        session = FakeSession(gets=[
            FakeResponse(payload={"models": [{
                "slug": "gpt-5.6-sol", "display_name": "Sol",
                "supported_reasoning_levels": [{"effort": "low"}, {"effort": "xhigh"}],
            }]}),
            FakeResponse(payload={
                "rate_limit": {
                    "allowed": True, "limit_reached": False,
                    "primary_window": {"used_percent": 0, "limit_window_seconds": 604800, "reset_at": 1790177176},
                },
                "additional_rate_limits": [{
                    "limit_name": "GPT-5.3-Codex-Spark",
                    "rate_limit": {"allowed": False, "primary_window": {"used_percent": 100, "limit_window_seconds": 18000}},
                }],
            }),
        ])
        result = CodexService(accounts, SessionFactory([session])).refresh_account("token-a")
        self.assertEqual(result["state"], "observed")
        self.assertEqual(result["models"][0]["reasoning_efforts"], ["low", "xhigh"])
        self.assertEqual(result["limits"][0]["id"], "codex")
        self.assertEqual(result["limits"][0]["windows"][0]["window_seconds"], 604800)
        self.assertEqual(result["limits"][1]["label"], "GPT-5.3-Codex-Spark")

    def test_general_wham_allowed_false_excludes_the_account(self):
        accounts = FakeAccounts([account()])
        session = FakeSession(gets=[
            FakeResponse(payload={"models": [{"slug": "gpt-5.6-sol"}]}),
            FakeResponse(payload={"rate_limit": {
                "allowed": False,
                "limit_reached": True,
                "primary_window": {"used_percent": 100, "limit_window_seconds": 604800},
            }}),
        ])
        result = CodexService(accounts, SessionFactory([session])).refresh_account("token-a")
        self.assertEqual(result["state"], "limited")

    def test_auth_failure_has_a_safe_projection_and_preserves_last_observation(self):
        accounts = FakeAccounts([account()])
        session = FakeSession(gets=[FakeResponse(status=401, payload={"detail": "raw bearer token-a"})])
        result = CodexService(accounts, SessionFactory([session])).refresh_account("token-a")
        self.assertEqual(result["state"], "auth_required")
        self.assertEqual(result["models"][0]["id"], "gpt-5.6-codex")
        self.assertEqual(result["error_code"], "models_auth_required")
        self.assertNotIn("token-a", json.dumps(result))

    def test_empty_usage_is_read_failed_instead_of_fabricated_observed(self):
        accounts = FakeAccounts([account()])
        session = FakeSession(gets=[
            FakeResponse(payload={"models": [{"id": "gpt-5.6-codex"}]}),
            FakeResponse(payload={}),
        ])
        result = CodexService(accounts, SessionFactory([session])).refresh_account("token-a")
        self.assertEqual(result["state"], "read_failed")
        self.assertEqual(result["error_code"], "usage_invalid_response")

    def test_forbidden_observation_does_not_claim_token_refresh_is_required(self):
        for failed_kind in ("models", "usage"):
            with self.subTest(kind=failed_kind):
                accounts = FakeAccounts([account()])
                responses = []
                if failed_kind == "usage":
                    responses.append(FakeResponse(payload={"models": [{"slug": "gpt-5.6-codex"}]}))
                responses.append(FakeResponse(status=403, payload={"detail": "private token-a"}))
                result = CodexService(accounts, SessionFactory([FakeSession(gets=responses)])).refresh_account("token-a")
                self.assertEqual(result["state"], "read_failed")
                self.assertEqual(result["error_code"], f"{failed_kind}_access_denied")
                self.assertEqual(result["models"][0]["id"], "gpt-5.6-codex")
                self.assertNotIn("token-a", json.dumps(result))

    def test_empty_rate_limit_and_nonfinite_usage_are_read_failed(self):
        for usage in (
            {"rate_limit": {}},
            {"rate_limit": {"primary_window": {"used_percent": float("nan"), "limit_window_seconds": 300}}},
            {"rate_limit": {"primary_window": {"used_percent": -1, "limit_window_seconds": 300}}},
            {"rate_limit": {"primary_window": {"used_percent": True, "limit_window_seconds": 300}}},
        ):
            with self.subTest(usage=usage):
                accounts = FakeAccounts([account()])
                session = FakeSession(gets=[
                    FakeResponse(payload={"models": [{"id": "gpt-5.6-codex"}]}),
                    FakeResponse(payload=usage),
                ])
                result = CodexService(accounts, SessionFactory([session])).refresh_account("token-a")
                self.assertEqual(result["state"], "read_failed")

    def test_management_models_reports_available_account_count(self):
        accounts = FakeAccounts([
            account("a"),
            account("b", codex_observation=observation(state="limited")),
        ])
        result = CodexService(accounts, SessionFactory([])).management_models()
        self.assertEqual(result["items"], [{
            "id": "gpt-5.6-codex", "label": "GPT-5.6 Codex", "route": "codex",
            "state": "available", "available_accounts": 1, "reasoning_efforts": ["medium"],
        }])

    def test_management_models_does_not_count_disabled_or_stale_accounts_as_available(self):
        accounts = FakeAccounts([
            account("disabled", managed_disabled=True),
            account("stale", codex_observation=observation(minutes_ago=10)),
        ])
        result = CodexService(accounts, SessionFactory([])).management_models()
        self.assertEqual(result["items"][0]["state"], "unknown")
        self.assertIsNone(result["items"][0]["available_accounts"])


class CodexRelayTests(unittest.TestCase):
    identity = {"id": "key-one", "role": "user"}
    headers = {"session-id": "session-one", "x-codex-window-id": "window-one", "authorization": "Bearer caller-secret"}

    def test_real_transport_signature_accepts_native_body_without_network(self):
        # Autospec uses the installed transport signature, unlike FakeSession's
        # **kwargs. Run under uv.lock's curl-cffi version to catch API drift.
        payload = {"model": "gpt-5.6-codex", "stream": True,
                   "input": [{"type": "function_call_output", "call_id": "call-1", "output": "fixture"}],
                   "tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}],
                   "reasoning": {"effort": "medium"}, "store": False}
        for compact in (False, True):
            with self.subTest(compact=compact), \
                    patch("services.codex_service.proxy_settings.build_session_kwargs", return_value={}), \
                    patch("curl_cffi.Curl.perform", side_effect=AssertionError("network prohibited")) as perform, \
                    patch("curl_cffi.requests.Session.request", autospec=True,
                          return_value=FakeResponse(payload={"id": "r-native"})) as request:
                service = CodexService(FakeAccounts([account()]))
                result = service.submit(self.identity, payload, self.headers, compact=compact)
                self.assertEqual(result.status_code, 200)
                request.assert_called_once()
                self.assertEqual(request.call_args.kwargs["url"], CODEX_COMPACT_URL if compact else CODEX_RESPONSES_URL)
                self.assertEqual(json.loads(request.call_args.kwargs["data"]), payload)
                self.assertFalse(request.call_args.kwargs["allow_redirects"])
                perform.assert_not_called()

    def test_native_tool_loop_sse_is_relayed_unchanged_and_binds_response(self):
        chunks = [
            b'data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"shell","arguments":"{}"}}\n\n',
            b'data: {"type":"response.completed","response":{"id":"resp_1","output":[{"type":"reasoning","encrypted_content":"cipher"}]}}\n\n',
        ]
        response = FakeResponse(payload=None, chunks=chunks, content_type="text/event-stream")
        session = FakeSession(post_response=response)
        accounts = FakeAccounts([account()])
        service = CodexService(accounts, SessionFactory([session]))
        payload = {
            "model": "gpt-5.6-codex", "stream": True,
            "input": [{"type": "function_call_output", "call_id": "call_1", "output": "ok"}],
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
            "reasoning": {"effort": "medium", "encrypted_content": "client-cipher"},
        }

        result = service.submit(self.identity, payload, self.headers)
        self.assertEqual(b"".join(result.stream), b"".join(chunks))
        call = session.calls[0]
        self.assertEqual(call[1], CODEX_RESPONSES_URL)
        self.assertEqual(json.loads(call[2]["data"]), payload)
        self.assertEqual(call[2]["headers"]["session-id"], "session-one")
        self.assertNotEqual(call[2]["headers"]["authorization"], "Bearer caller-secret")
        digest = hashlib.sha256(b"resp_1").hexdigest()
        self.assertIn(digest, accounts.accounts["token-a"]["codex_response_ids"])
        self.assertNotIn("codex_unknown_outcome", accounts.accounts["token-a"])

    def test_models_read_does_not_consume_a_session_binding(self):
        accounts = FakeAccounts([account()])
        session = FakeSession(gets=[FakeResponse(payload={"models": [{"slug": "gpt-5.6-codex"}]})])
        result = CodexService(accounts, SessionFactory([session])).list_native_models(self.identity, self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertNotIn("codex_affinities", accounts.accounts["token-a"])

    def test_compact_uses_fixed_endpoint_and_preserves_native_payload(self):
        payload = {"model": "gpt-5.6-codex", "input": [{"type": "compaction", "encrypted_content": "cipher"}]}
        session = FakeSession(post_response=FakeResponse(payload={"id": "cmp_1", "output": payload["input"]}))
        service = CodexService(FakeAccounts([account()]), SessionFactory([session]))
        result = service.submit(self.identity, payload, self.headers, compact=True)
        self.assertEqual(json.loads(result.body)["output"], payload["input"])
        self.assertEqual(session.calls[0][1], CODEX_COMPACT_URL)

    def test_missing_chatgpt_account_id_is_allowed(self):
        session = FakeSession(post_response=FakeResponse(payload={"id": "r"}))
        accounts = FakeAccounts([account(account_id="", access_token="opaque-token")])
        service = CodexService(accounts, SessionFactory([session]))
        result = service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertNotIn("chatgpt-account-id", session.calls[0][2]["headers"])

    def test_post_transport_error_is_not_retried_and_quarantines_unknown_outcome(self):
        session = FakeSession(post_error=TimeoutError("raw token-a upstream detail"))
        accounts = FakeAccounts([account()])
        service = CodexService(accounts, SessionFactory([session]))
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(raised.exception.code, "codex_upstream_outcome_unknown")
        self.assertEqual(len(session.calls), 1)
        states = [
            next(iter(update[1]["codex_affinities"].values()))["state"]
            for update in accounts.updates
            if "codex_affinities" in update[1]
        ]
        self.assertIn("pending", states)
        binding = next(iter(accounts.accounts["token-a"]["codex_affinities"].values()))
        self.assertEqual(binding["state"], "unknown")
        self.assertNotIn("token-a", str(raised.exception))

    def test_client_cancellation_quarantines_account_until_outcome_is_resolved(self):
        response = FakeResponse(chunks=[b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'], content_type="text/event-stream")
        accounts = FakeAccounts([account()])
        service = CodexService(accounts, SessionFactory([FakeSession(post_response=response)]))
        result = service.submit(self.identity, {"model": "gpt-5.6-codex", "input": [], "stream": True}, self.headers)
        iterator = iter(result.stream)
        self.assertTrue(next(iterator))
        iterator.close()
        binding = next(iter(accounts.accounts["token-a"]["codex_affinities"].values()))
        self.assertEqual(binding["state"], "unknown")
        self.assertEqual(service._inflight, set())

    def test_uniterated_stream_close_releases_slot_and_marks_exact_session_unknown(self):
        response = FakeResponse(chunks=[b'data: {"type":"response.completed","response":{"id":"r"}}\n\n'], content_type="text/event-stream")
        accounts = FakeAccounts([account()])
        service = CodexService(accounts, SessionFactory([FakeSession(post_response=response)]))
        result = service.submit(self.identity, {"model": "gpt-5.6-codex", "input": [], "stream": True}, self.headers)
        result.stream.close()
        self.assertTrue(response.closed)
        self.assertEqual(service._inflight, set())
        binding = next(iter(accounts.accounts["token-a"]["codex_affinities"].values()))
        self.assertEqual(binding["state"], "unknown")

    def test_close_after_terminal_chunk_preserves_receipt_and_allows_continuation(self):
        for event in ("response.completed", "response.failed"):
            with self.subTest(event=event):
                chunk = f'data: {{"type":"{event}","response":{{"id":"r-terminal"}}}}\n\n'.encode()
                response = FakeResponse(chunks=[chunk], content_type="text/event-stream")
                first = FakeSession(post_response=response)
                second = FakeSession(post_response=FakeResponse(payload={"id": "r-next"}))
                accounts = FakeAccounts([account()])
                service = CodexService(accounts, SessionFactory([first, second]), max_concurrency=1)
                result = service.submit(self.identity, {"model": "gpt-5.6-codex", "stream": True}, self.headers)
                self.assertEqual(next(result.stream), chunk)
                # Close without advancing the suspended generator to EOF.
                result.stream.close()
                result.stream.close()
                self.assertTrue(response.closed)
                self.assertTrue(first.closed)
                self.assertEqual(service._inflight, set())
                binding = next(iter(accounts.accounts["token-a"]["codex_affinities"].values()))
                self.assertEqual(binding["state"], "bound")
                resumed = service.submit(self.identity, {
                    "model": "gpt-5.6-codex", "previous_response_id": "r-terminal",
                }, self.headers)
                self.assertEqual(resumed.status_code, 200)
                self.assertEqual(len(second.calls), 1)

    def test_http_408_quarantines_exact_session_across_restart_without_replay(self):
        for compact in (False, True):
            with self.subTest(compact=compact):
                first = FakeSession(post_response=FakeResponse(status=408, payload={"detail": "private"}))
                accounts = FakeAccounts([account(), account("token-b")])
                service = CodexService(accounts, SessionFactory([first]), max_concurrency=1)
                payload = {"model": "gpt-5.6-codex", "input": []}
                with self.assertRaises(CodexServiceError) as raised:
                    service.submit(self.identity, payload, self.headers, compact=compact)
                self.assertEqual(raised.exception.status_code, 502)
                self.assertEqual(raised.exception.code, "codex_upstream_outcome_unknown")
                self.assertEqual(len(first.calls), 1)
                self.assertTrue(first.closed)
                self.assertEqual(service._inflight, set())
                self.assertTrue(service._capacity.acquire(blocking=False))
                service._capacity.release()
                binding = next(iter(accounts.accounts["token-a"]["codex_affinities"].values()))
                self.assertEqual(binding["state"], "unknown")
                factory = SessionFactory([])
                restarted = CodexService(accounts, factory)
                for current in (service, restarted):
                    with self.assertRaises(CodexServiceError) as retry:
                        current.submit(self.identity, payload, self.headers, compact=compact)
                    self.assertEqual(retry.exception.code, "codex_session_outcome_unknown")
                self.assertEqual(factory.kwargs, [])
                self.assertNotIn("codex_affinities", accounts.accounts["token-b"])

    def test_two_owners_share_resources_but_cannot_resume_each_others_response(self):
        owner_a = {"id": "key-a", "role": "user", "owner_subject": "workbench:o:a"}
        owner_b = {"id": "key-b", "role": "user", "owner_subject": "workbench:o:b"}
        accounts = FakeAccounts([account(managed_owner=owner_a["owner_subject"])])
        sessions = [
            FakeSession(post_response=FakeResponse(payload={"id": "response-a", "output": ["private-a"]})),
            FakeSession(post_response=FakeResponse(payload={"id": "response-b", "output": ["private-b"]})),
        ]
        factory = SessionFactory(sessions)
        service = CodexService(accounts, factory)
        payload = {"model": "gpt-5.6-codex", "input": []}
        for owner, expected in ((owner_a, "private-a"), (owner_b, "private-b")):
            result = service.submit(owner, payload, self.headers)
            self.assertEqual(json.loads(result.body)["output"], [expected])
        self.assertEqual(len(accounts.accounts["token-a"]["codex_affinities"]), 2)
        self.assertEqual(len(factory.kwargs), 2)
        for owner, foreign_response in ((owner_a, "response-b"), (owner_b, "response-a")):
            with self.assertRaises(CodexServiceError) as raised:
                service.submit(owner, {**payload, "previous_response_id": foreign_response}, self.headers)
            self.assertEqual(raised.exception.code, "codex_response_owner_mismatch")
        self.assertEqual(len(factory.kwargs), 2)

    def test_same_new_session_concurrency_dispatches_only_one_upstream_post(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingSession(FakeSession):
            def post(self, url, **kwargs):
                self.calls.append(("POST", url, kwargs))
                started.set()
                release.wait(2)
                return self.post_response

        first_session = BlockingSession(post_response=FakeResponse(payload={"id": "r"}))
        accounts = FakeAccounts([account()])
        factory = SessionFactory([first_session])
        service = CodexService(accounts, factory)
        outcomes = []

        def call():
            try:
                outcomes.append(service.submit(
                    self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers
                ))
            except CodexServiceError as exc:
                outcomes.append(exc)

        first = threading.Thread(target=call)
        second = threading.Thread(target=call)
        first.start()
        self.assertTrue(started.wait(1))
        second.start()
        second.join(1)
        release.set()
        first.join(1)
        self.assertEqual(len(first_session.calls), 1)
        self.assertEqual(len(factory.kwargs), 1)
        self.assertEqual(sum(isinstance(item, CodexHTTPResponse) for item in outcomes), 1)
        errors = [item for item in outcomes if isinstance(item, CodexServiceError)]
        self.assertEqual(errors[0].code, "codex_session_outcome_unknown")

    def test_unknown_session_survives_restart_but_does_not_disable_other_sessions(self):
        accounts = FakeAccounts([account()])
        first = CodexService(accounts, SessionFactory([FakeSession(post_error=TimeoutError("disconnect"))]))
        with self.assertRaises(CodexServiceError):
            first.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)

        restarted = CodexService(accounts, SessionFactory([FakeSession(post_response=FakeResponse(payload={"id": "r2"}))]))
        with self.assertRaises(CodexServiceError) as unknown:
            restarted.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(unknown.exception.code, "codex_session_outcome_unknown")
        result = restarted.submit(
            self.identity,
            {"model": "gpt-5.6-codex", "input": []},
            {**self.headers, "session-id": "different-session"},
        )
        self.assertEqual(result.status_code, 200)

    def test_bound_session_never_switches_when_original_account_becomes_disabled(self):
        affinity = CodexService._affinity_key(self.identity, self.headers)
        accounts = FakeAccounts([
            account("bound", managed_disabled=True, codex_affinities={affinity: {"state": "bound"}}),
            account("available"),
        ])
        service = CodexService(accounts, SessionFactory([]))
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(raised.exception.code, "codex_bound_account_unavailable")

    def test_previous_response_is_bound_to_caller_identity(self):
        response_id = "resp-private"
        digest = hashlib.sha256(response_id.encode()).hexdigest()
        accounts = FakeAccounts([account(codex_response_ids={
            digest: {"owner": CodexService._identity_digest(self.identity), "bound_at": "now"}
        })])
        service = CodexService(accounts, SessionFactory([]))
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(
                {"id": "other-key", "role": "user"},
                {"model": "gpt-5.6-codex", "input": [], "previous_response_id": response_id},
                self.headers,
            )
        self.assertEqual(raised.exception.code, "codex_response_owner_mismatch")

    def test_affinity_capacity_does_not_evict_an_existing_owner(self):
        bindings = {f"binding-{index}": {"state": "bound"} for index in range(256)}
        accounts = FakeAccounts([account(codex_affinities=bindings)])
        service = CodexService(accounts, SessionFactory([]))
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(raised.exception.code, "codex_binding_capacity")
        self.assertEqual(set(accounts.accounts["token-a"]["codex_affinities"]), set(bindings))

    def test_busy_is_safe_429_and_image_quota_or_image_limit_do_not_exclude_codex(self):
        accounts = FakeAccounts([account(status="限流", quota=0)])
        service = CodexService(accounts, SessionFactory([]))
        service._inflight.add("token-a")
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual((raised.exception.status_code, raised.exception.code), (429, "codex_busy"))

    def test_unknown_and_expired_observations_probe_before_selection(self):
        accounts = FakeAccounts([
            account("unknown", codex_observation={}),
            account("expired", codex_observation=observation(minutes_ago=10)),
        ])
        response = FakeResponse(payload={"id": "r"})
        service = CodexService(accounts, SessionFactory([FakeSession(post_response=response)]))
        probes = []

        def probe(token):
            probes.append(token)
            projected = observation(state="observed" if token == "expired" else "limited")
            accounts.update_account(token, {"codex_observation": projected}, quiet=True)
            return projected

        service.refresh_account = probe
        result = service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(probes, ["unknown", "expired"])

    def test_large_unknown_pool_probes_at_most_three_accounts_per_request(self):
        accounts = FakeAccounts([
            account(f"unknown-{index}", codex_observation={})
            for index in range(20)
        ])
        service = CodexService(accounts, SessionFactory([]))
        probes = []

        def probe(token):
            probes.append(token)
            projected = observation(state="limited")
            accounts.update_account(token, {"codex_observation": projected}, quiet=True)
            return projected

        service.refresh_account = probe
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(raised.exception.code, "codex_busy")
        self.assertEqual(probes, ["unknown-0", "unknown-1", "unknown-2"])

        with self.assertRaises(CodexServiceError):
            service.submit(
                self.identity,
                {"model": "gpt-5.6-codex", "input": []},
                {**self.headers, "session-id": "second-session"},
            )
        self.assertLessEqual(len(probes), 6)
        self.assertTrue(set(probes[3:]).isdisjoint(set(probes[:3])))

    def test_unknown_previous_response_is_rejected_before_post(self):
        service = CodexService(FakeAccounts([account()]), SessionFactory([]))
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {
                "model": "gpt-5.6-codex", "input": [], "previous_response_id": "other-owner-response"
            }, self.headers)
        self.assertEqual(raised.exception.code, "codex_response_owner_unknown")

    def test_upstream_401_is_known_terminal_and_is_never_replayed(self):
        session = FakeSession(post_response=FakeResponse(status=401, payload={"detail": "private"}))
        accounts = FakeAccounts([account()])
        service = CodexService(accounts, SessionFactory([session]))
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(raised.exception.code, "codex_auth_required")
        self.assertEqual(len(session.calls), 1)
        binding = next(iter(accounts.accounts["token-a"]["codex_affinities"].values()))
        self.assertEqual(binding["state"], "bound")
        self.assertEqual(accounts.accounts["token-a"]["codex_observation"]["state"], "auth_required")

    def test_catalog_refresh_does_not_clear_response_auth_rejection(self):
        rejected = FakeSession(post_response=FakeResponse(status=401, payload={}))
        catalog = FakeSession(gets=[
            FakeResponse(payload={"models": [{"slug": "gpt-5.6-codex"}]}),
            FakeResponse(payload={"rate_limit": {"primary_window": {"used_percent": 10}}}),
        ])
        accounts = FakeAccounts([account()])
        service = CodexService(accounts, SessionFactory([rejected, catalog]))
        with self.assertRaises(CodexServiceError):
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        # Reconstruct the service to cover a restart before the UI refresh.
        service = CodexService(accounts, SessionFactory([catalog]))
        result = service.refresh_account("token-a")
        self.assertEqual(result["state"], "auth_required")
        self.assertEqual(result["error_code"], "codex_http_401")
        self.assertEqual(service.management_models()["items"][0]["available_accounts"], 0)
        with self.assertRaises(CodexServiceError):
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(len(rejected.calls), 1)
        self.assertTrue(all(method == "GET" for method, _, _ in catalog.calls))

    def test_new_authorization_can_be_observed_after_rejection(self):
        for changed_field in ("access_token", "account_id"):
            with self.subTest(changed_field=changed_field):
                accounts = FakeAccounts([account()])
                service = CodexService(accounts, SessionFactory([]))
                service._mark_observation_state("token-a", "auth_required", "codex_http_401", accounts.get_account("token-a"), service._credential_digest(accounts.get_account("token-a")))
                old = accounts.accounts.pop("token-a")
                old[changed_field] = "new-authorization"
                token = old["access_token"]
                accounts.accounts[token] = old
                session = FakeSession(gets=[
                    FakeResponse(payload={"models": [{"slug": "gpt-5.6-codex"}]}),
                    FakeResponse(payload={"rate_limit": {"primary_window": {"used_percent": 10}}}),
                ])
                service = CodexService(accounts, SessionFactory([session]))
                result = service.refresh_account(token)
                self.assertEqual(result["state"], "observed")
                self.assertIsNone(result["error_code"])
                self.assertIsNotNone(service._eligible_account(accounts.get_account(token), allow_probe=False))
                public = json.dumps(result)
                self.assertNotIn("credential_digest", public)
                self.assertNotIn("new-authorization", public)

    def test_rejection_during_catalog_refresh_remains_unavailable(self):
        accounts = FakeAccounts([account()])
        session = FakeSession(gets=[
            FakeResponse(payload={"models": [{"slug": "gpt-5.6-codex"}]}),
            FakeResponse(payload={"rate_limit": {"primary_window": {"used_percent": 10}}}),
        ])
        service = CodexService(accounts, SessionFactory([session]))
        original_get = session.get

        def reject_while_reading(*args, **kwargs):
            service._mark_observation_state("token-a", "auth_required", "codex_http_401", accounts.get_account("token-a"), service._credential_digest(accounts.get_account("token-a")))
            return original_get(*args, **kwargs)

        session.get = reject_while_reading
        result = service.refresh_account("token-a")
        self.assertEqual(result["state"], "auth_required")
        self.assertIsNone(service._eligible_account(accounts.get_account("token-a"), allow_probe=False))
        self.assertNotIn("credential_digest", json.dumps(result))

    def test_legacy_401_survives_refresh_failure_then_success(self):
        prior = observation(state="auth_required", minutes_ago=10)
        prior.update(error_code="codex_http_401", failed_at="2026-09-16T20:27:36+00:00")
        accounts = FakeAccounts([account(codex_observation=prior)])
        sessions = [
            FakeSession(gets=[FakeResponse(status=502, payload={})]),
            FakeSession(gets=[
                FakeResponse(payload={"models": [{"slug": "gpt-5.6-codex"}]}),
                FakeResponse(payload={"rate_limit": {"primary_window": {"used_percent": 10}}}),
            ]),
        ]
        service = CodexService(accounts, SessionFactory(sessions))
        for _ in range(2):
            result = service.refresh_account("token-a")
            self.assertEqual(result["state"], "auth_required")
            self.assertEqual(result["failed_at"], prior["failed_at"])
            self.assertEqual(result["error_code"], "codex_http_401")
            self.assertNotIn("codex_auth_rejection", accounts.get_account("token-a"))

    def test_legacy_401_does_not_bind_to_token_issued_after_failure(self):
        prior = observation(state="auth_required", minutes_ago=10)
        prior.update(error_code="codex_http_401", failed_at="2026-09-16T20:27:36+00:00")
        failure_time = int(datetime.fromisoformat(prior["failed_at"]).timestamp())
        for issued_at, expected in ((failure_time + 60, "observed"), (failure_time - 60, "auth_required")):
            with self.subTest(issued_at=issued_at):
                claims = base64.urlsafe_b64encode(json.dumps({"iat": issued_at}).encode()).decode().rstrip("=")
                token = f"header.{claims}.signature"
                accounts = FakeAccounts([account(token, codex_observation=prior)])
                session = FakeSession(gets=[
                    FakeResponse(payload={"models": [{"slug": "gpt-5.6-codex"}]}),
                    FakeResponse(payload={"rate_limit": {"primary_window": {"used_percent": 10}}}),
                ])
                service = CodexService(accounts, SessionFactory([session]))
                self.assertEqual(service.refresh_account(token)["state"], expected)
                self.assertNotIn("codex_auth_rejection", accounts.get_account(token))

    def test_upstream_5xx_is_unknown_and_not_replayed_or_rebound(self):
        session = FakeSession(post_response=FakeResponse(status=503, payload={"detail": "private"}))
        accounts = FakeAccounts([account()])
        service = CodexService(accounts, SessionFactory([session]))
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(raised.exception.code, "codex_upstream_outcome_unknown")
        self.assertEqual(len(session.calls), 1)
        binding = next(iter(accounts.accounts["token-a"]["codex_affinities"].values()))
        self.assertEqual(binding["state"], "unknown")

    def test_upstream_403_is_known_rejection_without_refresh_claim_or_replay(self):
        session = FakeSession(post_response=FakeResponse(status=403, payload={"detail": "private token-a"}))
        accounts = FakeAccounts([account(), account("token-b")])
        service = CodexService(accounts, SessionFactory([session]))
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(raised.exception.code, "codex_access_denied")
        self.assertNotIn("refresh", str(raised.exception).lower())
        self.assertNotIn("token-a", str(raised.exception))
        self.assertEqual(len(session.calls), 1)
        binding = next(iter(accounts.accounts["token-a"]["codex_affinities"].values()))
        self.assertEqual(binding["state"], "bound")
        self.assertNotIn("codex_affinities", accounts.accounts["token-b"])
        observation = accounts.accounts["token-a"]["codex_observation"]
        self.assertEqual(observation["state"], "read_failed")
        self.assertEqual(observation["error_code"], "codex_http_403")
        with self.assertRaises(CodexServiceError) as unavailable:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(unavailable.exception.code, "codex_bound_account_unavailable")
        self.assertEqual(len(session.calls), 1)

    def test_upstream_429_immediately_limits_account_for_new_sessions(self):
        session = FakeSession(post_response=FakeResponse(status=429, payload={"detail": "private"}))
        accounts = FakeAccounts([account()])
        service = CodexService(accounts, SessionFactory([session]))
        with self.assertRaises(CodexServiceError) as raised:
            service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(raised.exception.code, "codex_limited")
        self.assertEqual(accounts.accounts["token-a"]["codex_observation"]["state"], "limited")
        with self.assertRaises(CodexServiceError) as unavailable:
            service.submit(
                self.identity,
                {"model": "gpt-5.6-codex", "input": []},
                {**self.headers, "session-id": "new-session"},
            )
        self.assertEqual(unavailable.exception.code, "codex_busy")

    def test_nonforced_token_refresh_uses_rotated_alias_before_post(self):
        class RotatingAccounts(FakeAccounts):
            def refresh_access_token(self, token, event=""):
                if token == "old-token":
                    item = self.accounts.pop(token)
                    item["access_token"] = "new-token"
                    self.accounts["new-token"] = item
                    return "new-token"
                return token

            def get_account(self, token):
                if token == "old-token":
                    token = "new-token"
                return super().get_account(token)

        accounts = RotatingAccounts([account("old-token")])
        session = FakeSession(post_response=FakeResponse(payload={"id": "r"}))
        service = CodexService(accounts, SessionFactory([session]))
        result = service.submit(self.identity, {"model": "gpt-5.6-codex", "input": []}, self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(session.calls[0][2]["headers"]["authorization"], "Bearer new-token")


if __name__ == "__main__":
    unittest.main()
