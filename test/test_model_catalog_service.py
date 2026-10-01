from __future__ import annotations

import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from threading import Event, Lock, Thread
from unittest import mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from services.account_service import AccountService
from services.model_service import ModelCatalogService
from services.openai_backend_api import OpenAIBackendAPI
from services.storage.json_storage import JSONStorageBackend
from utils.helper import UpstreamHTTPError


def model_list(*model_ids: str) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "chatgpt",
                "permission": [],
                "root": model_id,
                "parent": None,
            }
            for model_id in model_ids
        ],
    }


class FakeBackend:
    def __init__(self, access_token: str, outcomes: dict[str, object], calls: list[str], closed: list[str]) -> None:
        self.access_token = access_token
        self._outcomes = outcomes
        self._calls = calls
        self._closed = closed

    def list_models(self) -> dict:
        self._calls.append(self.access_token)
        outcome = self._outcomes[self.access_token]
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome()
        return outcome

    def close(self) -> None:
        self._closed.append(self.access_token)


class _RateLimitedModelResponse:
    status_code = 429
    text = "rate limited"

    def __init__(self, retry_after: str) -> None:
        self.headers = {"Retry-After": retry_after}

    def json(self) -> dict:
        return {"error": "rate limited"}


def http_rate_limited_model_backend(retry_after: str) -> OpenAIBackendAPI:
    backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
    backend.access_token = ""
    backend.base_url = "https://chatgpt.com"
    backend._bootstrap = lambda: None
    backend._headers = lambda _route: {}
    backend.session = type("Session", (), {
        "get": lambda _self, *_args, **_kwargs: _RateLimitedModelResponse(retry_after),
        "close": lambda _self: None,
    })()
    return backend


class ModelCatalogServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.accounts = AccountService(
            JSONStorageBackend(Path(self.temp_dir.name) / "accounts.json")
        )
        self.accounts.add_account_items(
            [
                {"access_token": "free-bad", "type": "free", "status": "正常"},
                {"access_token": "free-good", "type": "FREE", "status": "正常"},
                {"access_token": "plus", "type": "Plus", "status": "正常"},
                {"access_token": "pro", "type": "pro", "status": "正常"},
                {"access_token": "team-disabled", "type": "Team", "status": "禁用"},
            ]
        )
        self.accounts.refresh_access_token = lambda token, **_kwargs: token
        self.now = 1000.0
        self.calls: list[str] = []
        self.closed: list[str] = []
        self.outcomes: dict[str, object] = {
            "": model_list("anon", "shared"),
            "free-bad": RuntimeError("expired"),
            "free-good": model_list("free-only", "shared"),
            "plus": model_list("plus-only", "shared"),
            "pro": model_list("pro-only"),
        }
        self.catalog = ModelCatalogService(
            self.accounts,
            backend_factory=lambda access_token="": FakeBackend(
                access_token, self.outcomes, self.calls, self.closed
            ),
            cache_ttl_seconds=300,
            clock=lambda: self.now,
            observed_clock=lambda: self.now,
        )

    def test_catalog_unions_anonymous_and_each_active_account_type(self) -> None:
        result = self.catalog.list_models()

        self.assertEqual(
            [item["id"] for item in result["data"]],
            ["anon", "free-only", "plus-only", "pro-only", "shared"],
        )
        self.assertCountEqual(self.calls, ["", "free-bad", "free-good", "plus", "pro"])
        self.assertCountEqual(self.closed, self.calls)
        self.assertNotIn("team-disabled", self.calls)

        pro_route = self.catalog.route_for_model("pro-only")
        self.assertEqual(pro_route.account_types, frozenset({"Pro"}))
        self.assertFalse(pro_route.allow_anonymous)

        shared_route = self.catalog.route_for_model("shared")
        self.assertEqual(shared_route.account_types, frozenset({"free", "Plus"}))
        self.assertTrue(shared_route.allow_anonymous)

    def test_bound_text_catalog_read_does_not_lock_out_other_requests_or_health(self) -> None:
        self.outcomes["plus"] = model_list("gpt-5-6")
        with self.accounts._lock:
            bindings = [self.accounts._conversation_binding_for_token_locked("plus") for _ in range(3)]
        catalog_started = Event()
        release_catalog = Event()
        original_factory = self.catalog._backend_factory

        def factory(access_token=""):
            backend = original_factory(access_token=access_token)
            original_list = backend.list_models

            def read_models():
                catalog_started.set()
                if not release_catalog.wait(2):
                    raise RuntimeError("test catalog was not released")
                return original_list()

            backend.list_models = read_models
            return backend

        self.catalog._backend_factory = factory
        results, errors = [], []
        def read_binding(binding):
            try:
                results.append(self.accounts.get_bound_text_access_token(binding, model="gpt-5-6"))
            except Exception as exc:
                errors.append(type(exc).__name__)

        threads = [Thread(target=read_binding, args=(binding,), daemon=True) for binding in bindings]
        health_done = Event()
        def read_health():
            self.accounts.get_stats()
            health_done.set()

        with mock.patch("services.model_service.model_catalog_service", self.catalog):
            try:
                for thread in threads:
                    thread.start()
                self.assertTrue(catalog_started.wait(1), "bound text deadlocked before reaching model catalog backend")
                health = Thread(target=read_health, daemon=True)
                health.start()
                self.assertTrue(health_done.wait(1), "model lookup must not hold the account lock needed by health")
            finally:
                release_catalog.set()
                for thread in threads:
                    thread.join(1)
            self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(results, ["plus"] * 3)
        self.assertEqual(self.calls.count("plus"), 1)

    def test_bound_text_keeps_model_and_disabled_account_checks(self) -> None:
        with self.accounts._lock:
            binding = self.accounts._conversation_binding_for_token_locked("plus")
        with mock.patch("services.model_service.model_catalog_service", self.catalog):
            with self.assertRaisesRegex(RuntimeError, "cannot serve model"):
                self.accounts.get_bound_text_access_token(binding, model="pro-only")
            self.accounts.update_account("plus", {"status": "禁用"})
            with self.assertRaisesRegex(RuntimeError, "cannot serve text"):
                self.accounts.get_bound_text_access_token(binding, model="auto")

    def test_catalog_is_cached_until_ttl_expires(self) -> None:
        self.catalog.list_models()
        self.catalog.list_models()
        self.catalog.route_for_model("pro-only")

        self.assertEqual(self.calls.count("pro"), 1)
        self.assertEqual(self.calls.count(""), 1)

    def test_cold_discovery_starts_pool_accounts_in_parallel_with_a_bounded_worker_count(self) -> None:
        started = Event()
        release = Event()
        entered: list[str] = []
        entered_lock = Lock()
        original_factory = self.catalog._backend_factory

        def factory(access_token=""):
            backend = original_factory(access_token=access_token)
            original_list = backend.list_models

            def read_models():
                with entered_lock:
                    entered.append(access_token)
                    if len(entered) == 5:  # anonymous plus the four active accounts
                        started.set()
                if not release.wait(2):
                    raise RuntimeError("test catalog was not released")
                return original_list()

            backend.list_models = read_models
            return backend

        self.catalog._backend_factory = factory
        reader = Thread(target=self.catalog.list_models, daemon=True)
        try:
            reader.start()
            self.assertTrue(started.wait(1), "cold discovery queued an active account behind another timeout")
        finally:
            release.set()
            reader.join(2)
        self.assertFalse(reader.is_alive())
        self.assertLessEqual(self.catalog.MAX_DISCOVERY_WORKERS, 8)

    def test_slow_account_isolated_by_budget_and_collected_after_late_completion(self) -> None:
        entered, release, completed = Event(), Event(), Event()
        self.catalog.DISCOVERY_BUDGET_SECONDS = 0.05
        reads = 0

        def slow_models():
            nonlocal reads
            reads += 1
            if reads == 1:
                entered.set()
                release.wait(2)
                completed.set()
            return model_list("plus-only")

        self.outcomes["plus"] = slow_models
        started = time.monotonic()
        try:
            self.catalog.list_models()
            self.assertTrue(entered.is_set())
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(self.catalog.route_for_model("plus-only").account_types, frozenset())

            # Keep one pending read, without probing healthy peers again.
            self.now += 2
            self.catalog.list_models()
            self.assertEqual(reads, 1)
            self.assertLessEqual(self.catalog._expires_at - self.now, 1.0)

            release.set()
            self.assertTrue(completed.wait(1))
            for future, _kind, _fingerprint in self.catalog._inflight_accounts.values():
                future.result(timeout=1)
            self.now += 2
            self.catalog.list_models()
            self.assertEqual(reads, 1)
            self.assertEqual(self.calls.count("pro"), 1)
            self.assertEqual(self.catalog.route_for_model("plus-only").account_types, frozenset({"Plus"}))
        finally:
            release.set()

    def _start_late_catalog_read(self, token: str):
        release = Event()
        self.addCleanup(release.set)
        self.catalog.DISCOVERY_BUDGET_SECONDS = 0.01

        def slow_models():
            if not release.wait(2):
                raise RuntimeError("test did not release catalog")
            return model_list("late-model")

        self.outcomes[token] = slow_models
        self.catalog.list_models()
        identity = next((self.accounts._stable_account_identity(a) for a in self.accounts.list_accounts()
                         if a['access_token'] == token), None)
        future = self.catalog._inflight_accounts[identity][0] if identity else self.catalog._anonymous_inflight
        return release, future, identity

    def test_late_anonymous_read_is_collected_without_new_probe(self) -> None:
        release, future, _identity = self._start_late_catalog_read("")
        release.set()
        future.result(timeout=1)
        self.now += 2
        self.assertIn("late-model", {m['id'] for m in self.catalog.list_models()['data']})
        self.assertEqual(self.calls.count(""), 1)
        self.assertTrue(self.catalog.route_for_model("late-model").allow_anonymous)

    def test_late_read_from_replaced_credential_is_not_published(self) -> None:
        release, future, identity = self._start_late_catalog_read("plus")
        self.outcomes["plus-rotated"] = model_list("rotated-model")
        self.accounts._apply_refreshed_tokens("plus", {"access_token": "plus-rotated"}, "test")
        release.set()
        future.result(timeout=1)
        self.now += 2
        self.catalog.list_models()
        self.assertNotIn("late-model", self.catalog._model_observations[identity]['models'])
        self.assertIn("rotated-model", self.catalog._models_by_account[identity])
        self.assertEqual(self.calls.count("plus"), 1)
        self.assertEqual(self.calls.count("plus-rotated"), 1)

    def test_late_disabled_account_read_is_not_published(self) -> None:
        release, future, identity = self._start_late_catalog_read("plus")
        self.accounts.update_account("plus", {"managed_disabled": True})
        release.set()
        future.result(timeout=1)
        self.now += 2
        self.catalog.list_models()
        self.assertNotIn(identity, self.catalog._models_by_account)
        self.assertNotIn("late-model", self.catalog._model_observations[identity]['models'])

    def test_late_read_uses_completion_time_without_extending_expired_evidence(self) -> None:
        release, future, identity = self._start_late_catalog_read("plus")
        release.set()
        future.result(timeout=1)
        completed_at = self.now
        self.now += self.catalog._cache_ttl_seconds + 1
        self.catalog.list_models()
        observation = self.catalog._model_observations[identity]
        self.assertEqual(observation['observed_at'], completed_at)
        self.assertEqual(observation['refresh_after'], completed_at + self.catalog._cache_ttl_seconds)
        self.assertEqual(observation['observation_state'], 'stale')
        self.assertNotIn(identity, self.catalog._models_by_account)
        self.assertEqual(self.calls.count("plus"), 1)
        self.catalog.list_models()
        self.assertEqual(self.calls.count("plus"), 2)
        self.assertIn(identity, self.catalog._models_by_account)

    def test_account_changes_during_discovery_wait_cannot_publish_old_read(self) -> None:
        for mutation in ('disable', 'rotate'):
            with self.subTest(mutation=mutation):
                self.setUp()
                entered, release = Event(), Event()
                self.catalog.DISCOVERY_BUDGET_SECONDS = 1

                def slow_models():
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError('test did not release read')
                    return model_list('late-model')

                self.outcomes['plus'] = slow_models
                reader = Thread(target=self.catalog.list_models)
                try:
                    reader.start()
                    self.assertTrue(entered.wait(1))
                    identity = self.accounts._stable_account_identity(self.accounts.get_account('plus'))
                    if mutation == 'disable':
                        self.accounts.update_account('plus', {'managed_disabled': True})
                    else:
                        self.outcomes['plus-rotated'] = model_list('rotated-model')
                        self.accounts._apply_refreshed_tokens('plus', {'access_token': 'plus-rotated'}, 'test')
                    release.set()
                    reader.join(2)
                    self.assertFalse(reader.is_alive())
                    self.assertNotIn(identity, self.catalog._models_by_account)
                    self.assertNotIn('late-model', self.catalog._model_observations.get(identity, {}).get('models', {}))
                finally:
                    release.set()
                    reader.join(2)

    def test_expired_anonymous_capability_is_retained_but_not_routed(self) -> None:
        self.catalog.list_models()
        self.now += self.catalog._cache_ttl_seconds + 1
        self.outcomes[''] = RuntimeError('anonymous read unavailable')
        self.assertIn('anon', {row['id'] for row in self.catalog.list_models()['data']})
        self.assertFalse(self.catalog.route_for_model('anon').allow_anonymous)

    def test_failed_late_read_is_consumed_before_a_bounded_retry(self) -> None:
        release = Event()
        self.catalog.DISCOVERY_BUDGET_SECONDS = .01

        def late_failure():
            release.wait(2)
            raise RuntimeError('controlled late failure')

        self.outcomes['plus'] = late_failure
        try:
            self.catalog.list_models()
            identity = self.accounts._stable_account_identity(self.accounts.get_account('plus'))
            future = self.catalog._inflight_accounts[identity][0]
            release.set()
            with self.assertRaisesRegex(RuntimeError, 'controlled late failure'):
                future.result(timeout=1)
            self.now += 2
            self.catalog.list_models()
            self.assertEqual(self.calls.count('plus'), 1)
            self.assertNotIn(identity, self.catalog._models_by_account)
            self.outcomes['plus'] = model_list('plus-only')
            self.now += 2
            self.catalog.list_models()
            self.assertEqual(self.calls.count('plus'), 2)
            self.assertIn(identity, self.catalog._models_by_account)
        finally:
            release.set()

    def test_failed_account_retries_without_reprobing_fresh_peer(self) -> None:
        self.outcomes["pro"] = RuntimeError("temporary upstream failure")
        self.catalog.list_models()
        self.assertEqual(self.calls.count("plus"), 1)
        self.assertEqual(self.calls.count("pro"), 1)

        # The failed Pro row is due again, but the healthy Plus row retains
        # its own execution TTL and must not be dragged into B's retry loop.
        for seconds, expected_pro_calls in ((2, 2), (4, 3), (6, 4)):
            self.now += seconds
            self.catalog.list_models()
            self.catalog.route_for_model("plus-only")
            self.assertEqual(self.calls.count("plus"), 1)
            self.assertEqual(self.calls.count("pro"), expected_pro_calls)
        self.assertEqual(self.catalog.route_for_model("plus-only").account_types, frozenset({"Plus"}))

        self.outcomes["pro"] = model_list("pro-only")
        self.now += 2
        self.catalog.list_models()
        self.assertEqual(self.catalog.route_for_model("pro-only").account_types, frozenset({"Pro"}))

    def test_anonymous_failure_does_not_shorten_healthy_account_ttl(self) -> None:
        self.outcomes[""] = RuntimeError("anonymous unavailable")
        self.catalog.list_models()
        self.assertEqual(self.calls.count("plus"), 1)
        self.assertEqual(self.calls.count(""), 1)

        for seconds, expected_anonymous_calls in ((2, 2), (4, 3), (6, 4)):
            self.now += seconds
            self.catalog.list_models()
            self.catalog.route_for_model("plus-only")
            self.assertEqual(self.calls.count("plus"), 1)
            self.assertEqual(self.calls.count(""), expected_anonymous_calls)
        self.assertEqual(self.catalog.route_for_model("plus-only").account_types, frozenset({"Plus"}))

    def test_anonymous_429_keeps_retry_after_without_shortening_account_ttl(self) -> None:
        self.outcomes[""] = UpstreamHTTPError("anon models", 429, {}, retry_after=47)

        self.catalog.list_models()
        self.assertEqual(self.catalog._anonymous_observation["retry_after"], self.now + 47)
        self.assertEqual(self.calls.count("plus"), 1)

        self.now += 46
        self.catalog.list_models()
        self.assertEqual(self.calls.count(""), 1)
        self.assertEqual(self.calls.count("plus"), 1)

        self.now += 1
        self.catalog.list_models()
        self.assertEqual(self.calls.count(""), 2)
        self.assertEqual(self.calls.count("plus"), 1)

    def test_actual_anonymous_model_http_date_429_reaches_catalog_retry_window(self) -> None:
        retry_after = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=50), usegmt=True)
        original_factory = self.catalog._backend_factory

        def factory(access_token=""):
            if not access_token:
                return http_rate_limited_model_backend(retry_after)
            return original_factory(access_token=access_token)

        self.catalog._backend_factory = factory
        self.catalog.list_models()

        self.assertGreaterEqual(self.catalog._anonymous_observation["retry_after"] - self.now, 45)
        self.now += 2
        self.catalog.list_models()
        self.assertEqual(self.calls.count("plus"), 1)

    def test_actual_model_read_preserves_numeric_and_http_date_retry_after(self) -> None:
        with self.assertRaises(UpstreamHTTPError) as numeric:
            http_rate_limited_model_backend("47").list_models()
        self.assertEqual(numeric.exception.retry_after, 47)

        retry_after = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=50), usegmt=True)
        with self.assertRaises(UpstreamHTTPError) as http_date:
            http_rate_limited_model_backend(retry_after).list_models()
        self.assertGreaterEqual(http_date.exception.retry_after, 45)

    def test_refresh_completion_expires_peer_observation_before_routing(self) -> None:
        self.catalog.list_models()
        plus_identity = next(
            self.accounts._stable_account_identity(account)
            for account in self.accounts.list_accounts()
            if account["access_token"] == "plus"
        )
        pro_identity = next(
            self.accounts._stable_account_identity(account)
            for account in self.accounts.list_accounts()
            if account["access_token"] == "pro"
        )
        self.catalog._model_observations[plus_identity]["refresh_after"] = self.now + 1
        self.catalog._model_observations[pro_identity]["refresh_after"] = self.now
        self.catalog._expires_at = self.now

        def delayed_pro_result() -> dict:
            self.now += 2
            return model_list("pro-only")

        self.outcomes["pro"] = delayed_pro_result
        self.catalog.list_models()

        self.assertNotIn(plus_identity, self.catalog._models_by_account)
        self.assertEqual(self.calls.count("plus"), 1)
        self.catalog.list_models()
        self.assertEqual(self.calls.count("plus"), 2)

    def test_successful_account_rechecks_on_expiry_state_and_credential_change(self) -> None:
        self.catalog.list_models()
        self.assertEqual(self.calls.count("plus"), 1)

        self.now += self.catalog._cache_ttl_seconds + 1
        self.catalog.list_models()
        self.assertEqual(self.calls.count("plus"), 2)

        self.accounts.update_account("plus", {"managed_disabled": True})
        self.catalog.list_models()
        self.assertEqual(self.catalog.route_for_model("plus-only").account_types, frozenset())
        self.accounts.update_account("plus", {"managed_disabled": False, "status": "正常"})
        self.catalog.list_models()
        self.assertEqual(self.calls.count("plus"), 3)

        self.outcomes["plus-rotated"] = model_list("plus-only")
        self.accounts._apply_refreshed_tokens("plus", {"access_token": "plus-rotated"}, "test")
        self.catalog.list_models()
        self.assertEqual(self.calls.count("plus-rotated"), 1)
        self.assertEqual(self.catalog.route_for_model("plus-only").account_types, frozenset({"Plus"}))

    def test_concurrent_readers_share_one_catalog_refresh(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _index: self.catalog.list_models(), range(8)))

        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(self.calls.count(""), 1)
        self.assertEqual(self.calls.count("free-good"), 1)
        self.assertEqual(self.calls.count("plus"), 1)
        self.assertEqual(self.calls.count("pro"), 1)

    def test_failed_refresh_retains_stale_capability_but_drops_execution_route(self) -> None:
        self.catalog.list_models()
        self.outcomes["pro"] = RuntimeError("temporary upstream failure")
        self.now += 301

        result = self.catalog.list_models()

        self.assertIn("pro-only", {item["id"] for item in result["data"]})
        self.assertEqual(
            self.catalog.route_for_model("pro-only").account_types,
            frozenset(),
        )
        identity = next(self.accounts._stable_account_identity(account) for account in self.accounts.list_accounts()
                        if account["access_token"] == "pro")
        observation = self.catalog._model_observations[identity]
        self.assertIn(observation["observation_state"], {"read_failed", "stale"})
        self.assertIn(observation["reason"], {"read_failed", "stale"})
        # Old success makes pro-only queueable, but cannot prove that a new
        # request is permanently unsupported while this paid read failed.
        self.assertTrue(self.catalog.catalog_is_unknown())

        self.outcomes["pro"] = model_list("pro-only")
        self.now += 301
        self.catalog.list_models()
        self.assertEqual(self.catalog.route_for_model("pro-only").account_types, frozenset({"Pro"}))
        self.assertEqual(self.calls.count("pro"), 3)

    def test_limited_account_keeps_capability_evidence_but_never_routes(self) -> None:
        self.catalog.list_models()
        self.accounts.update_account("pro", {"status": "限流"})

        self.assertIn("pro-only", {item["id"] for item in self.catalog.list_models()["data"]})
        self.assertEqual(self.catalog.route_for_model("pro-only").account_types, frozenset())
        row = self.catalog.public_accounts_for_model("pro-only", ["text"])[0]
        self.assertEqual(row["reason"], "limited")
        self.assertEqual(row["observation_state"], "observed")
        # A rate-limited account retains its known model but cannot prove that
        # a newly requested model is permanently unsupported.
        self.assertTrue(self.catalog.catalog_is_unknown())

    def test_successful_catalog_removal_is_permanent_not_stale(self) -> None:
        self.catalog.list_models()
        self.outcomes["pro"] = model_list("replacement")
        self.now += 301

        self.catalog.list_models()

        self.assertNotIn("pro-only", {item["id"] for item in self.catalog.list_models()["data"]})
        self.assertEqual(self.catalog.known_account_types_for_model("pro-only"), frozenset())

    def test_initial_paid_catalog_failure_is_unknown_not_unsupported(self) -> None:
        self.outcomes.update({
            "": RuntimeError("anonymous unavailable"),
            "free-bad": RuntimeError("unavailable"),
            "free-good": RuntimeError("unavailable"),
            "plus": RuntimeError("unavailable"),
            "pro": RuntimeError("unavailable"),
        })

        self.catalog.list_models()

        self.assertTrue(self.catalog.catalog_is_unknown())

    def test_removed_account_type_drops_its_stale_capabilities(self) -> None:
        self.catalog.list_models()
        self.accounts.delete_accounts(["pro"])

        result = self.catalog.list_models()

        self.assertNotIn("pro-only", {item["id"] for item in result["data"]})
        self.assertEqual(
            self.catalog.route_for_model("pro-only").account_types,
            frozenset(),
        )

    def test_same_plan_accounts_keep_disjoint_model_observations(self) -> None:
        self.accounts.add_account_items([
            {"access_token": "plus-second", "type": "Plus", "status": "正常"},
        ])
        self.outcomes["plus-second"] = model_list("second-plus-only")

        self.catalog.list_models()

        identities = {
            str(account["access_token"]): self.accounts._stable_account_identity(account)
            for account in self.accounts.list_accounts()
        }
        plus_only = self.catalog.route_for_model("plus-only")
        second_only = self.catalog.route_for_model("second-plus-only")
        self.assertEqual(plus_only.account_types, frozenset({"Plus"}))
        self.assertEqual(plus_only.account_identities, frozenset({identities["plus"]}))
        self.assertEqual(second_only.account_identities, frozenset({identities["plus-second"]}))

    def test_bound_same_plan_account_fails_closed_when_only_a_peer_observed_the_model(self) -> None:
        self.accounts.add_account_items([
            {"access_token": "plus-second", "type": "Plus", "status": "正常"},
        ])
        self.outcomes["plus-second"] = model_list("second-plus-only")
        with self.accounts._lock:
            binding = self.accounts._conversation_binding_for_token_locked("plus")

        self.catalog.list_models()

        with mock.patch("services.model_service.model_catalog_service", self.catalog):
            self.assertEqual(
                self.accounts.get_bound_text_access_token(binding, model="plus-only"),
                "plus",
            )
            with self.assertRaisesRegex(RuntimeError, "bound account cannot serve model"):
                self.accounts.get_bound_text_access_token(binding, model="second-plus-only")

    def test_bound_text_rechecks_managed_disable_and_rate_limit(self) -> None:
        with self.accounts._lock:
            binding = self.accounts._conversation_binding_for_token_locked("plus")
        self.catalog.list_models()

        with mock.patch("services.model_service.model_catalog_service", self.catalog):
            self.accounts.update_account("plus", {"managed_disabled": True})
            with self.assertRaisesRegex(RuntimeError, "bound account cannot serve text"):
                self.accounts.get_bound_text_access_token(binding, model="plus-only")
            self.accounts.update_account("plus", {"managed_disabled": False, "status": "限流"})
            self.assertEqual(self.accounts.get_bound_text_access_token(binding, model="auto"), "plus")
            with self.assertRaisesRegex(RuntimeError, "bound account cannot serve text"):
                self.accounts.get_bound_text_access_token(binding, model="plus-only", for_message=True)


if __name__ == "__main__":
    unittest.main()
