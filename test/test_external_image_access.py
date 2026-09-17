import tempfile
import io
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import ai, image_tasks, owned_accounts
from api.external_images import external_image_boundary
from services.account_service import AccountService
from services.auth_service import AuthService
from services.image_task_service import ImageTaskService
from services.owned_accounts import observed_capacity
from services.storage.json_storage import JSONStorageBackend


class ExternalImageAccessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        storage = JSONStorageBackend(Path(self.tmp.name) / "accounts.json")
        self.auth = AuthService(storage)
        self.account = AccountService(storage)
        self.key_a, self.secret_a = self.auth.create_key(role="user", name="a", owner_subject="workbench:org:a")
        self.key_b, self.secret_b = self.auth.create_key(role="user", name="b", owner_subject="workbench:org:b")
        self.calls = []
        def handler(payload):
            self.calls.append(payload)
            return {"data": [{"url": "http://content-account-pool/images/2026/output.png"}], "_provider_binding_id": payload.get("provider_binding_id"), "_provider_account_identity": payload.get("provider_account_identity"), "_conversation_id": "conversation", "_parent_message_id": "parent"}
        self.tasks = ImageTaskService(Path(self.tmp.name) / "tasks.json", generation_handler=handler)
        for target, value in [
            ("services.account_service.account_service.create_conversation_binding", lambda **_: ("binding", "account", "internal-token")),
            ("services.account_service.account_service.release_image_slot", lambda *_: None),
            ("services.account_service.account_service.get_account", lambda *_: {"quota": 2}),
            ("api.support.auth_service", self.auth),
            ("api.image_tasks.image_task_service", self.tasks),
            ("services.image_task_service.image_task_service", self.tasks),
            ("api.owned_accounts.account_service", self.account),
            ("services.codex_service.codex_service.refresh_account", lambda *_: {"state": "unknown"}),
            ("api.owned_accounts.auth_service", self.auth),
        ]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = FastAPI()
        self.app.middleware("http")(external_image_boundary)
        self.app.include_router(image_tasks.create_router())
        self.app.include_router(owned_accounts.create_router())
        self.app.include_router(ai.create_router())
        self.client = TestClient(self.app)

    def headers(self, secret=None):
        return {"Authorization": "Bearer " + (secret or self.secret_a), "X-Workbench-Image-Client": "1", "X-Forwarded-Prefix": "/ai"}

    def test_original_receipt_isolation_conflict_download_and_revocation(self):
        body = {"client_task_id": "original", "prompt": "sample", "model": "gpt-image-2"}
        self.assertEqual(self.client.post("/api/image-tasks/generations", headers=self.headers(), json=body).status_code, 200)
        for _ in range(100):
            result = self.client.get("/api/image-tasks?ids=original", headers=self.headers()).json()
            if result["items"][0]["status"] == "success":
                break
            time.sleep(.01)
        self.assertEqual(result["items"][0]["data"], [{"url": "/ai/api/image-tasks/original/images/0"}])
        self.assertEqual(self.client.post("/api/image-tasks/generations", headers=self.headers(), json=body).status_code, 200)
        self.assertEqual(self.client.post("/api/image-tasks/generations", headers=self.headers(), json={**body, "prompt": "different"}).status_code, 409)
        self.assertEqual(len(self.calls), 1)
        other = self.client.get("/api/image-tasks?ids=original", headers=self.headers(self.secret_b)).json()
        self.assertEqual(other, {"items": [], "missing_ids": ["original"]})
        with patch("api.external_images.image_storage_service.get_bytes", return_value=b"png") as read:
            response = self.client.get("/api/image-tasks/original/images/0", headers=self.headers())
            self.assertEqual(response.content, b"png")
            self.assertEqual(response.headers["cache-control"], "private, no-store")
            read.assert_called_once_with("2026/output.png")
            self.assertEqual(self.client.get("/api/image-tasks/original/images/0", headers=self.headers(self.secret_b)).status_code, 404)
        self.assertFalse(self.auth.revoke_owned_key("workbench:org:b", str(self.key_a["id"])))
        self.assertTrue(self.auth.revoke_owned_key("workbench:org:a", str(self.key_a["id"])))
        self.assertEqual(self.client.get("/api/image-tasks?ids=original", headers=self.headers()).status_code, 401)

    def test_unknown_keeps_original_binding_and_receipt_after_restart(self):
        from services.protocol.conversation import ImageGenerationError
        def uncertain(payload):
            self.calls.append(payload)
            payload["progress_callback"].record_conversation_id("original-chat")
            raise ImageGenerationError("timed out", code="CONVERSATION_OUTCOME_UNKNOWN",
                                       conversation_id="original-chat")
        self.tasks.generation_handler = uncertain
        body = {"client_task_id": "unknown", "prompt": "sample", "model": "gpt-image-2"}
        self.client.post("/api/image-tasks/generations", headers=self.headers(), json=body)
        for _ in range(100):
            raw = self.tasks.list_tasks(self.key_a, ["unknown"])["items"][0]
            if raw["status"] == "error":
                break
            time.sleep(.01)
        self.assertEqual(raw["provider_binding_id"], "binding")
        self.assertTrue(raw["upstream_unfinished"])
        self.assertEqual(raw["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
        restarted = ImageTaskService(self.tasks.path, generation_handler=uncertain)
        self.assertEqual(restarted.list_tasks(self.key_a, ["unknown"])["items"][0]["image_session_id"], "original-chat")
        self.client.post("/api/image-tasks/generations", headers=self.headers(), json=body)
        self.assertEqual(len(self.calls), 1)
        record = next(iter(restarted._tasks.values()))
        self.assertTrue(record["request_message_id"])
        record.update(status="success", updated_at="2000-01-01 00:00:00", upstream_unfinished=False, error_code="")
        self.assertFalse(restarted._cleanup_locked())
        self.assertEqual(len(restarted._tasks), 1)

    def test_recovered_base64_result_download(self):
        from api.external_images import task_image_bytes
        self.assertEqual(task_image_bytes({"status": "success", "data": [{"b64_json": "cG5n"}]}, 0), b"png")
        with patch("api.external_images.image_storage_service.settings", return_value={"public_base_url": "https://cdn.example.test/assets"}), patch("api.external_images.image_storage_service.get_bytes", return_value=b"png") as read:
            self.assertEqual(task_image_bytes({"status": "success", "data": [{"url": "https://cdn.example.test/assets/day/file.png"}]}, 0), b"png")
            read.assert_called_once_with("day/file.png")

    def test_external_task_projection_hides_adoption_failure_detail(self):
        from api.external_images import client_task
        from starlette.requests import Request

        request = Request({
            "type": "http", "method": "GET", "path": "/api/image-tasks",
            "headers": [(b"x-workbench-image-client", b"1")],
            "query_string": b"", "server": ("testserver", 80), "scheme": "http",
        })
        projected = client_task({
            "id": "task", "status": "success",
            "adopted_from_error_code": "content_policy_violation",
            "adopted_from_error": "unbounded upstream detail",
        }, request)
        self.assertEqual(projected["adopted_from_error_code"], "content_policy_violation")
        self.assertNotIn("adopted_from_error", projected)
        self.assertNotIn("unbounded", str(projected))

    def test_sync_requires_durable_id_and_reuses_original(self):
        body = {"prompt": "sample", "model": "gpt-image-2"}
        self.assertEqual(self.client.post("/v1/images/generations", headers=self.headers(), json=body).status_code, 400)
        self.assertEqual(self.calls, [])
        with patch("api.external_images.image_storage_service.get_bytes", return_value=b"png"):
            body["client_task_id"] = "sync-original"
            first = self.client.post("/v1/images/generations", headers=self.headers(), json=body)
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(first.json()["data"], [{"b64_json": "cG5n"}])
            self.assertEqual(self.client.post("/v1/images/generations", headers=self.headers(), json=body).status_code, 200)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.client.post("/v1/images/generations", headers=self.headers(), json={**body, "prompt": "drift"}).status_code, 409)
        for overrides in ({"n": 2}, {"model": "codex"}, {"response_format": "url"}):
            self.assertEqual(self.client.post("/v1/images/generations", headers=self.headers(), json={**body, **overrides}).status_code, 400)

    def test_multi_reference_edit_and_model_catalog_preserve_internal_consumers(self):
        from PIL import Image
        self.tasks.edit_handler = self.tasks.generation_handler
        stream = io.BytesIO()
        Image.new("RGB", (2, 2), "blue").save(stream, format="PNG")
        content = stream.getvalue()
        with patch("api.external_images.image_storage_service.get_bytes", return_value=content):
            response = self.client.post("/v1/images/edits", headers=self.headers(), data={"client_task_id": "edit-two", "prompt": "combine"}, files=[("image", ("front.png", content, "image/png")), ("image", ("detail.png", content, "image/png"))])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.calls[0]["images"]), 2)
        catalogue = {"object": "list", "data": [{"id": "gpt-image-2"}, {"id": "text-model"}]}
        with patch.object(ai.openai_v1_models, "list_models", return_value=catalogue):
            external = self.client.get("/v1/models", headers=self.headers()).json()
            internal = self.client.get("/v1/models", headers={"Authorization": "Bearer " + self.secret_a}).json()
        self.assertEqual([item["id"] for item in external["data"]], ["gpt-image-2"])
        self.assertEqual(internal, catalogue)

    def test_sync_uncertain_response_keeps_caller_id(self):
        from services.protocol.conversation import ImageGenerationError
        def uncertain(payload):
            self.calls.append(payload)
            raise ImageGenerationError("timed out", code="CONVERSATION_OUTCOME_UNKNOWN", conversation_id="original-chat")
        self.tasks.generation_handler = uncertain
        body = {"prompt": "sample", "client_task_id": "sync-unknown"}
        for _ in range(2):
            response = self.client.post("/v1/images/generations", headers=self.headers(), json=body)
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.json()["task_id"], body["client_task_id"])
        self.assertEqual(len(self.calls), 1)

    def test_known_pre_send_failure_does_not_become_unknown(self):
        from services.protocol.conversation import ConversationRequest, _generate_bound_single_image
        def unavailable(payload):
            request = ConversationRequest(model="gpt-image-2", prompt=payload["prompt"],
                                          provider_binding_id=payload["provider_binding_id"],
                                          provider_account_identity=payload["provider_account_identity"],
                                          client_conversation_id=payload["client_conversation_id"])
            return _generate_bound_single_image(request, 1, 1)
        self.tasks.generation_handler = unavailable
        with patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="account"), patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", side_effect=RuntimeError("account was disabled")), patch("services.protocol.conversation.OpenAIBackendAPI") as backend:
            response = self.client.post("/v1/images/generations", headers=self.headers(), json={"client_task_id": "pre-send", "prompt": "sample"})
            self.assertEqual(response.status_code, 502)
            task = self.tasks.list_tasks(self.key_a, ["pre-send"])["items"][0]
            self.assertEqual(task["error_code"], "CONVERSATION_BINDING_UNAVAILABLE")
            self.assertFalse(task.get("upstream_unfinished", False))
            backend.assert_not_called()

    def test_public_ingress_cannot_read_raw_files_or_management(self):
        for path in ("/images/2026/output.png", "/files/reference.png", "/api/accounts", "/api/settings", "/api/workbench/ai/accounts"):
            self.assertEqual(self.client.get(path, headers=self.headers()).status_code, 404)
        self.assertEqual(self.client.get("/api/workbench/ai/accounts", headers={"Authorization": "Bearer " + self.secret_a, "X-Workbench-Account-Owner": "workbench:org:a"}).status_code, 403)
        self.assertEqual(self.client.get("/api/image-tasks", headers={"X-Workbench-Image-Client": "1"}).status_code, 401)

    def test_management_creates_only_scoped_user_keys(self):
        _, admin = self.auth.create_key(role="admin", name="management")
        self.assertEqual(self.client.get("/v1/models", headers=self.headers(admin)).status_code, 403)
        trusted = {"Authorization": "Bearer " + admin, "X-Workbench-Account-Owner": "workbench:org:a"}
        created = self.client.post("/api/workbench/ai/keys", headers=trusted, json={"name": "program"})
        self.assertEqual(created.status_code, 200)
        result = created.json()
        self.assertEqual(result["item"]["role"], "user")
        metadata = self.client.get("/api/workbench/ai/keys", headers=trusted).json()
        self.assertNotIn(result["key"], str(metadata))
        other = {**trusted, "X-Workbench-Account-Owner": "workbench:org:b"}
        self.assertEqual(self.client.delete("/api/workbench/ai/keys/" + result["item"]["id"], headers=other).status_code, 404)
        self.assertEqual(self.client.get("/api/workbench/ai/keys", headers={**trusted, "Authorization": "Bearer " + result["key"]}).status_code, 403)
        self.assertEqual(self.client.delete("/api/workbench/ai/keys/" + result["item"]["id"], headers=trusted).status_code, 200)
        self.assertEqual(self.client.get("/api/image-tasks", headers=self.headers(result["key"])).status_code, 401)

    def test_pool_read_requires_admin_and_never_exposes_account_secrets(self):
        self.account.add_account_items([{"access_token": "private-legacy", "refresh_token": "private-refresh"}])
        route = "/api/workbench/ai/pool/accounts"
        self.assertEqual(self.client.get(route).status_code, 401)
        self.assertEqual(self.client.get(route, headers={"Authorization": "Bearer " + self.secret_a, "X-Workbench-Account-Owner": "workbench:org:a"}).status_code, 403)
        _, admin = self.auth.create_key(role="admin", name="pool-view")
        trusted = {"Authorization": "Bearer " + admin, "X-Workbench-Account-Owner": "workbench:org:a"}
        response = self.client.get(route, headers=trusted)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["items"]), 1)
        self.assertNotIn("private", response.text)
        self.assertEqual(self.client.get(route, headers=self.headers(admin)).status_code, 404)

    def test_foreign_session_cannot_be_submitted(self):
        response = self.client.post("/api/image-tasks/generations", headers=self.headers(), json={"client_task_id": "x", "prompt": "sample", "conversation_id": "foreign"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.calls, [])

    def test_owned_account_projection_rotation_disable_and_no_takeover(self):
        owner = "workbench:org:a"
        account = self.account.import_owned_account(owner, {"access_token": "private-upstream", "refresh_token": "private-refresh", "quota": 999})
        self.assertIsNone(account["capacity"]["remaining"])
        self.assertEqual(account["capacity"]["state"], "unknown")
        self.assertNotIn("private", str(account))
        self.assertEqual(self.account.list_owned_accounts("workbench:org:b"), [])
        with self.assertRaises(ValueError):
            self.account.import_owned_account("workbench:org:b", {"access_token": "private-upstream"})
        self.account._apply_refreshed_tokens("private-upstream", {"access_token": "rotated-private"}, "test")
        self.assertEqual(self.account.list_owned_accounts(owner)[0]["id"], account["id"])
        disabled = self.account.set_owned_account_enabled(owner, account["id"], False)
        self.assertFalse(disabled["enabled"])
        self.account.update_account("rotated-private", {"status": "正常", "quota": 10}, quiet=True)
        self.assertEqual(self.account.get_account("rotated-private")["status"], "禁用")
        with patch.object(self.account, "fetch_remote_info", side_effect=RuntimeError("secret upstream error")):
            failed = self.account.refresh_owned_account(owner, account["id"])
        self.assertEqual(failed["capacity"]["state"], "read_failed")
        self.assertNotIn("secret", str(failed))
        restarted = AccountService(self.account.storage)
        self.assertEqual(restarted.list_owned_accounts(owner)[0]["id"], account["id"])

    def test_admission_cannot_use_the_same_last_capacity_twice(self):
        self.account.add_account_items([
            {"access_token": "last-slot", "type": "Pro", "status": "正常", "quota": 1},
            {"access_token": "other-slot", "type": "Pro", "status": "正常", "quota": 2},
        ])
        from services.config import config
        with patch.object(type(config), "image_account_concurrency", new_callable=lambda: property(lambda _: 4)):
            with self.account._lock:
                self.account._image_inflight["last-slot"] = 1
                candidates = self.account._list_available_candidate_tokens()
            self.assertNotIn("last-slot", candidates)
            self.assertIn("other-slot", candidates)
        self.account.mark_image_result("last-slot", True)
        self.assertEqual(self.account.get_account("last-slot")["quota"], 0)
        self.assertTrue(self.account.get_account("last-slot")["capacity_used_since_observation"])

    def test_durable_occupancy_precedes_temporary_slot_release(self):
        entered, finish = threading.Event(), threading.Event()
        original_handler = self.tasks.generation_handler
        def held_handler(payload):
            entered.set()
            finish.wait(3)
            return original_handler(payload)
        self.tasks.generation_handler = held_handler
        released_with_durable_slot = []
        def release(_token):
            records = self.tasks.list_tasks(self.key_a, ["held"])["items"]
            released_with_durable_slot.append(bool(records and records[0].get("upstream_unfinished")))
        body = {"client_task_id": "held", "prompt": "sample"}
        with patch("services.account_service.account_service.get_account", return_value={"quota": 1}), patch("services.account_service.account_service.release_image_slot", side_effect=release):
            try:
                self.client.post("/api/image-tasks/generations", headers=self.headers(), json=body)
                self.assertTrue(entered.wait(2))
                self.assertEqual(released_with_durable_slot, [True])
                self.client.post("/api/image-tasks/generations", headers=self.headers(), json={**body, "client_task_id": "racer"})
                for _ in range(100):
                    second = self.tasks.list_tasks(self.key_a, ["racer"])["items"][0]
                    if second["status"] == "error":
                        break
                    time.sleep(.01)
                self.assertEqual(second["error_code"], "IMAGE_RESOURCE_UNAVAILABLE")
                self.assertFalse(second.get("upstream_unfinished", False))
                self.assertEqual(self.calls, [])
            finally:
                finish.set()
                for _ in range(200):
                    final = self.tasks.list_tasks(self.key_a, ["held"])["items"][0]
                    if final["status"] in {"success", "error"}:
                        break
                    time.sleep(.01)

    def test_capacity_zero_missing_invalid_and_stale_are_distinct(self):
        for remaining in (None, "", "unknown", True, -1, float("inf")):
            self.assertIsNone(observed_capacity({"limits_progress": [{"feature_name": "image_gen", "remaining": remaining}]})["remaining"])
        observed = {"limits_progress": [{"feature_name": "image_gen", "remaining": 0, "reset_after": 3600}], "capacity_observed_at": "2026-09-16T00:00:00Z"}
        self.assertEqual(observed_capacity(observed)["remaining"], 0)
        self.assertEqual(observed_capacity(observed)["state"], "observed")
        self.assertEqual(observed_capacity({**observed, "capacity_used_since_observation": True})["state"], "stale")
        self.assertEqual(observed_capacity({**observed, "capacity_read_failed_at": "later"})["state"], "read_failed")
        self.assertIsNone(observed_capacity(observed)["codex_capacity"])


if __name__ == "__main__":
    unittest.main()
