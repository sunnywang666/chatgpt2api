from __future__ import annotations

import base64
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.image_tasks as image_tasks_module
from api.company_requests import company_request_boundary
from api.external_images import external_image_boundary
from services.image_thread import ImageThreadError


AUTH_HEADERS = {"Authorization": "Bearer chatgpt2api"}
PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPkEpFjYGBgYgADAALmAEAUQs4PAAAAAElFTkSuQmCC")
DATA_IMAGE_URL = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"


class FakeImageTaskService:
    def __init__(self):
        self.generation_calls = []
        self.edit_calls = []
        self.resume_calls = []
        self.adoption_calls = []
        self.manual_recovery_calls = []

    def submit_generation(self, identity, **kwargs):
        self.generation_calls.append((identity, kwargs))
        return {
            "id": kwargs["client_task_id"],
            "status": "success",
            "mode": "generate",
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
            "data": [{"url": f"{kwargs['base_url']}/images/fake.png"}],
        }

    def submit_edit(self, identity, **kwargs):
        self.edit_calls.append((identity, kwargs))
        return {
            "id": kwargs["client_task_id"],
            "status": "queued",
            "mode": "edit",
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
        }

    def list_tasks(self, _identity, ids):
        return {
            "items": [
                {
                    "id": task_id,
                    "status": "success",
                    "mode": "generate",
                    "created_at": "2026-01-01 00:00:00",
                    "updated_at": "2026-01-01 00:00:00",
                    "data": [{"url": "http://testserver/images/fake.png"}],
                }
                for task_id in ids
                if task_id != "missing"
            ],
            "missing_ids": [task_id for task_id in ids if task_id == "missing"],
        }

    def resume_poll(self, identity, task_id, extra_timeout_secs, base_url, allow_unrecoverable_retry=False):
        self.resume_calls.append((identity, task_id, extra_timeout_secs, base_url, allow_unrecoverable_retry))
        return {
            "id": task_id,
            "status": "running",
            "mode": "generate",
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
        }

    def adopt_latest_conversation_image(self, identity, task_id, **kwargs):
        self.adoption_calls.append((identity, task_id, kwargs))
        return {
            "id": task_id,
            "status": "success",
            "mode": "generate",
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
            "adopted_source_request_message_id": kwargs.get("source_request_message_id"),
            "adopted_source_image_message_id": kwargs.get("source_image_message_id"),
        }

    def recover_manual(self, identity, task_id, **kwargs):
        self.manual_recovery_calls.append((identity, task_id, kwargs))
        return {
            "id": task_id,
            "status": "success",
            "mode": "generate",
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
            "data": [{"url": "http://testserver/images/manual.png"}],
        }


class ImageTasksApiTests(unittest.TestCase):
    def test_b_edit_passes_request_owned_model_without_a_global_setting_write(self):
        response = self.client.post("/api/image-tasks/edits", headers=AUTH_HEADERS, json={
            "client_task_id": "b-chat-image", "prompt": "edit product", "model": "gpt-image-2",
            "images": [{"url": DATA_IMAGE_URL}], "upstream_model": "gpt-5-6-instant",
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.fake_service.edit_calls[0][1]["upstream_model"], "gpt-5-6-instant")

    def setUp(self):
        self.fake_service = FakeImageTaskService()
        self.service_patcher = mock.patch.object(image_tasks_module, "image_task_service", self.fake_service)
        self.service_patcher.start()
        self.addCleanup(self.service_patcher.stop)
        self.identity_patcher = mock.patch.object(
            image_tasks_module,
            "require_identity",
            return_value={"id": "test-key", "name": "Test", "role": "admin"},
        )
        self.identity_patcher.start()
        self.addCleanup(self.identity_patcher.stop)
        app = FastAPI()
        app.include_router(image_tasks_module.create_router())
        self.client = TestClient(app)

    def test_create_generation_task(self):
        response = self.client.post(
            "/api/image-tasks/generations",
            headers=AUTH_HEADERS,
            json={
                "client_task_id": "task-1",
                "prompt": "cat",
                "model": "gpt-image-2",
                "provider_binding_id": "cb-1",
                "provider_account_identity": "account-1",
                "client_conversation_id": "content-conversation-1",
                "retain_conversation": True,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["id"], "task-1")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(len(self.fake_service.generation_calls), 1)
        request = self.fake_service.generation_calls[0][1]
        self.assertEqual(request["conversation_id"], "")
        self.assertEqual(request["parent_message_id"], "")

    def test_create_edit_task_accepts_multiple_images(self):
        """测试图片编辑任务接口支持多个上传图片。"""
        response = self.client.post(
            "/api/image-tasks/edits",
            headers=AUTH_HEADERS,
            data={"client_task_id": "edit-1", "prompt": "edit", "model": "gpt-image-2"},
            files=[
                ("image", ("one.png", PNG_BYTES, "image/png")),
                ("image", ("two.png", PNG_BYTES, "image/png")),
            ],
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["id"], "edit-1")
        self.assertEqual(len(self.fake_service.edit_calls), 1)
        images = self.fake_service.edit_calls[0][1]["images"]
        self.assertEqual(len(images), 2)

    def test_resume_poll_uses_the_current_authoritative_provider_base_url(self):
        response = self.client.post(
            "/api/image-tasks/task-1/resume-poll",
            headers=AUTH_HEADERS,
            json={"extra_timeout_secs": 45},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "running")
        self.assertEqual(self.fake_service.resume_calls[0][1:], ("task-1", 45.0, "http://testserver", False))

    def test_resume_poll_forwards_explicit_unrecoverable_retry_authorization(self):
        response = self.client.post(
            "/api/image-tasks/task-1/resume-poll",
            headers=AUTH_HEADERS,
            json={"extra_timeout_secs": 45, "allow_unrecoverable_retry": True},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.fake_service.resume_calls[0][1:], ("task-1", 45.0, "http://testserver", True))

    def test_adopt_latest_conversation_image_forwards_exact_authority_and_source_nodes(self):
        response = self.client.post(
            "/api/image-tasks/policy-task/adopt-latest-conversation-image",
            headers=AUTH_HEADERS,
            json={
                "provider_binding_id": "binding-1",
                "provider_account_identity": "account-1",
                "client_conversation_id": "client-chat-1",
                "conversation_id": "conversation-1",
                "source_request_message_id": "manual-latest",
                "source_image_message_id": "latest-image",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        identity, task_id, kwargs = self.fake_service.adoption_calls[0]
        self.assertEqual(identity["id"], "test-key")
        self.assertEqual(task_id, "policy-task")
        self.assertEqual(kwargs, {
            "provider_binding_id": "binding-1",
            "provider_account_identity": "account-1",
            "client_conversation_id": "client-chat-1",
            "conversation_id": "conversation-1",
            "source_request_message_id": "manual-latest",
            "source_image_message_id": "latest-image",
            "base_url": "http://testserver",
        })

    def test_adopt_latest_conversation_image_requires_complete_authority(self):
        response = self.client.post(
            "/api/image-tasks/policy-task/adopt-latest-conversation-image",
            headers=AUTH_HEADERS,
            json={"provider_binding_id": "binding-1"},
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.fake_service.adoption_calls, [])

    def test_manual_recover_accepts_exact_workbench_authority_without_a_conversation_override(self):
        response = self.client.post(
            "/api/image-tasks/policy-task/manual-recover",
            headers=AUTH_HEADERS,
            json={
                "provider_binding_id": "binding-1",
                "provider_account_identity": "account-1",
                "client_conversation_id": "client-chat-1",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["id"], "policy-task")
        self.assertEqual(self.fake_service.manual_recovery_calls, [(
            {"id": "test-key", "name": "Test", "role": "admin"},
            "policy-task",
            {
                "provider_binding_id": "binding-1",
                "provider_account_identity": "account-1",
                "client_conversation_id": "client-chat-1",
                "base_url": "http://testserver",
            },
        )])

    def test_manual_recover_rejects_missing_or_extra_authority(self):
        for body in (
            {"provider_binding_id": "binding-1"},
            {
                "provider_binding_id": "binding-1",
                "provider_account_identity": "account-1",
                "client_conversation_id": "client-chat-1",
                "conversation_id": "caller-chosen-conversation",
            },
        ):
            with self.subTest(body=body):
                response = self.client.post(
                    "/api/image-tasks/policy-task/manual-recover",
                    headers=AUTH_HEADERS,
                    json=body,
                )
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.fake_service.manual_recovery_calls, [])

    def test_manual_recover_returns_upstream_read_429_with_retry_after(self):
        error = ImageThreadError("RECOVERY_RATE_LIMITED", status=429)
        error.retry_after = 12
        with mock.patch.object(self.fake_service, "recover_manual", side_effect=error):
            response = self.client.post(
                "/api/image-tasks/policy-task/manual-recover",
                headers=AUTH_HEADERS,
                json={
                    "provider_binding_id": "binding-1",
                    "provider_account_identity": "account-1",
                    "client_conversation_id": "client-chat-1",
                },
            )
        self.assertEqual(response.status_code, 429, response.text)
        self.assertEqual(response.headers["Retry-After"], "12")
        self.assertEqual(response.json()["detail"]["rate_limit"]["layer"], "chatgpt_upstream")

    def test_manual_recover_uses_workbench_direct_bearer_route_without_opening_other_ingress(self):
        app = FastAPI()
        app.middleware("http")(external_image_boundary)
        app.middleware("http")(company_request_boundary)
        app.include_router(image_tasks_module.create_router())
        client = TestClient(app)
        body = {
            "provider_binding_id": "binding-1",
            "provider_account_identity": "account-1",
            "client_conversation_id": "client-chat-1",
        }
        response = client.post(
            "/api/image-tasks/policy-task/manual-recover",
            headers={**AUTH_HEADERS, "x-workbench-consumer": "listing"},
            json=body,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["id"], "policy-task")
        self.assertEqual(len(self.fake_service.manual_recovery_calls), 1)
        blocked = client.post(
            "/api/image-tasks/policy-task/manual-recover",
            headers={**AUTH_HEADERS, "x-workbench-image-client": "1"},
            json=body,
        )
        self.assertEqual(blocked.status_code, 404)
        self.assertEqual(len(self.fake_service.manual_recovery_calls), 1)

    def test_create_edit_task_accepts_image_url(self):
        """测试图片编辑任务接口支持表单 image_url 引用。"""
        response = self.client.post(
            "/api/image-tasks/edits",
            headers=AUTH_HEADERS,
            data={
                "client_task_id": "edit-url-1",
                "prompt": "edit",
                "model": "gpt-image-2",
                "image_url": DATA_IMAGE_URL,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.fake_service.edit_calls), 1)
        images = self.fake_service.edit_calls[0][1]["images"]
        self.assertEqual(images, [(PNG_BYTES, "image_url.png", "image/png")])

    def test_list_tasks_reports_missing_ids(self):
        response = self.client.get("/api/image-tasks?ids=task-1,missing", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual([item["id"] for item in payload["items"]], ["task-1"])
        self.assertEqual(payload["missing_ids"], ["missing"])


if __name__ == "__main__":
    unittest.main()
