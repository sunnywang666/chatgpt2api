from __future__ import annotations

import base64
import unittest
from unittest import mock

from services.config import config
from services.openai_backend_api import ImageContentPolicyError, OpenAIBackendAPI, _is_content_policy_error
from services.protocol.conversation import (
    ConversationRequest,
    ImageOutput,
    extract_conversation_ids,
    stream_image_outputs,
)
from services.protocol.openai_v1_response import stream_image_response


def _conversation(file_ids: list[str], sediment_ids: list[str] | None = None) -> dict:
    parts: list[object] = [
        {"content_type": "image_asset_pointer", "asset_pointer": f"file-service://{file_id}"}
        for file_id in file_ids
    ]
    parts.extend(f"sediment://{sediment_id}" for sediment_id in (sediment_ids or []))
    return {
        "current_node": "tool",
        "mapping": {
            "request": {
                "parent": "prior-turn",
                "message": {"author": {"role": "user"}},
            },
            "tool": {
                "parent": "request",
                "message": {
                    "author": {"role": "tool"},
                    "create_time": 1,
                    "metadata": {"async_task_type": "image_gen"},
                    "content": {"content_type": "multimodal_text", "parts": parts},
                }
            }
        }
    }


class FakeBackend(OpenAIBackendAPI):
    def __init__(self, conversations: list[dict] | None = None) -> None:
        self.conversations = conversations or []
        self.calls = 0
        self.file_urls: dict[str, str] = {}
        self.sediment_urls: dict[str, str] = {}

    def _get_conversation(self, conversation_id: str) -> dict:
        self.calls += 1
        index = min(self.calls - 1, len(self.conversations) - 1)
        return self.conversations[index]

    def _get_file_download_url(self, file_id: str) -> str:
        return self.file_urls.get(file_id, "")

    def _get_attachment_download_url(self, conversation_id: str, attachment_id: str) -> str:
        return self.sediment_urls.get(attachment_id, "")


class MultiImageResultTests(unittest.TestCase):
    def test_stream_id_extractor_keeps_full_file_ids(self) -> None:
        payload = (
            '{"conversation_id":"conv-1"} '
            'file-service://file-first_123-extra sediment://sed-second_456-extra'
        )

        conversation_id, file_ids, sediment_ids = extract_conversation_ids(payload)

        self.assertEqual(conversation_id, "conv-1")
        self.assertEqual(file_ids, ["file-first_123-extra"])
        self.assertEqual(sediment_ids, ["sed-second_456-extra"])

    def test_conversation_record_extractor_finds_all_generated_assets(self) -> None:
        backend = FakeBackend()
        conversation = {
            "mapping": {
                "user": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["file-service://file-user-input"]},
                    }
                },
                "tool": {
                    "message": {
                        "author": {"role": "tool"},
                        "create_time": 1,
                        "metadata": {
                            "async_task_type": "image_gen",
                            "nested": {"asset": "file-service://file-second"},
                        },
                        "content": {
                            "content_type": "text",
                            "parts": [
                                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-first"},
                                "sediment://sed-first",
                            ],
                        },
                    }
                },
                "assistant": {
                    "message": {
                        "author": {"role": "assistant"},
                        "create_time": 2,
                        "metadata": {},
                        "content": {
                            "parts": [
                                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-third"}
                            ]
                        },
                    }
                },
            }
        }

        conversation["current_node"] = "assistant"
        conversation["mapping"]["tool"]["parent"] = "user"
        conversation["mapping"]["assistant"]["parent"] = "tool"
        records = backend._extract_image_tool_records(conversation, "user")
        file_ids = [file_id for record in records for file_id in record["file_ids"]]
        sediment_ids = [sediment_id for record in records for sediment_id in record["sediment_ids"]]

        self.assertEqual(file_ids, ["file-first", "file-second", "file-third"])
        self.assertEqual(sediment_ids, ["sed-first"])

    def test_poll_waits_for_generated_asset_ids_to_settle(self) -> None:
        backend = FakeBackend([
            _conversation(["file-one"]),
            _conversation(["file-one", "file-two"], ["sed-one"]),
            _conversation(["file-one", "file-two"], ["sed-one"]),
        ])

        with (
            mock.patch.dict(config.data, {
                "image_poll_initial_wait_secs": 0,
                "image_poll_interval_secs": 0.5,
                "image_check_before_hit_enabled": True,
                "image_settle_enabled": True,
                "image_settle_secs": 0.5,
            }),
            mock.patch("services.openai_backend_api.time.sleep", lambda _seconds: None),
        ):
            file_ids, sediment_ids = backend._poll_image_results(
                "conv-1", timeout_secs=10, request_message_id="request",
            )

        self.assertEqual(file_ids, ["file-one", "file-two"])
        self.assertEqual(sediment_ids, ["sed-one"])
        self.assertEqual(backend.calls, 3)

    def test_resolver_uses_file_and_sediment_urls(self) -> None:
        backend = FakeBackend()
        backend.file_urls = {"file-one": "https://files.test/one.png"}
        backend.sediment_urls = {
            "sed-one": "https://attachments.test/one.png",
            "sed-two": "https://attachments.test/two.png",
        }

        urls = backend._resolve_image_urls("conv-1", ["file-one"], ["sed-one", "sed-two"])

        self.assertEqual(urls, [
            "https://files.test/one.png",
            "https://attachments.test/one.png",
            "https://attachments.test/two.png",
        ])

    def test_resolver_keeps_stream_ids_when_poll_extension_fails(self) -> None:
        backend = FakeBackend()
        backend.file_urls = {"file-one": "https://files.test/one.png"}
        backend._get_conversation = mock.Mock(side_effect=RuntimeError("poll failed"))

        with mock.patch("services.openai_backend_api.time.sleep", lambda _seconds: None):
            urls = backend.resolve_conversation_image_urls(
                "conv-1", ["file-one"], [], poll=True, request_message_id="request",
            )

        self.assertEqual(urls, ["https://files.test/one.png"])

    def test_policy_detection_requires_an_explicit_refusal(self) -> None:
        self.assertFalse(_is_content_policy_error('{"reason":"delivery_return_policy"}'))
        self.assertFalse(_is_content_policy_error("I cannot help with this color adjustment."))
        self.assertTrue(_is_content_policy_error("This request violates our content policy."))

    def test_poll_uses_only_the_current_request_branch(self) -> None:
        backend = FakeBackend([{
            "current_node": "current-image",
            "mapping": {
                "old-request": {"parent": "root", "message": {"author": {"role": "user"}}},
                "old-rejection": {
                    "parent": "old-request",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ['{"reason":"delivery_return_policy"}']},
                    },
                },
                "old-image": {
                    "parent": "old-rejection",
                    "message": {
                        "author": {"role": "tool"},
                        "create_time": 1,
                        "metadata": {"async_task_type": "image_gen"},
                        "content": {"parts": [
                            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-old"},
                        ]},
                    },
                },
                "current-request": {"parent": "root", "message": {"author": {"role": "user"}}},
                "current-image": {
                    "parent": "current-request",
                    "message": {
                        "author": {"role": "tool"},
                        "create_time": 2,
                        "metadata": {"async_task_type": "image_gen"},
                        "content": {"parts": [
                            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-current"},
                        ]},
                    },
                },
                "parallel-image": {
                    "parent": "current-request",
                    "message": {
                        "author": {"role": "tool"},
                        "create_time": 3,
                        "metadata": {"async_task_type": "image_gen"},
                        "content": {"parts": [
                            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-parallel"},
                        ]},
                    },
                },
            },
        }])

        with (
            mock.patch.dict(config.data, {"image_poll_initial_wait_secs": 0, "image_check_before_hit_enabled": False}),
            mock.patch("services.openai_backend_api.time.sleep", lambda _seconds: None),
        ):
            file_ids, sediment_ids = backend._poll_image_results(
                "conv-1", timeout_secs=10, request_message_id="current-request",
            )

        self.assertEqual(file_ids, ["file-current"])
        self.assertEqual(sediment_ids, [])

    def test_request_branch_stops_before_a_later_user_turn(self) -> None:
        backend = FakeBackend()
        conversation = {
            "current_node": "later-image",
            "mapping": {
                "request": {"parent": "root", "message": {"author": {"role": "user"}}},
                "request-image": {
                    "parent": "request",
                    "message": {
                        "author": {"role": "tool"}, "create_time": 1,
                        "metadata": {"async_task_type": "image_gen"},
                        "content": {"parts": [
                            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-request"},
                        ]},
                    },
                },
                "later-user": {"parent": "request-image", "message": {"author": {"role": "user"}}},
                "later-rejection": {
                    "parent": "later-user",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["This request violates our content policy."]},
                    },
                },
                "later-image": {
                    "parent": "later-rejection",
                    "message": {
                        "author": {"role": "tool"}, "create_time": 2,
                        "metadata": {"async_task_type": "image_gen"},
                        "content": {"parts": [
                            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-later"},
                        ]},
                    },
                },
            },
        }

        records = backend._extract_image_tool_records(conversation, "request")

        self.assertEqual([record["file_ids"] for record in records], [["file-request"]])
        self.assertEqual(backend._find_content_policy_error_in_conversation(conversation, "request"), "")

    def test_stream_sets_persisted_request_id_before_the_backend_starts(self) -> None:
        class Backend:
            def stream_conversation(self, **_kwargs):
                self.started_with_request_id = self.image_request_message_id
                return iter(["[DONE]"])

            def resolve_conversation_image_urls(self, *_args, **_kwargs):
                return []

        callback = lambda _step: None
        callback.request_message_id = "request-before-post"
        backend = Backend()
        list(stream_image_outputs(
            backend,
            ConversationRequest(prompt="cat", model="gpt-image-2", progress_callback=callback),
        ))

        self.assertEqual(backend.image_request_message_id, "request-before-post")
        self.assertEqual(backend.started_with_request_id, "request-before-post")

    def test_poll_requires_the_submitted_message_id(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "submitted message id missing"):
            FakeBackend()._poll_image_results("conv-1", timeout_secs=1)

    def test_poll_preserves_a_current_request_policy_rejection(self) -> None:
        backend = FakeBackend([{
            "current_node": "rejection",
            "mapping": {
                "request": {"parent": "root", "message": {"author": {"role": "user"}}},
                "rejection": {
                    "parent": "request",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["This request violates our content policy."]},
                    },
                },
            },
        }])

        with mock.patch.dict(config.data, {"image_poll_initial_wait_secs": 0}):
            with self.assertRaises(ImageContentPolicyError):
                backend._poll_image_results("conv-1", timeout_secs=10, request_message_id="request")

    def test_responses_stream_emits_all_image_output_items(self) -> None:
        first = base64.b64encode(b"first").decode("ascii")
        second = base64.b64encode(b"second").decode("ascii")
        events = list(stream_image_response(
            [ImageOutput(
                kind="result",
                model="gpt-image-2",
                index=1,
                total=1,
                data=[{"b64_json": first}, {"b64_json": second}],
            )],
            "draw two options",
            "gpt-image-2",
        ))

        done_events = [event for event in events if event.get("type") == "response.output_item.done"]
        completed = next(event["response"] for event in events if event.get("type") == "response.completed")

        self.assertEqual([event["output_index"] for event in done_events], [0, 1])
        self.assertEqual([item["result"] for item in completed["output"]], [first, second])


if __name__ == "__main__":
    unittest.main()
