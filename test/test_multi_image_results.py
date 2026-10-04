from __future__ import annotations

import base64
import json
import unittest
from unittest import mock

from services.config import config
from services.openai_backend_api import (
    ImageContentPolicyError,
    ImagePollTimeoutError,
    OpenAIBackendAPI,
    _is_content_policy_error,
)
from services.protocol import openai_v1_image_generations
from services.protocol.conversation import (
    ConversationRequest,
    ImageGenerationError,
    ImageOutput,
    _message_output_error,
    extract_conversation_ids,
    iter_conversation_payloads,
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
    def test_polled_only_result_reaches_exact_tool_turn_confirmation(self) -> None:
        from services.image_thread import finished_parent
        asset = "file_00000000" + "a" * 24
        document = _conversation([asset], [asset])
        document["conversation_id"] = "conv-1"
        document["mapping"]["code"] = {
            "parent": "request", "message": {"id": "code", "author": {"role": "assistant"},
                "status": "finished_successfully", "end_turn": False}}
        document["mapping"]["tool"]["parent"] = "code"
        for mid in ("request", "tool"):
            document["mapping"][mid]["message"].update(id=mid, status="finished_successfully")
        backend = FakeBackend([document])
        backend.file_urls[asset] = backend.sediment_urls[asset] = "https://files.test/result.png"
        backend._query_backend_tasks = lambda **kwargs: []
        backend.stream_conversation = mock.Mock(return_value=iter([
            json.dumps({"conversation_id": "conv-1"}), "[DONE]"]))
        saved = []
        callback = lambda _step: None
        callback.request_message_id = "request"
        callback.record_result_ids = lambda files, sediments: saved.append((files, sediments))
        backend.progress_callback = callback
        backend.download_image_bytes = mock.Mock(return_value=[b"original-image-bytes"])
        with (
            mock.patch.object(type(config), "image_poll_initial_wait_secs", property(lambda _: 0)),
            mock.patch.object(type(config), "image_settle_enabled", property(lambda _: False)),
            mock.patch.object(type(config), "image_check_before_hit_enabled", property(lambda _: False)),
            mock.patch("services.protocol.conversation._get_detailed_error_from_tasks", return_value=""),
        ):
            outputs = list(stream_image_outputs(backend, ConversationRequest(
                prompt="Change the handle to green", model="gpt-image-2", images=["fixture"],
                progress_callback=callback)))
        self.assertEqual(saved, [([asset], [asset])])
        self.assertEqual(finished_parent(document, "conv-1", "request", expected_parent="prior-turn",
            expected_result_ids=saved[0][0] + saved[0][1]), "tool")
        self.assertEqual(len([x for x in outputs if x.kind == "result"]), 1)
        backend.stream_conversation.assert_called_once()
        backend.download_image_bytes.assert_called_once_with(["https://files.test/result.png"])

    def test_generated_ids_are_saved_before_stream_or_url_resolution_can_fail(self) -> None:
        for failure_phase in ("stream", "resolve"):
            with self.subTest(failure_phase=failure_phase):
                saved = []
                callback = lambda _step: None
                callback.record_result_ids = lambda files, sediments: saved.append((files, sediments))
                class Backend:
                    def stream_conversation(self, **_kwargs):
                        yield json.dumps({"conversation_id": "original", "message": {
                            "author": {"role": "user"}, "content": {"parts": ["file-service://uploaded-input"]},
                        }})
                        yield json.dumps({"conversation_id": "original", "message": {
                            "author": {"role": "tool"}, "metadata": {"async_task_type": "image_gen"},
                            "content": {"parts": ["file-service://generated-image sediment://generated-sediment"]},
                        }})
                        if failure_phase == "stream":
                            raise ConnectionError("stream ended after image tool result")
                        yield "[DONE]"

                    def resolve_conversation_image_urls(self, *_args, **_kwargs):
                        if saved != [(["generated-image"], ["generated-sediment"])]:
                            raise AssertionError("result IDs were not durable before URL resolution")
                        raise ConnectionError("URL resolution failed")

                with self.assertRaises(ConnectionError):
                    list(stream_image_outputs(Backend(), ConversationRequest(
                        prompt="circle", model="gpt-image-2", progress_callback=callback,
                    )))
                self.assertEqual(saved, [(["generated-image"], ["generated-sediment"])])

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
        observed = []
        backend.progress_callback = lambda _step: None
        backend.progress_callback.record_pending_result_ids = lambda files, sediments: observed.append((files, sediments))

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
        self.assertEqual(observed, [(["file-one"], []), (["file-one", "file-two"], ["sed-one"])])

    def test_strict_terminal_shortcut_does_not_change_recovery_or_legacy_polling(self) -> None:
        for mode in ("initial-ids", "initial-document", "fresh-recovery", "check-disabled",
                     "settle-disabled", "asset-drift", "legacy"):
            with self.subTest(mode=mode):
                docs = [_conversation(["file-one"])]
                if mode == "asset-drift":
                    docs += [_conversation(["file-two"]), _conversation(["file-two"])]
                backend = FakeBackend(docs)
                check = mock.Mock(return_value=True)
                if mode != "legacy": backend.image_poll_terminal_check = check
                # First changed observation is not confirmed; accumulated stale
                # IDs must never be passed to the shortcut on a later snapshot.
                if mode == "asset-drift": check.return_value = False
                kwargs = {"request_message_id": "request"}
                if mode == "initial-ids": kwargs["initial_file_ids"] = ["file-one"]
                if mode == "initial-document": kwargs["initial_document"] = docs[0]
                if mode == "fresh-recovery": kwargs["require_fresh_result_ids"] = True
                with mock.patch.dict(config.data, {
                    "image_poll_initial_wait_secs": 0, "image_poll_interval_secs": .01,
                    "image_check_before_hit_enabled": mode != "check-disabled",
                    "image_settle_enabled": mode != "settle-disabled", "image_settle_secs": .01,
                }):
                    if mode == "initial-document":
                        with self.assertRaises(ImagePollTimeoutError):
                            backend._poll_image_results("conv-1", 1, **kwargs)
                    else:
                        backend._poll_image_results("conv-1", 10, **kwargs)
                self.assertEqual(check.call_count, 1 if mode == "asset-drift" else 0)
                if mode in {"fresh-recovery", "legacy"}: self.assertEqual(backend.calls, 2)
                if mode == "initial-document": self.assertEqual(backend.calls, 0)

    def test_poll_keeps_request_scoped_ids_when_settle_exhausts_budget(self) -> None:
        document = _conversation(["file-original"])
        document["mapping"].update({
            "later-user": {"parent": "tool", "message": {"author": {"role": "user"}}},
            "later-image": {"parent": "later-user", "message": {
                "author": {"role": "tool"}, "metadata": {"async_task_type": "image_gen"},
                "content": {"parts": ["file-service://file-later"]},
            }},
        })
        document["current_node"] = "later-image"
        backend = FakeBackend([document])
        backend._query_backend_tasks = mock.Mock(return_value=[])
        observed = []
        backend.progress_callback = lambda _step: None
        backend.progress_callback.record_pending_result_ids = lambda files, sediments: observed.append((files, sediments))
        clock = [100.0]
        def advance(seconds):
            clock[0] += seconds
        with (
            mock.patch.dict(config.data, {"image_poll_initial_wait_secs": 0,
                "image_check_before_hit_enabled": True, "image_settle_enabled": True, "image_settle_secs": 2}),
            mock.patch("services.openai_backend_api.time.time", lambda: clock[0]),
            mock.patch("services.openai_backend_api.time.sleep", advance),
        ):
            with self.assertRaises(ImagePollTimeoutError):
                backend._poll_image_results("conv-1", timeout_secs=1, request_message_id="request")
        self.assertEqual(backend.calls, 1)
        self.assertEqual(observed, [(["file-original"], [])])

    def test_pending_ids_require_a_fresh_matching_request_branch(self) -> None:
        backend = FakeBackend([_conversation([])])
        backend._query_backend_tasks = mock.Mock(return_value=[])
        clock = [100.0]
        def advance(seconds):
            clock[0] += seconds
        with (
            mock.patch.dict(config.data, {"image_poll_initial_wait_secs": 0,
                "image_poll_interval_secs": 0.1, "image_check_before_hit_enabled": True,
                "image_settle_enabled": True, "image_settle_secs": 0.1}),
            mock.patch("services.openai_backend_api.time.time", lambda: clock[0]),
            mock.patch("services.openai_backend_api.time.sleep", advance),
        ):
            with self.assertRaises(ImagePollTimeoutError):
                backend._poll_image_results("conv-1", timeout_secs=1, request_message_id="request",
                    initial_file_ids=["pending-file"], require_fresh_result_ids=True)
        self.assertGreaterEqual(backend.calls, 1)

    def test_recovery_reuses_one_fresh_read_across_short_poll_windows(self) -> None:
        backend = FakeBackend()
        backend._get_conversation = mock.Mock(side_effect=AssertionError("second GET would wait read60"))
        backend._query_backend_tasks = mock.Mock(side_effect=AssertionError("snapshot already read"))
        observed = []
        backend.progress_callback = lambda _step: None
        backend.progress_callback.record_pending_result_ids = lambda files, sediments: observed.append((files, sediments))
        with mock.patch.dict(config.data, {
            "image_poll_initial_wait_secs": 10, "image_check_before_hit_enabled": True,
            "image_settle_enabled": True, "image_settle_secs": 2,
        }), mock.patch("services.openai_backend_api.time.sleep") as sleep:
            with self.assertRaisesRegex(ImagePollTimeoutError, "尚未确认稳定图片结果"):
                backend._poll_image_results("conv-1", 5, request_message_id="request",
                    initial_document=_conversation(["one"]))
            self.assertEqual(observed, [(["one"], [])])
            with self.assertRaises(ImagePollTimeoutError):
                backend._poll_image_results("conv-1", 5, request_message_id="request",
                    initial_file_ids=["one"], require_fresh_result_ids=True,
                    initial_document=_conversation(["one", "two"]))
            self.assertEqual(observed[-1], (["one", "two"], []))
            self.assertEqual(backend._poll_image_results("conv-1", 5,
                request_message_id="request", initial_file_ids=["one", "two"],
                require_fresh_result_ids=True, initial_document=_conversation(["two", "one"])),
                (["one", "two"], []))
            sleep.assert_not_called()
        backend._get_conversation.assert_not_called()
        backend._query_backend_tasks.assert_not_called()

    def test_fresh_recovery_snapshot_cannot_confirm_missing_or_later_turn_ids(self) -> None:
        later = _conversation([])
        later["mapping"].update({
            "later-user": {"parent": "tool", "message": {"author": {"role": "user"}}},
            "later-image": {"parent": "later-user", "message": {
                "author": {"role": "tool"}, "metadata": {"async_task_type": "image_gen"},
                "content": {"parts": ["file-service://one"]}}},
        })
        later["current_node"] = "later-image"
        with mock.patch.dict(config.data, {"image_check_before_hit_enabled": True,
                "image_settle_enabled": True}), mock.patch("services.openai_backend_api.time.sleep"):
            for document in (_conversation([]), later):
                with self.subTest(document=document):
                    backend = FakeBackend()
                    backend._get_conversation = mock.Mock(side_effect=AssertionError("unexpected GET"))
                    with self.assertRaises(ImagePollTimeoutError):
                        backend._poll_image_results("conv-1", 5, request_message_id="request",
                            initial_file_ids=["one"], require_fresh_result_ids=True,
                            initial_document=document)
                    backend._get_conversation.assert_not_called()

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

    def test_failed_poll_does_not_promote_pending_ids_to_caller(self) -> None:
        backend = FakeBackend()
        files, sediments, pending = [], [], []
        def interrupted_poll(*args, **kwargs):
            self.assertEqual(kwargs["request_message_id"], "request")
            pending.append((["unsettled-file"], ["unsettled-sediment"]))
            raise ImagePollTimeoutError("not settled", "conv-1")
        backend._poll_image_results = interrupted_poll
        with mock.patch.object(type(config), "image_check_before_hit_enabled", property(lambda _: False)), \
             mock.patch("services.openai_backend_api.OpenAIBackendAPI._query_backend_tasks", return_value=[]):
            with self.assertRaises(ImagePollTimeoutError):
                backend.resolve_conversation_image_urls("conv-1", files, sediments,
                    request_message_id="request")
        self.assertTrue(pending)
        self.assertEqual((files, sediments), ([], []))

    def test_policy_detection_requires_an_explicit_refusal(self) -> None:
        self.assertFalse(_is_content_policy_error('{"reason":"delivery_return_policy"}'))
        self.assertFalse(_is_content_policy_error("I cannot help with this color adjustment."))
        self.assertFalse(_is_content_policy_error("Generated without content policy violations."))
        self.assertFalse(_is_content_policy_error("No content policy violation."))
        self.assertTrue(_is_content_policy_error("This request violates our content policy."))
        self.assertTrue(_is_content_policy_error("I can't generate that because it violates our content policy."))

    def test_text_only_image_message_is_not_reclassified_as_policy(self) -> None:
        ordinary = _message_output_error(ImageOutput(
            kind="message", model="gpt-image-2", index=1, total=1,
            text="I can only provide a written description for this request.", conversation_id="conv-1",
        ))
        policy = _message_output_error(ImageOutput(
            kind="message", model="gpt-image-2", index=1, total=1,
            text="This request violates our content policy.", conversation_id="conv-1",
        ))

        self.assertIsInstance(ordinary, ImageGenerationError)
        self.assertEqual(ordinary.code, "NO_IMAGE_GENERATED")
        self.assertEqual(policy.code, "content_policy_violation")

        still_processing = _message_output_error(ImageOutput(
            kind="message", model="gpt-image-2", index=1, total=1,
            text="The image may still be processing. Please try again in a moment.",
            conversation_id="conv-1",
        ))
        self.assertEqual(still_processing.code, "CONVERSATION_OUTCOME_UNKNOWN")
        incomplete = _message_output_error(ImageOutput(
            kind="message", model="gpt-image-2", index=1, total=1,
            text="Image generation started upstream but the response was incomplete. Please try again.",
            conversation_id="",
        ))
        self.assertEqual(incomplete.code, "CONVERSATION_OUTCOME_UNKNOWN")

    def test_unbudgeted_stream_keeps_legacy_bounded_fallback_retries(self) -> None:
        class Backend:
            poll_calls = 0

            def stream_conversation(self, **_kwargs):
                yield json.dumps({
                    "conversation_id": "conv-1",
                    "type": "server_ste_metadata",
                    "metadata": {"turn_use_case": "image gen"},
                })
                yield "[DONE]"

            def resolve_conversation_image_urls(self, *_args, **_kwargs):
                return []

            def _query_backend_tasks(self, **_kwargs):
                return []

            def _poll_image_results(self, *_args, **_kwargs):
                type(self).poll_calls += 1
                if type(self).poll_calls < 3:
                    raise ConnectionError("temporary network failure")
                return [], []

        Backend.poll_calls = 0
        with mock.patch("services.protocol.conversation.time.sleep", return_value=None):
            outputs = list(stream_image_outputs(
                Backend(), ConversationRequest(prompt="cat", model="gpt-image-2"),
            ))

        self.assertEqual(Backend.poll_calls, 3)
        self.assertEqual(
            _message_output_error(outputs[-1]).code,
            "CONVERSATION_OUTCOME_UNKNOWN",
        )

    def test_structured_moderation_block_is_policy_even_with_generic_text(self) -> None:
        class Backend:
            def stream_conversation(self, **_kwargs):
                yield json.dumps({"type": "moderation", "moderation_response": {"blocked": True}})
                yield json.dumps({
                    "conversation_id": "conv-1",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["I cannot help with that request."]},
                    },
                })
                yield "[DONE]"

            def _query_backend_tasks(self, **_kwargs):
                return []

        with self.assertRaises(ImageContentPolicyError):
            list(stream_image_outputs(
                Backend(),
                ConversationRequest(prompt="blocked request", model="gpt-image-2"),
            ))

    def test_stream_collects_only_image_tool_output_not_input_or_tool_arguments(self) -> None:
        payloads = iter([
            json.dumps({
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": [
                        {"content_type": "image_asset_pointer", "asset_pointer": "file-service://input-file"},
                    ]},
                },
            }),
            json.dumps({
                "type": "server_ste_metadata",
                "metadata": {"tool_invoked": True, "turn_use_case": "image gen"},
            }),
            json.dumps({
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": [
                        '{"referenced_image_ids":["file-service://input-file"]}',
                    ]},
                },
            }),
            json.dumps({
                "message": {
                    "author": {"role": "tool"},
                    "metadata": {"async_task_type": "image_gen"},
                    "content": {"content_type": "multimodal_text", "parts": [
                        {"content_type": "image_asset_pointer", "asset_pointer": "file-service://generated-file"},
                    ]},
                },
            }),
            "[DONE]",
        ])

        events = list(iter_conversation_payloads(payloads))

        self.assertEqual(events[-1]["file_ids"], ["generated-file"])

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

    def test_unbound_generation_uses_the_request_id_written_by_the_provider_post(self) -> None:
        observed = {}

        class Backend:
            def __init__(self, access_token=None):
                self.access_token = access_token

            def stream_conversation(self, **_kwargs):
                self.image_request_message_id = "request-written-by-post"
                yield '{"conversation_id":"conversation-from-sse"}'
                yield "[DONE]"

            def resolve_conversation_image_urls(
                    self, conversation_id, _file_ids, _sediment_ids, **kwargs,
            ):
                observed["conversation_id"] = conversation_id
                observed["request_message_id"] = kwargs.get("request_message_id")
                return ["https://images.test/result.png"]

            def _query_backend_tasks(self, **_kwargs):
                return []

            def download_image_bytes(self, _urls):
                return [b"generated-image"]

            def close(self):
                return None

        with (
            mock.patch("services.protocol.conversation.account_service.get_available_access_token", return_value="token"),
            mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "account@example.test"}),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", Backend),
            mock.patch("services.protocol.conversation._remove_image_conversation_later"),
        ):
            result = openai_v1_image_generations.handle({
                "prompt": "cat",
                "model": "gpt-image-2",
                "n": 1,
            })

        self.assertEqual(observed, {
            "conversation_id": "conversation-from-sse",
            "request_message_id": "request-written-by-post",
        })
        self.assertEqual(len(result["data"]), 1)

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

    def test_poll_keeps_empty_tasks_and_empty_finished_text_unknown(self) -> None:
        backend = FakeBackend([{
            "current_node": "assistant",
            "mapping": {
                "request": {
                    "parent": "root",
                    "message": {"author": {"role": "user"}},
                },
                "assistant": {
                    "parent": "request",
                    "message": {
                        "author": {"role": "assistant"},
                        "status": "finished_successfully",
                        "end_turn": True,
                        "content": {"content_type": "text", "parts": []},
                    },
                },
            },
        }])
        backend._query_backend_tasks = mock.Mock(return_value=[])

        with mock.patch.dict(config.data, {
            "image_poll_initial_wait_secs": 0,
            "image_poll_interval_secs": 0.001,
        }):
            with self.assertRaises(ImagePollTimeoutError):
                backend._poll_image_results(
                    "conv-1", timeout_secs=0.005, request_message_id="request",
                )

        backend._query_backend_tasks.assert_called()
        self.assertGreaterEqual(backend.calls, 1)

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
