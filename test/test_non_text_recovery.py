"""Original-branch native image outcomes must never authorize regeneration."""
from __future__ import annotations

import copy
import json
from unittest.mock import Mock

import pytest

from services.conversation_binding_service import (
    ConversationBindingError,
    ConversationBindingService,
    NON_TEXT_RESULT_FIELD,
)
from services.public_chat_service import project_public_chat_receipt
from services.text_task_service import TextTaskService


def image_document(request_id="request-user", pointer="file-service://file_generated"):
    def node(node_id, parent, role, content, **fields):
        return {"parent": parent, "message": {
            "id": node_id, "author": {"role": role}, "status": "finished_successfully",
            "content": content, **fields,
        }}

    return {"conversation_id": "original-chat", "current_node": "later-answer", "mapping": {
        request_id: node(request_id, "prior-answer", "user", {"content_type": "text", "parts": ["original input"]}),
        "code": node("code", request_id, "assistant", {"content_type": "code", "text": "image tool call"},
                     channel="commentary", end_turn=False),
        "image-tool": node("image-tool", "code", "tool", {"content_type": "multimodal_text", "parts": [{
            "content_type": "image_asset_pointer", "asset_pointer": pointer, "width": 1024, "height": 1024,
        }]}),
        "recap": node("recap", "image-tool", "assistant", {"content_type": "reasoning_recap", "content": "done"},
                      channel="commentary", end_turn=False),
        "empty-final": node("empty-final", "recap", "assistant", {"content_type": "text", "parts": [""]},
                            channel="final", end_turn=True),
        "later-user": node("later-user", "empty-final", "user", {"content_type": "text", "parts": ["next turn"]}),
        "later-answer": node("later-answer", "later-user", "assistant", {"content_type": "text", "parts": ["next answer"]},
                             channel="final", end_turn=True),
    }}


def original_receipt():
    return {
        "provider_binding_id": "original-binding", "provider_account_identity": "original-account",
        "client_conversation_id": "original-client", "conversation_id": "original-chat",
        "request_message_id": "request-user", "request_parent_message_id": "prior-answer",
    }


def read_document(receipt, document=None):
    return ConversationBindingService._read_text_request_result(
        Mock(), receipt, document=document or image_document(receipt["request_message_id"]),
    )


@pytest.mark.parametrize("pointer", ["file-service://file_generated", "sediment://generated_01"])
def test_exact_completed_image_lineage_with_empty_final_is_non_text(pointer):
    result = read_document(original_receipt(), image_document(pointer=pointer))
    assert result["status"] == "failed"
    assert result["error_code"] == "CHAT_RESPONSE_NOT_TEXT"
    assert result["recovery_reason"] == "REQUEST_RESULT_NON_TEXT"
    assert "content" not in result
    assert result[NON_TEXT_RESULT_FIELD] == {
        "conversation_id": "original-chat", "request_message_id": "request-user",
        "final_message_id": "empty-final",
        "artifacts": [{"tool_message_id": "image-tool", "asset_pointer": pointer}],
    }


@pytest.mark.parametrize("node_id,patch", [
    ("image-tool", {"status": "in_progress"}),
    ("image-tool", {"status": "failed"}),
    ("image-tool", {"id": "another-tool"}),
    ("image-tool", {"author": {"role": "assistant"}}),
    ("image-tool", {"metadata": {"async_task_type": "image_gen"}, "content": {"content_type": "text", "parts": ["tool_invoked"]}}),
    ("code", {"status": "pending"}),
    ("code", {"id": "wrong-code-message"}),
    ("code", {"status": "unknown"}),
    ("empty-final", {"status": "in_progress"}),
    ("empty-final", {"end_turn": False}),
    ("empty-final", {"channel": "commentary"}),
    ("empty-final", {"id": "wrong-final"}),
    ("empty-final", {"content": {"content_type": "text", "parts": [None]}}),
])
def test_incomplete_or_malformed_result_is_not_terminal(node_id, patch):
    document = image_document()
    document["mapping"][node_id]["message"].update(patch)
    result = read_document(original_receipt(), document)
    assert result["status"] in {"unknown", "running"}
    assert NON_TEXT_RESULT_FIELD not in result


@pytest.mark.parametrize("part", [
    {"content_type": "image_asset_pointer"},
    {"content_type": "image_asset_pointer", "asset_pointer": "https://private.example/image?secret=hidden"},
    {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file_a?token=secret"},
    {"content_type": "text", "asset_pointer": "file-service://file_a"},
    "file-service://tool_argument_only",
])
def test_image_marker_or_argument_is_not_an_artifact(part):
    document = image_document()
    document["mapping"]["image-tool"]["message"]["content"]["parts"] = [part]
    result = read_document(original_receipt(), document)
    assert result["status"] == "unknown"
    assert result["recovery_reason"] == "REQUEST_RESULT_TERMINAL_EMPTY"


@pytest.mark.parametrize("placement", ["input", "later_user", "other_request"])
def test_images_outside_original_tool_branch_do_not_qualify(placement):
    document = image_document()
    artifact_content = document["mapping"]["image-tool"]["message"]["content"]
    document["mapping"]["image-tool"]["message"]["content"] = {"content_type": "text", "parts": [""]}
    if placement == "input":
        document["mapping"]["request-user"]["message"]["content"] = artifact_content
    else:
        document["mapping"]["unrelated-image"] = {
            "parent": "later-user" if placement == "later_user" else "other-request",
            "message": {"id": "unrelated-image", "author": {"role": "tool"},
                        "status": "finished_successfully", "content": artifact_content},
        }
    assert read_document(original_receipt(), document)["status"] == "unknown"


@pytest.mark.parametrize("branch,status", [("sibling", "finished_successfully"), ("sibling", "in_progress"),
                                          ("after_final", "finished_successfully"), ("cycle", "finished_successfully")])
def test_branch_ambiguity_and_continuations_block_terminal(branch, status):
    document = image_document()
    sibling = copy.deepcopy(document["mapping"]["empty-final"])
    sibling["message"].update(id="other-final", status=status)
    sibling["parent"] = "empty-final" if branch == "after_final" else "request-user"
    document["mapping"]["other-final"] = sibling
    if branch == "cycle":
        document["mapping"]["code"]["parent"] = "recap"
    assert read_document(original_receipt(), document)["status"] in {"unknown", "running"}


def test_nonempty_original_answer_keeps_existing_success_contract():
    document = image_document()
    document["mapping"]["empty-final"]["message"]["content"]["parts"] = ["original answer"]
    result = read_document(original_receipt(), document)
    assert result["status"] == "succeeded"
    assert result["content"] == "original answer"
    assert NON_TEXT_RESULT_FIELD not in result


def recovering_service(tmp_path, reader=None):
    executor = Mock()
    reader = reader or Mock(side_effect=read_document)
    service = TextTaskService(tmp_path / "tasks.sqlite3", runner=Mock(), executor=executor, recovery_reader=reader)
    body = {
        "client_request_id": "original-request", "client_conversation_id": "original-client",
        "messages": [{"role": "user", "content": "original input"}], "_public_route": "chat",
        "thinking_effort": "high",
    }
    service.submit("owner", body)
    service._update("owner", "original-request", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                    provider_binding_id="original-binding", provider_account_identity="original-account",
                    conversation_id="original-chat", request_parent_message_id="prior-answer",
                    recovery_requires_new_conversation=True, original_failure_phase="result_check")
    return service, body, reader, executor


def test_non_text_is_durable_private_nonretryable_and_same_id_never_resubmits(tmp_path):
    service, body, reader, executor = recovering_service(tmp_path)
    result = service.recover("owner", "original-request", allow_unrecoverable_retry=True)
    assert result["status"] == "failed"
    assert result["error_code"] == "CHAT_RESPONSE_NOT_TEXT"
    assert result["recovery_retryable"] is False
    assert result["recovery_requires_new_conversation"] is False
    assert result["recovery_next_at"] is None
    assert result["upstream_outcome"] == "completed"
    assert result["original_failure_phase"] == "result_check"
    assert result["result"] == {"type": "non_text", "artifact_type": "image", "artifact_count": 1}
    assert NON_TEXT_RESULT_FIELD not in result
    with service._db() as db:
        raw = json.loads(db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", ("owner", "original-request")).fetchone()[0])
    assert raw[NON_TEXT_RESULT_FIELD]["request_message_id"] == result["request_message_id"]
    assert raw[NON_TEXT_RESULT_FIELD]["artifacts"][0]["asset_pointer"] == "file-service://file_generated"
    assert raw["provider_account_identity"] == "original-account"
    restarted = TextTaskService(service.path, runner=Mock(), executor=executor, recovery_reader=reader)
    for _ in range(4):
        assert restarted.recover("owner", "original-request", True) == result
        assert restarted.submit("owner", body) == result
    assert reader.call_count == 1
    assert executor.submit.call_count == 1
    assert restarted.read("other-owner", "original-request")["status"] == "not_found"
    with pytest.raises(ConversationBindingError, match="different input"):
        restarted.submit("owner", {**body, "thinking_effort": "extended"})
    service._update("owner", "original-request", status="unknown", parent_message_id="late-progress")
    assert service.read("owner", "original-request") == result


@pytest.mark.parametrize("change", ["request", "conversation", "account", "binding", "final", "pointer", "extra", "missing"])
def test_malformed_or_rebound_terminal_result_remains_unknown(tmp_path, change):
    def reader(receipt):
        result = read_document(receipt)
        proof = result[NON_TEXT_RESULT_FIELD]
        if change == "request":
            proof["request_message_id"] = "different-request"
        elif change == "conversation":
            proof["conversation_id"] = result["conversation_id"] = "different-conversation"
        elif change == "account":
            result["provider_account_identity"] = "different-account"
        elif change == "binding":
            result["provider_binding_id"] = "different-binding"
        elif change == "final":
            proof["final_message_id"] = "different-final"
        elif change == "pointer":
            proof["artifacts"][0]["asset_pointer"] = "https://private.example/?secret=hidden"
        elif change == "extra":
            proof["access_token"] = "private-secret"
        else:
            result.pop(NON_TEXT_RESULT_FIELD)
        return result
    service, _, _, _ = recovering_service(tmp_path, reader)
    result = service.recover("owner", "original-request", True)
    assert result["status"] == "unknown"
    assert result["recovery_error_code"] == "RECOVERY_INVALID_RESULT"
    assert result.get("recovery_no_result_reads", 0) == 0
    assert not result.get("recovery_retryable")
    assert b"private-secret" not in service.path.read_bytes()


def test_late_non_text_read_cannot_replace_a_prior_success(tmp_path):
    def reader(receipt):
        service._update("owner", "original-request", status="succeeded", content="real recovered text", parent_message_id="text-final")
        return read_document(receipt)
    service, _, _, _ = recovering_service(tmp_path, reader)
    result = service.read("owner", "original-request")
    assert result["status"] == "succeeded"
    assert result["content"] == "real recovered text"
    assert NON_TEXT_RESULT_FIELD not in result


def test_public_projection_exposes_summary_only(tmp_path):
    service, _, _, _ = recovering_service(tmp_path)
    receipt = service.read("owner", "original-request")
    receipt[NON_TEXT_RESULT_FIELD] = {"asset_pointer": "file-service://private_asset"}
    receipt["result"]["download_url"] = "https://private.example/?secret=hidden"
    public = project_public_chat_receipt(receipt)
    assert public["result"] == {"type": "non_text", "artifact_type": "image", "artifact_count": 1}
    assert public["recovery"]["retryable"] is False
    for private in ("original-account", "original-chat", "image-tool", "file-service://", "private.example", "private_asset"):
        assert private not in json.dumps(public)
