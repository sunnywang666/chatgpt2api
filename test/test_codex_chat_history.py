"""Focused fixtures for the local-only Codex Chat history bridge."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from services.codex_chat_history import load_bound_chat_history
from services.conversation_binding_service import ConversationBindingError
from services.task_store import TaskStore
from services.text_task_service import TextTaskService

OWNER = "owner"
ORIGINAL = "00000000-0000-0000-0000-000000000004"
ROOT = "00000000-0000-0000-0000-000000000001"
MIDDLE = "00000000-0000-0000-0000-000000000002"


def _body(*, messages, parent="", bound=True, forward=False, product=False):
    body = {"client_request_id": "", "client_conversation_id": "client", "messages": messages}
    if bound:
        body.update(provider_binding_id="binding", provider_account_identity="account",
                    conversation_id="conversation", parent_message_id=parent)
    if forward:
        body["_forward"] = {"protocol": "openai"}
    if product:
        body["product_work"] = {"product_ref": "product", "step_ref": "step"}
    return body


def _receipt(*, ref, parent, status="succeeded", outcome="completed", account="account"):
    result = {"_input_ref": ref, "status": status, "upstream_outcome": outcome,
              "provider_binding_id": "binding", "provider_account_identity": account,
              "client_conversation_id": "client", "conversation_id": "conversation",
              "parent_message_id": parent, "_route": "chat", "_operation": "text"}
    if status == "succeeded":
        result["content"] = "SENTINEL-ASSISTANT"
    return result


class HistoryFixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tasks.sqlite3"
        self.store = TaskStore(self.path)
        self.store.initialize()

    def close(self):
        self.tmp.cleanup()

    def insert(self, request_id, body, receipt):
        body = {**body, "client_request_id": request_id}
        ref = self.store.save_input(body)
        receipt = {**receipt, "_input_ref": ref}
        request_hash = TextTaskService._submission_identity(OWNER, body)[1]
        with self.store.transaction() as db:
            db.execute("INSERT INTO requests VALUES(?,?,?,?)",
                       (OWNER, request_id, request_hash, json.dumps(receipt)))
        return body, receipt

    def history(self):
        with self.store.connect() as db:
            return load_bound_chat_history(self.store, db, OWNER, ORIGINAL)


@pytest.fixture
def fixture():
    item = HistoryFixture()
    try:
        yield item
    finally:
        item.close()


def _complete_chain(fixture):
    fixture.insert(
        ROOT,
        _body(bound=False, messages=[{"role": "user", "content": [
            {"type": "text", "text": "SENTINEL-ROOT"},
            {"type": "input_image", "image_url": "data:image/png;base64,ROOT", "detail": "high"},
        ]}]),
        _receipt(ref="", parent="root-parent"),
    )
    fixture.insert(
        MIDDLE,
        _body(parent="root-parent", messages=[{"role": "user", "content": [
            {"type": "text", "text": "SENTINEL-MIDDLE"},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64,MIDDLE", "detail": "auto",
            }},
        ]}]),
        _receipt(ref="", parent="middle-parent"),
    )
    fixture.insert(
        ORIGINAL,
        _body(parent="middle-parent", messages=[{"role": "user", "content": [
            {"type": "text", "text": "SENTINEL-ORIGINAL"},
            {"type": "image_url", "image_url": {"url": "https://sentinel.invalid/original", "detail": "low"}},
        ]}]),
        _receipt(ref="", parent="", status="failed", outcome="rejected"),
    )


def _unavailable(fixture):
    with pytest.raises(ConversationBindingError) as error:
        fixture.history()
    assert error.value.code == "CODEX_FALLBACK_CONTEXT_UNAVAILABLE"
    assert "SENTINEL" not in str(error.value)


def test_rebuilds_complete_local_chain_root_to_target_and_preserves_image_detail(fixture):
    _complete_chain(fixture)

    messages = fixture.history()

    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"][1]["detail"] == "high"
    assert messages[2]["content"][1]["image_url"]["detail"] == "auto"
    assert messages[-1]["content"][1]["image_url"]["detail"] == "low"
    assert messages[1]["content"] == "SENTINEL-ASSISTANT"
    assert messages[3]["content"] == "SENTINEL-ASSISTANT"


@pytest.mark.parametrize("mutator", [
    "missing_input", "multiple_parent", "identity_mismatch", "hash_mismatch",
    "forward", "tool_message", "generated_image_output", "scope_whitespace", "cycle",
])
def test_rejects_incomplete_or_non_text_local_history(fixture, mutator):
    _complete_chain(fixture)

    if mutator == "missing_input":
        with fixture.store.connect() as db:
            receipt = json.loads(db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?",
                                            (OWNER, MIDDLE)).fetchone()[0])
        (fixture.store.input_dir / receipt["_input_ref"]).unlink()
    elif mutator == "multiple_parent":
        fixture.insert("00000000-0000-0000-0000-000000000003",
                       _body(parent="middle-parent", messages=[{"role": "user", "content": "duplicate"}]),
                       _receipt(ref="", parent="middle-parent"))
    elif mutator == "identity_mismatch":
        with fixture.store.transaction() as db:
            raw = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (OWNER, MIDDLE)).fetchone()[0]
            receipt = json.loads(raw); receipt["provider_account_identity"] = "other"
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(receipt), OWNER, MIDDLE))
    elif mutator == "hash_mismatch":
        with fixture.store.transaction() as db:
            db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", ("bad", OWNER, MIDDLE))
    else:
        target = ROOT
        with fixture.store.connect() as db:
            receipt = json.loads(db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (OWNER, target)).fetchone()[0])
            body = fixture.store.load_input(receipt["_input_ref"])
        if mutator == "forward":
            body["_forward"] = {"protocol": "openai"}
        elif mutator == "tool_message":
            body["messages"] = [{"role": "tool", "content": "tool output"}]
        elif mutator == "generated_image_output":
            receipt["_operation"] = "image"
        elif mutator == "scope_whitespace":
            receipt["provider_account_identity"] = " account "
        elif mutator == "cycle":
            body.update(provider_binding_id="binding", provider_account_identity="account",
                        conversation_id="conversation", parent_message_id="middle-parent")
            receipt["parent_message_id"] = "root-parent"
        ref = fixture.store.save_input({**body, "client_request_id": target})
        body["client_request_id"] = target
        receipt["_input_ref"] = ref
        with fixture.store.transaction() as db:
            db.execute("UPDATE requests SET request_hash=?,receipt=? WHERE owner=? AND id=?", (
                TextTaskService._submission_identity(OWNER, body)[1], json.dumps(receipt), OWNER, target
            ))

    _unavailable(fixture)


def test_history_reader_does_not_modify_existing_receipts_or_inputs(fixture):
    _complete_chain(fixture)
    before = fixture.path.read_bytes()
    before_inputs = sorted((path.name, path.read_bytes()) for path in fixture.store.input_dir.iterdir())

    fixture.history()

    assert fixture.path.read_bytes() == before
    assert sorted((path.name, path.read_bytes()) for path in fixture.store.input_dir.iterdir()) == before_inputs


def test_bound_product_step_history_remains_context_without_copying_membership(fixture):
    _complete_chain(fixture)
    with fixture.store.transaction() as db:
        raw = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (OWNER, MIDDLE)).fetchone()[0]
        receipt = json.loads(raw)
        body = fixture.store.load_input(receipt["_input_ref"])
        body["product_work"] = {"product_ref": "retained-product", "step_ref": "prior-vision"}
        receipt["_product_work"] = body["product_work"]
        receipt["_input_ref"] = fixture.store.save_input(body)
        db.execute("UPDATE requests SET request_hash=?,receipt=? WHERE owner=? AND id=?", (
            TextTaskService._submission_identity(OWNER, body)[1], json.dumps(receipt), OWNER, MIDDLE))
    messages = fixture.history()
    assert len(messages) == 5
    assert all("product_work" not in message for message in messages)
