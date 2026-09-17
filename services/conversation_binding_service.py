from __future__ import annotations

from enum import Enum
from typing import Any

from services.account_service import account_service
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol.conversation import conversation_events
from utils.helper import UpstreamHTTPError


class TextRecoveryReason(str, Enum):
    ACCOUNT_IDENTITY_MISMATCH = "ACCOUNT_IDENTITY_MISMATCH"
    CONVERSATION_ID_MISMATCH = "CONVERSATION_ID_MISMATCH"
    REQUEST_MESSAGE_NOT_FOUND = "REQUEST_MESSAGE_NOT_FOUND"
    REQUEST_PARENT_MISMATCH = "REQUEST_PARENT_MISMATCH"
    REQUEST_BRANCH_AMBIGUOUS = "REQUEST_BRANCH_AMBIGUOUS"
    REQUEST_BRANCH_SUPERSEDED = "REQUEST_BRANCH_SUPERSEDED"
    REQUEST_RESULT_INCOMPLETE = "REQUEST_RESULT_INCOMPLETE"
    REQUEST_RESULT_NOT_FOUND = "REQUEST_RESULT_NOT_FOUND"
    REQUEST_RESULT_TERMINAL_EMPTY = "REQUEST_RESULT_TERMINAL_EMPTY"
    REQUEST_CONVERSATION_UNATTRIBUTABLE = "REQUEST_CONVERSATION_UNATTRIBUTABLE"
    CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"


_ACTIVE_TEXT_RESULT_STATUSES = frozenset({"in_progress", "running", "pending", "queued"})


class ConversationBindingError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "CONVERSATION_BINDING_UNAVAILABLE",
        provider_binding_id: str = "",
        provider_account_identity: str = "",
        conversation_id: str = "",
        parent_message_id: str = "",
        recovery_reason: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.provider_binding_id = provider_binding_id
        self.provider_account_identity = provider_account_identity
        self.conversation_id = conversation_id
        self.parent_message_id = parent_message_id
        self.recovery_reason = recovery_reason


class ConversationBindingService:
    RECOVERY_RECENT_CONVERSATION_LIMIT = 20

    def archive(self, body: dict[str, Any]) -> dict[str, Any]:
        binding = body["provider_binding_id"]
        if account_service.get_bound_account_identity(binding) != body["provider_account_identity"]:
            raise ConversationBindingError("provider account identity changed", code="CONVERSATION_BINDING_MISMATCH")
        token = account_service.get_bound_text_access_token(binding, model="auto")
        with account_service.conversation_binding_lock(binding, body["client_conversation_id"]):
            backend = OpenAIBackendAPI(access_token=token)
            try:
                result = backend.archive_conversation(body["conversation_id"], body["parent_message_id"])
                return {**body, **result}
            finally:
                backend.close()

    def read_text_request(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Recover only the answer descending from this request's own user turn."""
        binding = receipt["provider_binding_id"]
        if account_service.get_bound_account_identity(binding) != receipt["provider_account_identity"]:
            raise ConversationBindingError(
                "provider account identity changed",
                code="CONVERSATION_BINDING_MISMATCH",
                recovery_reason=TextRecoveryReason.ACCOUNT_IDENTITY_MISMATCH.value,
            )
        token = account_service.get_bound_text_access_token(binding, model="auto")
        with account_service.conversation_binding_lock(binding, receipt["client_conversation_id"]):
            backend = OpenAIBackendAPI(access_token=token)
            try:
                if not str(receipt.get("conversation_id") or "").strip():
                    located_receipt = self._locate_text_request_conversation(
                        backend, receipt,
                    )
                    return self._read_text_request_result(backend, located_receipt)
                try:
                    return self._read_text_request_result(backend, receipt)
                except UpstreamHTTPError as exc:
                    if exc.status_code != 404:
                        raise
                    raise ConversationBindingError(
                        "original conversation is missing",
                        code="CONVERSATION_OUTCOME_UNKNOWN",
                        conversation_id=str(receipt.get("conversation_id") or ""),
                        recovery_reason=TextRecoveryReason.CONVERSATION_NOT_FOUND.value,
                    ) from exc
            finally:
                backend.close()

    @classmethod
    def _locate_text_request_conversation(
        cls,
        backend: OpenAIBackendAPI,
        receipt: dict[str, Any],
    ) -> dict[str, Any]:
        """Find one exact request user node within a bounded recent-account scan."""
        request_message_id = str(receipt.get("request_message_id") or "").strip()
        if not request_message_id:
            raise ConversationBindingError(
                "original request message identity is required",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )
        recent = backend._list_recent_conversations(
            limit=cls.RECOVERY_RECENT_CONVERSATION_LIMIT,
            timeout_secs=10.0,
            strict_schema=True,
        )
        matches: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in recent:
            conversation_id = str(
                item.get("id") or item.get("conversation_id") or ""
            ).strip()
            if conversation_id in seen:
                continue
            seen.add(conversation_id)
            document = backend._get_conversation(conversation_id)
            if not isinstance(document, dict):
                raise ConversationBindingError(
                    "conversation document is invalid",
                    code="CONVERSATION_BINDING_CONTRACT_INVALID",
                )
            returned_conversation_id = str(
                document.get("conversation_id") or conversation_id
            ).strip()
            if returned_conversation_id != conversation_id:
                raise ConversationBindingError(
                    "conversation identity changed during request recovery",
                    code="CONVERSATION_BINDING_CONTRACT_INVALID",
                )
            mapping = document.get("mapping")
            if not isinstance(mapping, dict):
                raise ConversationBindingError(
                    "conversation mapping is missing or invalid",
                    code="CONVERSATION_BINDING_CONTRACT_INVALID",
                )
            request_node = mapping.get(request_message_id)
            if request_node is None:
                continue
            request_message = (
                request_node.get("message")
                if isinstance(request_node, dict) else None
            )
            request_author = (
                request_message.get("author")
                if isinstance(request_message, dict) else None
            )
            if (
                not isinstance(request_node, dict)
                or not isinstance(request_message, dict)
                or request_message.get("id") != request_message_id
                or not isinstance(request_author, dict)
                or request_author.get("role") != "user"
            ):
                raise ConversationBindingError(
                    "request message anchor is invalid",
                    code="CONVERSATION_BINDING_CONTRACT_INVALID",
                )
            matches.append((conversation_id, str(request_node.get("parent") or "").strip()))
            if len(matches) > 1:
                break

        if len(matches) != 1:
            raise ConversationBindingError(
                "original request conversation cannot be attributed uniquely",
                code="CONVERSATION_OUTCOME_UNKNOWN",
                recovery_reason=(
                    TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value
                ),
            )
        conversation_id, request_parent_message_id = matches[0]
        expected_parent_message_id = str(
            receipt.get(
                "request_parent_message_id",
                receipt.get("parent_message_id"),
            ) or ""
        ).strip()
        if (
            expected_parent_message_id
            and request_parent_message_id != expected_parent_message_id
        ):
            raise ConversationBindingError(
                "original request parent changed",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=conversation_id,
                parent_message_id=request_parent_message_id,
                recovery_reason=TextRecoveryReason.REQUEST_PARENT_MISMATCH.value,
            )
        return {
            **receipt,
            "conversation_id": conversation_id,
            # Both anchors are exact nodes from the recovered mapping. The
            # request parent validates branch identity; the request node is a
            # usable continuation cursor until a completed answer replaces it.
            "parent_message_id": request_message_id,
            **(
                {"request_parent_message_id": request_parent_message_id}
                if request_parent_message_id else {}
            ),
        }

    def read_text(self, body: dict[str, Any]) -> dict[str, Any]:
        """Read an already-issued cursor on its bound account; never send a message."""
        keys = ("provider_binding_id", "provider_account_identity", "client_conversation_id",
                "conversation_id", "parent_message_id")
        if any(not isinstance(body.get(key), str) or not body[key].strip() for key in keys):
            raise ConversationBindingError("original text cursor is required", code="CONVERSATION_BINDING_CONTRACT_INVALID")
        binding_id = body["provider_binding_id"]
        if account_service.get_bound_account_identity(binding_id) != body["provider_account_identity"]:
            raise ConversationBindingError("provider account identity changed", code="CONVERSATION_BINDING_MISMATCH")
        token = account_service.get_bound_text_access_token(binding_id, model="auto")
        with account_service.conversation_binding_lock(binding_id, body["client_conversation_id"]):
            backend = OpenAIBackendAPI(access_token=token)
            try:
                return self._read_text_result(backend, body)
            finally:
                backend.close()

    @staticmethod
    def _read_text_result(backend: OpenAIBackendAPI, cursor: dict[str, Any]) -> dict[str, Any]:
        document = backend._get_conversation(cursor["conversation_id"])
        if document.get("conversation_id", cursor["conversation_id"]) != cursor["conversation_id"]:
            raise ConversationBindingError("conversation identity changed", code="CONVERSATION_BINDING_MISMATCH")
        mapping = document.get("mapping") or {}
        current = str(document.get("current_node") or "")
        node_id = current
        visited: set[str] = set()
        while node_id and node_id not in visited:
            visited.add(node_id)
            node = mapping.get(node_id) or {}
            if node_id == cursor["parent_message_id"]:
                break
            # A later user turn is not the result of the original request.
            if (node.get("message") or {}).get("author", {}).get("role") == "user":
                raise ConversationBindingError("original turn was superseded", code="CONVERSATION_BINDING_MISMATCH")
            node_id = str(node.get("parent") or "")
        if node_id != cursor["parent_message_id"]:
            raise ConversationBindingError("original turn is not on the active branch", code="CONVERSATION_BINDING_MISMATCH")
        message = (mapping.get(current) or {}).get("message") or {}
        result = {key: cursor[key] for key in ("provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id", "parent_message_id")}
        if (message.get("id") != current or message.get("author", {}).get("role") != "assistant"
                or message.get("status") != "finished_successfully" or message.get("end_turn") is not True
                or message.get("channel") not in (None, "final")):
            return {**result, "binding_status": "unknown", "status": "running"}
        content = message.get("content") or {}
        parts = content.get("parts")
        if content.get("content_type") != "text" or not isinstance(parts, list) or not parts or not all(isinstance(part, str) for part in parts):
            return {**result, "binding_status": "unknown", "status": "running"}
        text = "".join(parts).strip()
        if not text:
            return {**result, "binding_status": "unknown", "status": "running"}
        return {**result, "binding_status": "bound", "status": "succeeded", "parent_message_id": current, "content": text}

    @staticmethod
    def _read_text_request_result(backend: OpenAIBackendAPI, receipt: dict[str, Any]) -> dict[str, Any]:
        """Read the exact request branch without requiring it to be current.

        A later user turn can make the original request no longer reachable
        from ``current_node`` even though its completed answer is still in the
        conversation mapping. This reader starts at the persisted request user
        node, walks only assistant/tool descendants, and stops at later user
        nodes. It never changes the receipt's binding or cursor.
        """
        conversation_id = str(receipt.get("conversation_id") or "").strip()
        request_message_id = str(receipt.get("request_message_id") or "").strip()
        document = backend._get_conversation(conversation_id)
        returned_conversation_id = str(document.get("conversation_id") or conversation_id).strip()
        if returned_conversation_id != conversation_id:
            raise ConversationBindingError(
                "conversation identity changed",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=returned_conversation_id,
                recovery_reason=TextRecoveryReason.CONVERSATION_ID_MISMATCH.value,
            )

        if "mapping" not in document or not isinstance(document.get("mapping"), dict):
            raise ConversationBindingError(
                "conversation mapping is missing or invalid",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
                conversation_id=conversation_id,
            )
        mapping = document["mapping"]
        request_node = mapping.get(request_message_id)
        result = {key: receipt[key] for key in (
            "provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id",
            "parent_message_id", "request_parent_message_id",
        ) if key in receipt and (
            key not in {"parent_message_id", "request_parent_message_id"}
            or str(receipt[key] or "").strip()
        )}
        if request_message_id and request_node is None:
            current_node_id = str(document.get("current_node") or "").strip()
            current_node = mapping.get(current_node_id)
            current_message = current_node.get("message") if isinstance(current_node, dict) else None
            current_author = current_message.get("author") if isinstance(current_message, dict) else None
            current_role = str(current_author.get("role") or "").strip().lower() \
                if isinstance(current_author, dict) else ""
            current_status = str(current_message.get("status") or "").strip().lower() \
                if isinstance(current_message, dict) else ""
            if current_role in {"assistant", "tool"} and current_status in _ACTIVE_TEXT_RESULT_STATUSES:
                return {
                    **result,
                    "binding_status": "unknown",
                    "status": "running",
                    "recovery_reason": TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value,
                }
        request_message = request_node.get("message") if isinstance(request_node, dict) else None
        request_author = request_message.get("author") if isinstance(request_message, dict) else None
        request_role = request_author.get("role") if isinstance(request_author, dict) else None
        if (not request_message_id or not isinstance(request_node, dict)
                or not isinstance(request_message, dict)
                or request_message.get("id") != request_message_id
                or request_role != "user"):
            raise ConversationBindingError(
                "original request user turn is missing",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=conversation_id,
                recovery_reason=TextRecoveryReason.REQUEST_MESSAGE_NOT_FOUND.value,
            )

        expected_parent = str(receipt.get("request_parent_message_id", receipt.get("parent_message_id")) or "").strip()
        request_parent = str(request_node.get("parent") or "").strip()
        if expected_parent and request_parent != expected_parent:
            raise ConversationBindingError(
                "original request parent changed",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=conversation_id,
                parent_message_id=request_parent,
                recovery_reason=TextRecoveryReason.REQUEST_PARENT_MISMATCH.value,
            )

        children: dict[str, list[str]] = {}
        for raw_node_id, raw_node in mapping.items():
            if not isinstance(raw_node, dict):
                continue
            node_id = str(raw_node_id)
            parent = str(raw_node.get("parent") or "").strip()
            if parent:
                children.setdefault(parent, []).append(node_id)

        candidates: list[tuple[str, str]] = []
        visited: set[str] = set()
        later_user_seen = False
        active_result_seen = False
        terminal_empty_seen = False
        pending = list(children.get(request_message_id, []))
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = mapping.get(node_id)
            if not isinstance(node, dict):
                continue
            message = node.get("message") or {}
            if not isinstance(message, dict):
                message = {}
            author = message.get("author") or {}
            role = str(author.get("role") or "").strip().lower() if isinstance(author, dict) else ""
            if role == "user":
                later_user_seen = True
                continue
            if role and role not in {"assistant", "tool"}:
                continue
            status = str(message.get("status") or "").strip().lower()
            if status in _ACTIVE_TEXT_RESULT_STATUSES:
                active_result_seen = True
            if role == "assistant":
                content = message.get("content") or {}
                parts = content.get("parts") if isinstance(content, dict) else None
                terminal = status == "finished_successfully" and message.get("end_turn") is True
                text = "".join(parts).strip() if (
                    message.get("id") == node_id
                    and terminal
                    and message.get("channel") in (None, "final")
                    and isinstance(parts, list)
                    and parts
                    and all(isinstance(part, str) for part in parts)
                    and content.get("content_type") == "text"
                ) else ""
                if text:
                    candidates.append((node_id, text))
                elif terminal:
                    terminal_empty_seen = True
            pending.extend(children.get(node_id, []))

        if len(candidates) > 1:
            raise ConversationBindingError(
                "multiple completed answers descend from the original request",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=conversation_id,
                recovery_reason=TextRecoveryReason.REQUEST_BRANCH_AMBIGUOUS.value,
            )
        if not candidates:
            if active_result_seen:
                recovery_reason = TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value
            elif terminal_empty_seen:
                recovery_reason = TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value
            elif later_user_seen:
                recovery_reason = TextRecoveryReason.REQUEST_BRANCH_SUPERSEDED.value
            else:
                recovery_reason = TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value
            return {
                **result,
                "binding_status": "unknown",
                "status": "running" if active_result_seen else "unknown",
                "recovery_reason": recovery_reason,
            }
        parent_message_id, text = candidates[0]
        return {
            **result,
            "binding_status": "bound",
            "status": "succeeded",
            "parent_message_id": parent_message_id,
            "content": text,
        }

    def complete_text(self, body: dict[str, Any], *, on_cursor=None) -> dict[str, Any]:
        binding_id = str(body.get("provider_binding_id") or "").strip()
        account_identity = str(body.get("provider_account_identity") or "").strip()
        client_conversation_id = str(body.get("client_conversation_id") or "").strip()
        conversation_id = str(body.get("conversation_id") or "").strip()
        parent_message_id = str(body.get("parent_message_id") or "").strip()
        model = str(body.get("model") or "auto").strip() or "auto"
        image_model = str(body.get("image_model") or "gpt-image-2").strip() or "gpt-image-2"
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ConversationBindingError(
                "conversation messages are required",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )
        if not client_conversation_id:
            raise ConversationBindingError(
                "client_conversation_id is required",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )
        if binding_id:
            if not account_identity:
                raise ConversationBindingError(
                    "provider account identity is required for a bound account",
                    code="CONVERSATION_BINDING_CONTRACT_INVALID",
                )
            if bool(conversation_id) != bool(parent_message_id):
                raise ConversationBindingError(
                    "conversation continuation requires conversation_id and parent_message_id",
                    code="CONVERSATION_BINDING_CONTRACT_INVALID",
                )
            try:
                authoritative_identity = account_service.get_bound_account_identity(binding_id)
            except RuntimeError as exc:
                raise ConversationBindingError(str(exc)) from exc
            if authoritative_identity != account_identity:
                raise ConversationBindingError(
                    "provider account identity changed",
                    code="CONVERSATION_BINDING_MISMATCH",
                )
        elif conversation_id or parent_message_id:
            raise ConversationBindingError(
                "upstream cursor requires provider_binding_id",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )
        else:
            try:
                binding_id, account_identity, image_token = account_service.create_conversation_binding(
                    image_model=image_model, text_model=model
                )
                account_service.release_image_slot(image_token)
            except RuntimeError as exc:
                raise ConversationBindingError(str(exc)) from exc

        if on_cursor:
            on_cursor({"provider_binding_id": binding_id, "provider_account_identity": account_identity,
                       "client_conversation_id": client_conversation_id,
                       **({"conversation_id": conversation_id, "parent_message_id": parent_message_id} if conversation_id else {})})
        try:
            access_token = account_service.get_bound_text_access_token(
                binding_id,
                model=model,
                for_message=True,
            )
        except RuntimeError as exc:
            raise ConversationBindingError(str(exc)) from exc

        with account_service.conversation_binding_lock(binding_id, client_conversation_id):
            backend = OpenAIBackendAPI(access_token=access_token)
            backend.retain_bound_conversation = True
            backend.text_request_message_id = str(body.get("_request_message_id") or "")
            try:
                parts: list[str] = []
                returned_conversation_id = ""
                for event in conversation_events(
                    backend,
                    messages=messages,
                    model=model,
                    thinking_effort=str(body.get("thinking_effort") or ""),
                    conversation_id=conversation_id,
                    parent_message_id=parent_message_id,
                ):
                    old_conversation_id = returned_conversation_id
                    returned_conversation_id = str(
                        event.get("conversation_id") or returned_conversation_id
                    )
                    if on_cursor and returned_conversation_id and returned_conversation_id != old_conversation_id:
                        on_cursor({"conversation_id": returned_conversation_id})
                    if event.get("type") == "conversation.delta":
                        delta = str(event.get("delta") or "")
                        if delta:
                            parts.append(delta)
                if not returned_conversation_id:
                    raise ConversationBindingError(
                        "upstream response has no conversation_id",
                        code="CONVERSATION_OUTCOME_UNKNOWN",
                        provider_binding_id=binding_id,
                        provider_account_identity=account_identity,
                    )
                if conversation_id and returned_conversation_id != conversation_id:
                    raise ConversationBindingError(
                        "upstream conversation identity changed",
                        code="CONVERSATION_BINDING_MISMATCH",
                        provider_binding_id=binding_id,
                        provider_account_identity=account_identity,
                        conversation_id=returned_conversation_id,
                    )
                content = "".join(parts).strip()
                if not content:
                    raise ConversationBindingError(
                        "upstream response was empty",
                        code="CONVERSATION_OUTCOME_UNKNOWN",
                        provider_binding_id=binding_id,
                        provider_account_identity=account_identity,
                        conversation_id=returned_conversation_id,
                    )
                next_parent_message_id = backend.get_conversation_parent_message_id(
                    returned_conversation_id
                )
                account_service.mark_text_used(access_token)
                return {
                    "content": content,
                    "provider_binding_id": binding_id,
                    "provider_account_identity": account_identity,
                    "conversation_id": returned_conversation_id,
                    "parent_message_id": next_parent_message_id,
                    "binding_status": "bound",
                }
            except ConversationBindingError:
                raise
            except Exception as exc:
                recovered_parent = ""
                if returned_conversation_id:
                    try:
                        recovered_parent = backend.get_conversation_parent_message_id(
                            returned_conversation_id
                        )
                        # A stream timeout may happen after the answer was saved.
                        # Read that turn once, never regenerate it or accept an old continuation answer.
                        if recovered_parent and not conversation_id:
                            recovered = self._read_text_result(backend, {
                                "provider_binding_id": binding_id,
                                "provider_account_identity": account_identity,
                                "client_conversation_id": client_conversation_id,
                                "conversation_id": returned_conversation_id,
                                "parent_message_id": recovered_parent,
                            })
                            if recovered.get("status") == "succeeded":
                                return recovered
                    except Exception:
                        pass
                raise ConversationBindingError(
                    str(exc) or "upstream conversation outcome is unknown",
                    code="CONVERSATION_OUTCOME_UNKNOWN",
                    provider_binding_id=binding_id,
                    provider_account_identity=account_identity,
                    conversation_id=returned_conversation_id,
                    parent_message_id=recovered_parent,
                ) from exc
            finally:
                backend.close()


conversation_binding_service = ConversationBindingService()
