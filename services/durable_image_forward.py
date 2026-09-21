"""Recovery of legacy image protocols inside their existing text receipt.

Slots are execution progress, not additional tasks or capacity reservations.
Only the original admission worker sends; the recovery lease only reads and
downloads. Completed slot files share the original private output directory.
"""
import base64
from dataclasses import asdict, replace
import json
import os
import uuid

from services.request_context import AdmissionLost, current_request, executing


PROTOCOLS = {"openai_v1_image_generations", "openai_v1_image_edit", "openai_v1_chat_complete", "openai_v1_response"}


class _RecoveryContext:
    """Attribute paced GETs to the existing lease; never authorize a POST."""
    def __init__(self, service, owner, receipt):
        self.service, self.owner, self.original = service, owner, receipt
        self.request_id = receipt["request_id"]
        self.phase = "read_image_request"

    def receipt(self):
        return self.original

    def before_send(self):
        raise AdmissionLost("image recovery cannot submit a model request")

    def release_turn(self):
        pass  # Only verified original-branch evidence releases its reservation.

    def log_fields(self):
        return {"model": self.original.get("model"), "operation": "image", "route": "chat",
                "retained_input_bytes": self.original.get("_input_bytes"), "recovery_phase": self.phase}

    def record_limit(self, evidence):
        self.service._update_recovery_claim(self.owner, self.original,
            rate_limit={**evidence, "recovery_phase": self.phase})

    def set_phase(self, phase):
        self.service._update_recovery_claim(self.owner, self.original, recovery_phase=phase)
        self.phase = phase


def supported(receipt):
    return (receipt.get("_route") == "chat" and receipt.get("_operation") == "image"
            and receipt.get("_forward_protocol") in PROTOCOLS and receipt.get("_input_ref")
            and isinstance(receipt.get("_image_recovery"), dict) and bool(receipt["_image_recovery"]))


def wire_identity():
    context = current_request.get()
    receipt = context.receipt() if context is not None else {}
    return receipt.get("_wire_identity", {}) if receipt.get("_operation") == "image" else {}


def prepare(context, request):
    receipt = context.receipt()
    if (context.kind != "text" or receipt.get("_route") != "chat"
            or receipt.get("_forward_protocol") not in PROTOCOLS or receipt.get("_operation") != "image"):
        return False
    if not receipt.get("_image_recovery"):
        from services.protocol.conversation import count_text_tokens
        from utils.image_tokens import count_image_content_tokens
        context.admission.update_claim(context, _image_recovery={
            "model": request.model, "n": request.n, "response_format": request.response_format,
            "base_url": request.base_url, "size": request.size, "quality": request.quality,
            "input_text_tokens": count_text_tokens(request.prompt, request.model),
            "input_image_tokens": count_image_content_tokens([
                {"type": "image_url", "image_url": {"url": value}} for value in request.images or []], request.model),
        })
    return True


def _slot(receipt, index):
    return dict((receipt.get("_image_slots") or {}).get(str(index)) or {})


def _slot_changes(receipt, index, **changes):
    return {"_image_slots": {**receipt.get("_image_slots", {}), str(index): {**_slot(receipt, index), **changes}}}


def _load_outputs(store, slot):
    from services.protocol.conversation import ImageOutput
    with store.output_file(slot["output_ref"]) as handle:
        return [ImageOutput(**value) for value in json.load(handle)]


def cached_slot(context, index):
    slot = _slot(context.receipt(), index)
    return _load_outputs(context.admission.store, slot) if slot.get("completed") else None


def slot_request(context, request, index):
    receipt = context.receipt()
    # A fresh admission claim resets _submission_started. The original slot's
    # persisted send boundary must still forbid replay after that reset.
    if "_last_sent_sequence" in receipt and index <= receipt["_last_sent_sequence"]:
        raise AdmissionLost("original image slot must be recovered before continuing")
    message_id = receipt["request_message_id"] if index == 0 else str(uuid.uuid5(
        uuid.NAMESPACE_URL, context.owner + ":" + context.request_id + ":" + str(index)))
    context.admission.update_claim(context, **_slot_changes(receipt, index, request_message_id=message_id))
    def update(**changes):
        context.admission.update_claim(context, **_slot_changes(context.receipt(), index, **changes))
    def progress(step):
        update(progress=step)
    def cursor(conversation_id):
        previous = _slot(context.receipt(), index).get("conversation_id")
        if previous and previous != conversation_id:
            raise AdmissionLost("original image conversation changed")
        if conversation_id:
            update(conversation_id=conversation_id)
    def result_ids(file_ids, sediment_ids):
        if file_ids or sediment_ids:
            update(file_ids=list(dict.fromkeys(file_ids)), sediment_ids=list(dict.fromkeys(sediment_ids)))
    progress.request_message_id = message_id
    progress.record_conversation_id = cursor
    progress.record_result_ids = result_ids
    return replace(request, progress_callback=progress)


def _save_outputs(store, outputs):
    from services.durable_forward import MAX_OUTPUT_BYTES
    completed = [asdict(item) for item in outputs if item.kind in {"result", "message"}]
    if not completed:
        raise RuntimeError("original image slot has no completed output")
    data = json.dumps(completed, ensure_ascii=False).encode()
    if len(data) > MAX_OUTPUT_BYTES:
        raise ValueError("private image output limit")
    ref = store.create_output()
    with store.output_file(ref, append=True) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return ref


def save_slot(context, index, outputs):
    ref = _save_outputs(context.admission.store, outputs)
    context.admission.update_claim(context, **_slot_changes(context.receipt(), index, output_ref=ref, completed=True),
                                   _completed_slot=index)


def _prompt(spec):
    protocol, body = spec["protocol"], spec["payload"]
    if protocol == "openai_v1_chat_complete":
        from services.protocol.openai_v1_chat_complete import chat_image_args
        return chat_image_args(body)[1]
    if protocol == "openai_v1_response":
        from services.protocol.openai_v1_response import extract_response_prompt
        return extract_response_prompt(body.get("input"))
    return str(body.get("prompt") or "")


def _format(service, receipt, spec):
    from services.protocol import openai_v1_chat_complete as chat, openai_v1_response as responses
    from services.protocol.conversation import collect_image_outputs, stream_image_chunks
    from services.durable_forward import publish_recovered_wire
    from utils.image_tokens import image_usage, count_image_output_items_tokens
    metadata, identity = receipt["_image_recovery"], receipt.get("_wire_identity") or {}
    outputs = [output for index in range(metadata["n"]) for output in _load_outputs(service.store, _slot(receipt, index))]
    result = collect_image_outputs(outputs)
    model, protocol = metadata["model"], receipt["_forward_protocol"]
    usage = image_usage(input_text_tokens=metadata["input_text_tokens"], input_image_tokens=metadata["input_image_tokens"],
                        output_tokens=count_image_output_items_tokens(result.get("data"), metadata["size"], metadata["quality"]))
    if protocol in {"openai_v1_image_generations", "openai_v1_image_edit"}:
        response, events = {**result, "usage": usage}, stream_image_chunks(outputs)
    elif protocol == "openai_v1_chat_complete":
        response = chat.completion_response(model, chat.image_result_content(result), identity.get("created") or result.get("created"))
        response["id"] = identity.get("id") or response["id"]
        response["usage"] = chat.chat_usage_from_image_usage(usage)
        events = chat.stream_image_chat_completion(outputs, model, identity=identity)
    else:
        events = list(responses.stream_image_response(outputs, _prompt(spec), model, metadata["input_image_tokens"],
                                                     metadata["size"], metadata["quality"], identity=identity))
        response = responses.collect_response(events)
    return publish_recovered_wire(service, receipt, response, events, bool(spec["payload"].get("stream")))


def _ended(document, conversation_id, message_id):
    from services.conversation_binding_service import ConversationBindingService, _completed_request_turn
    if ConversationBindingService._request_anchor_from_document(
            document, conversation_id, message_id, recovery_scan={}) is None:
        raise RuntimeError("original image user message is missing")
    mapping, children = document["mapping"], {}
    for node_id, node in mapping.items():
        if isinstance(node, dict) and node.get("parent"):
            children.setdefault(str(node["parent"]), []).append(str(node_id))
    return _completed_request_turn(mapping, children, message_id, conversation_id)


def _resolve_original_urls(backend, conversation_id, file_ids, sediment_ids):
    # The normal polling resolver tolerates partial URL failures. Recovery
    # must retain every saved original ID and propagate its 429/retry evidence.
    urls = []
    for source, ids in (("file", file_ids), ("sediment", sediment_ids)):
        for result_id in dict.fromkeys(ids):
            if result_id == "file_upload":
                continue
            url = (backend._get_file_download_url(result_id) if source == "file"
                   else backend._get_attachment_download_url(conversation_id, result_id))
            if not url:
                raise RuntimeError("original image download URL is unavailable")
            if url not in urls:
                urls.append(url)
    return urls


def _terminal_result_ids(backend, document, ended, message_id):
    records = backend._extract_image_tool_records({**document, "current_node": ended["final_message_id"]}, message_id)
    result = {key: list(dict.fromkeys(value for record in records for value in record[key]))
              for key in ("file_ids", "sediment_ids")}
    if not any(result.values()):
        raise RuntimeError("original image is still unresolved")
    return result


def recover(service, owner, receipt):
    context = _RecoveryContext(service, owner, receipt)
    with executing(context):
        context.set_phase("read_image_request")
        return _recover(service, owner, receipt, context)


def _recover(service, owner, receipt, context):
    """Read/download the submitted slot; requeue only its unsent successors."""
    from services.account_service import account_service
    from services.openai_backend_api import OpenAIBackendAPI
    from services.conversation_binding_service import ConversationBindingService
    from services.protocol.conversation import ImageOutput, format_image_result
    spec = service.store.load_input(receipt["_input_ref"])["_forward"]
    metadata = receipt["_image_recovery"]
    index = next((i for i in range(metadata["n"]) if not _slot(receipt, i).get("completed")), metadata["n"])
    def save_slot_progress(**changes):
        nonlocal receipt
        receipt = service._update_recovery_claim(owner, receipt, **_slot_changes(receipt, index, **changes))
    if index < metadata["n"] and index <= receipt.get("_last_sent_sequence", -1):
        slot = _slot(receipt, index)
        if not slot.get("request_message_id"):
            raise RuntimeError("original image slot identity is unavailable")
        binding = receipt["provider_binding_id"]
        if account_service.get_bound_account_identity(binding) != receipt["provider_account_identity"]:
            raise RuntimeError("original image account changed")
        token = account_service.get_bound_text_access_token(binding, model="auto")
        with account_service.conversation_binding_lock(binding, receipt["client_conversation_id"]):
            backend = OpenAIBackendAPI(token)
            try:
                conversation_id = slot.get("conversation_id")
                document = None
                def download_results(result_ids):
                    context.set_phase("resolve_image_download")
                    urls = _resolve_original_urls(backend, conversation_id, **result_ids)
                    context.set_phase("download_image")
                    downloaded = []
                    for url in urls:
                        # The existing downloader deduplicates equal bytes.
                        # First prove every URL succeeded, then deduplicate.
                        images = backend.download_image_bytes([url])
                        if len(images) != 1 or not images[0]:
                            raise RuntimeError("original image download is incomplete")
                        if images[0] not in downloaded:
                            downloaded.append(images[0])
                    if not downloaded:
                        raise RuntimeError("original image download is incomplete")
                    context.set_phase("save_image_result")
                    data = format_image_result([{"b64_json": base64.b64encode(value).decode("ascii")} for value in downloaded],
                                               _prompt(spec), metadata["response_format"], metadata["base_url"])["data"]
                    ref = _save_outputs(service.store, [ImageOutput(kind="result", model=metadata["model"], index=index + 1,
                                        total=metadata["n"], data=data, conversation_id=conversation_id)])
                    save_slot_progress(output_ref=ref, output_result_ids=result_ids)

                if not conversation_id:
                    located, document = ConversationBindingService._locate_text_request_conversation(backend, {
                        **receipt, "request_message_id": slot["request_message_id"], "conversation_id": "",
                        "request_parent_message_id": "", "parent_message_id": "",
                    })
                    conversation_id = located["conversation_id"]
                    save_slot_progress(conversation_id=conversation_id)
                file_ids, sediment_ids = slot.get("file_ids") or [], slot.get("sediment_ids") or []
                # Saved image IDs are download authority. Never rerun an image
                # tool or poll generation when those original IDs are present.
                if not file_ids and not sediment_ids:
                    document = document or backend._get_conversation(conversation_id)
                    ended = _ended(document, conversation_id, slot["request_message_id"])
                    if not ended:
                        raise RuntimeError("original image branch is not known to have ended")
                    receipt = service._update_recovery_claim(owner, receipt,
                        **_slot_changes(receipt, index, turn_end=ended), _upstream_terminal=True, _turn_reserved=False)
                    result_ids = _terminal_result_ids(backend, document, ended, slot["request_message_id"])
                    file_ids, sediment_ids = result_ids["file_ids"], result_ids["sediment_ids"]
                    save_slot_progress(file_ids=file_ids, sediment_ids=sediment_ids)
                if not slot.get("output_ref"):
                    download_results({"file_ids": file_ids, "sediment_ids": sediment_ids})
                # A saved image is not, by itself, proof that other work on the
                # original branch ended. Keep the downloaded file while waiting.
                context.set_phase("read_image_request")
                document = document or backend._get_conversation(conversation_id)
                ended = _ended(document, conversation_id, slot["request_message_id"])
                if not ended:
                    raise RuntimeError("original image branch is not known to have ended")
                complete_ids = _terminal_result_ids(backend, document, ended, slot["request_message_id"])
                # SSE IDs and cached output can precede additional results on
                # the same original branch. Keep the cached file and its exact
                # coverage until a complete replacement has been saved.
                coverage = _slot(receipt, index).get("output_result_ids") or {"file_ids": file_ids, "sediment_ids": sediment_ids}
                receipt = service._update_recovery_claim(owner, receipt,
                    **_slot_changes(receipt, index, **complete_ids, output_result_ids=coverage, turn_end=ended),
                    _upstream_terminal=True, _turn_reserved=False)
                if any(set(complete_ids[key]) != set(coverage[key]) for key in complete_ids):
                    download_results(complete_ids)
                receipt = service._update_recovery_claim(owner, receipt,
                    **_slot_changes(receipt, index, completed=True, turn_end=ended), _completed_slot=index,
                    _upstream_terminal=True, _turn_reserved=False)
                index += 1
            finally:
                backend.close()
    if index < metadata["n"]:
        # Every preceding submitted slot is complete and saved. The original
        # scheduler will choose this source/account again before any next send.
        return {"status": "queued", "_claim_id": None, "_executing": False, "_submission_started": False,
                "_turn_reserved": False, "_upstream_terminal": False, "_wire_head": None,
                "upstream_unfinished": False, "upstream_outcome": "generated", "_ready_at": service._now()}
    context.set_phase("format_image_response")
    return {"status": "succeeded", **_format(service, receipt, spec)}
