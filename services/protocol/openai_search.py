from __future__ import annotations

from services.account_service import account_service
from services.openai_backend_api import OpenAIBackendAPI, SEARCH_MODEL

MODEL = SEARCH_MODEL


def handle(body: dict[str, object]) -> dict[str, object]:
    from services.durable_forward import prepare_chat_recovery
    from services.request_context import current_request
    context = current_request.get()
    if context is not None:
        prepare_chat_recovery(context, MODEL, [{"role": "user", "content": str(body["prompt"])}], search=True)
    token = account_service.get_text_access_token()
    account = account_service.get_account(token) or {}
    backend = OpenAIBackendAPI(token)
    try:
        result = backend.search(str(body["prompt"]))
    finally:
        backend.close()
    account_service.mark_text_used(token)
    result["_account_email"] = str(account.get("email") or "")
    return result
