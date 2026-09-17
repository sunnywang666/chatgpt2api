"""Trusted request classification before any ordinary-key upstream admission."""
from fastapi import HTTPException

from services.auth_service import READY_CAPABILITIES
from services.program_key_policy import (
    Capability, PolicyError, ProgramKeyPolicy, RequestUse, Route,
    authorize_submission, codex_use, image_use,
)
from utils.helper import IMAGE_MODELS, is_codex_image_model

# The image adapter uses exactly this registry (including plan-prefixed aliases).
IMAGE_ROUTES = {model: Route.CODEX if is_codex_image_model(model) else Route.CHAT
                for model in IMAGE_MODELS}
# Native tools executed by the upstream Codex Responses adapter. Local function
# and custom tools are handled structurally, never inferred from their names.
CODEX_TOOLS = {
    "image_generation": frozenset({Capability.CODEX_IMAGE}),
    **{name: frozenset({Capability.CODEX_CODING}) for name in (
        "web_search", "web_search_preview", "web_search_preview_2025_03_11",
        "file_search", "code_interpreter", "computer", "computer_use_preview",
        "mcp", "shell", "local_shell", "apply_patch", "tool_search",
    )},
}


def _authorize(identity: dict, resolve) -> None:
    # Preserve the existing server-only management/Content path. Public ingress
    # independently rejects admin keys; ordinary users cannot choose this role.
    if identity.get("role") == "admin":
        return
    try:
        policy = ProgramKeyPolicy.from_record(identity.get("policy"))
        authorize_submission(policy, key_enabled=identity.get("enabled", True),
                             use=resolve(), ready=READY_CAPABILITIES)
    except PolicyError as exc:
        raise HTTPException(403, detail={"code": exc.code}) from None


def require_image_policy(identity: dict, model: object) -> None:
    _authorize(identity, lambda: image_use(str(model or "gpt-image-2").strip().lower(),
                                         model_routes=IMAGE_ROUTES))


def require_codex_policy(identity: dict, payload: dict) -> None:
    def resolve():
        use = codex_use(payload, native_tool_capabilities=CODEX_TOOLS)
        if str(payload.get("model") or "").strip().lower() in IMAGE_ROUTES:
            return RequestUse(Route.CODEX, use.capabilities | {Capability.CODEX_IMAGE})
        return use
    _authorize(identity, resolve)


def require_chat_text_policy(identity: dict, *, endpoint: str = "", model: object = "") -> None:
    # Retain only observed pre-policy compatibility for explicitly reconciled
    # keys. Updating the key policy removes this non-transferable exception.
    if identity.get("enabled", True) and identity.get("policy") is not None:
        for entry in identity.get("legacy_text_compatibility", []):
            if entry["endpoint"] == endpoint and str(model or "auto").strip() in entry["models"]:
                return
    _authorize(identity, lambda: RequestUse(Route.CHAT, frozenset({Capability.CHAT_TEXT})))
