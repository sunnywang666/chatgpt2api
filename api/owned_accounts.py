"""Internal Workbench management bridge; never exposed by public AI ingress."""
from fastapi import APIRouter, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from typing import Literal

from api.support import require_admin
from services.account_service import CodexAuthorizationAttachError, account_service
from services.auth_service import auth_service
from services.chat_login_service import ChatLoginError, chat_login_service
from services.codex_login_service import CodexLoginError, codex_login_service
from services.program_key_policy import PolicyError


class ImportAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    access_token: SecretStr
    refresh_token: SecretStr | None = None
    id_token: SecretStr | None = None
    source_type: str = "web"
    account_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,200}$")
    account_ref: str | None = Field(default=None, pattern=r"^car_[A-Za-z0-9_-]{43}$")


class EnabledAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


class AccountLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(max_length=80)


class AccountRefresh(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routes: list[Literal["chat", "codex"]] | None = Field(default=None, min_length=1, max_length=2)
    stale_only: bool = False


class ResourceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=0)
    image_account_concurrency: int = Field(ge=1, le=16)
    codex_max_concurrency: int = Field(ge=1, le=32)


class CodexAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid")
    access_token: SecretStr = Field(min_length=1, max_length=20_000)
    refresh_token: SecretStr = Field(min_length=1, max_length=20_000)
    id_token: SecretStr = Field(min_length=1, max_length=20_000)
    account_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,200}$")
    account_ref: str | None = Field(default=None, pattern=r"^car_[A-Za-z0-9_-]{43}$")


class CodexObservationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_ref: str = Field(pattern=r"^car_[A-Za-z0-9_-]{43}$")


class CodexLoginStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_request_id: str = Field(min_length=36, max_length=36)
    mode: Literal["import", "attach"]
    account_ref: str | None = Field(default=None, pattern=r"^car_[A-Za-z0-9_-]{43}$")


class ChatLoginStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_request_id: str = Field(min_length=36, max_length=36)
    mode: Literal["import", "attach"]
    account_ref: str | None = Field(default=None, pattern=r"^car_[A-Za-z0-9_-]{43}$")


class ChatLoginCallback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    callback_url: SecretStr = Field(min_length=1, max_length=30_000)


class KeyName(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    routes: list[str] = Field(default_factory=lambda: ["chat"])


class KeyPolicyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routes: list[str]
    expected_revision: int = Field(ge=0, strict=True)


def owner_scope(authorization: str | None, owner: str | None) -> str:
    require_admin(authorization)
    if not owner or not owner.startswith("workbench:") or len(owner) > 300:
        raise HTTPException(400, detail={"error": "trusted Workbench owner is required"})
    return owner


async def account_operation(handler, *args):
    try:
        return await run_in_threadpool(handler, *args)
    except KeyError:
        raise HTTPException(404, detail={"error": "account not found"}) from None
    except CodexAuthorizationAttachError as exc:
        status = 404 if exc.code == "codex_authorization_account_not_found" else 409
        raise HTTPException(status, detail={"code": exc.code}) from None
    except ValueError:
        raise HTTPException(409, detail={"error": "account import conflicts with existing account scope or has invalid input"}) from None


async def login_operation(handler, *args):
    try:
        return await run_in_threadpool(handler, *args)
    except CodexLoginError as exc:
        raise HTTPException(exc.status_code, detail={"code": exc.code}) from None
    except CodexAuthorizationAttachError as exc:
        status = 404 if exc.code == "codex_authorization_account_not_found" else 409
        raise HTTPException(status, detail={"code": exc.code}) from None


async def chat_login_operation(handler, *args):
    try:
        return await run_in_threadpool(handler, *args)
    except ChatLoginError as exc:
        raise HTTPException(exc.status_code, detail={"code": exc.code}) from None
    except CodexAuthorizationAttachError as exc:
        status = 404 if exc.code == "codex_authorization_account_not_found" else 409
        raise HTTPException(status, detail={"code": exc.code}) from None


def create_router() -> APIRouter:
    router = APIRouter(prefix="/api/workbench/ai")

    @router.get("/accounts")
    async def accounts(authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return {"items": await run_in_threadpool(account_service.list_owned_accounts, owner)}

    @router.get("/pool/accounts")
    async def pool_accounts(authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner_scope(authorization, x_workbench_account_owner)
        return {"items": await run_in_threadpool(account_service.list_pool_accounts)}

    @router.get("/pool/resources")
    async def pool_resources(authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner_scope(authorization, x_workbench_account_owner)
        from services.pool_resources import resource_snapshot
        from services.image_task_service import image_task_service
        from services.codex_service import codex_service
        return await run_in_threadpool(resource_snapshot, account_service, image_task_service, codex_service)

    @router.post("/pool/resource-settings")
    async def resource_settings(body: ResourceSettings, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner_scope(authorization, x_workbench_account_owner)
        from services.config import config
        return await account_operation(config.update_resource_settings, body.expected_revision,
                                       body.image_account_concurrency, body.codex_max_concurrency)

    @router.post("/accounts")
    async def import_account(body: ImportAccount, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        payload = {key: value.get_secret_value() if isinstance(value, SecretStr) else value for key, value in body if value is not None}
        return {"item": await account_operation(account_service.import_owned_account, owner, payload)}

    @router.post("/accounts/{account_id}/codex-authorization")
    async def attach_owned_codex_authorization(account_id: str, body: CodexAuthorization, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        payload = {key: value.get_secret_value() if isinstance(value, SecretStr) else value for key, value in body if value is not None}
        payload.pop("account_ref", None)
        await account_operation(account_service.attach_owned_codex_authorization, owner, account_id, payload)
        return {"attached": True}

    @router.post("/pool/codex-authorization")
    async def attach_codex_authorization(body: CodexAuthorization, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        # This internal bridge is reached only through the BFF's existing boss
        # gate. A normal program key never authorizes account management.
        owner_scope(authorization, x_workbench_account_owner)
        payload = {key: value.get_secret_value() if isinstance(value, SecretStr) else value for key, value in body if value is not None}
        account_ref = payload.pop("account_ref", None)
        await account_operation(account_service.attach_codex_authorization, payload, account_ref)
        return {"attached": True}

    @router.post("/pool/accounts/{account_ref}/enabled")
    async def enable_pool_account(account_ref: str, body: EnabledAccount, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner_scope(authorization, x_workbench_account_owner)
        return {"item": await account_operation(account_service.set_pool_account_enabled, account_ref, body.enabled)}

    @router.post("/pool/accounts/{account_ref}/label")
    async def label_pool_account(account_ref: str, body: AccountLabel, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner_scope(authorization, x_workbench_account_owner)
        return {"item": await account_operation(account_service.set_pool_account_label, account_ref, body.label)}

    @router.post("/pool/accounts/{account_ref}/refresh")
    async def refresh_pool_account(account_ref: str, body: AccountRefresh, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner_scope(authorization, x_workbench_account_owner)
        return {"item": await account_operation(account_service.refresh_pool_account, account_ref, body.routes or ["chat", "codex"], body.stale_only)}

    @router.post("/codex-observation")
    async def refresh_owned_codex_observation(body: CodexObservationTarget, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await account_operation(account_service.refresh_codex_observation, owner, body.account_ref, False)

    @router.post("/submitted-codex-observation")
    async def refresh_submitted_codex_observation(body: CodexObservationTarget, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await account_operation(account_service.refresh_submitted_codex_observation, owner, body.account_ref)

    @router.post("/pool/codex-observation")
    async def refresh_pool_codex_observation(body: CodexObservationTarget, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await account_operation(account_service.refresh_codex_observation, owner, body.account_ref, True)

    @router.post("/codex-login")
    async def start_codex_login(body: CodexLoginStart, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await login_operation(
            codex_login_service.start,
            owner,
            "owned",
            body.mode,
            body.client_request_id,
            body.account_ref,
        )

    @router.get("/codex-login/{session_id}")
    async def get_codex_login(session_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await login_operation(codex_login_service.get, owner, "owned", session_id)

    @router.delete("/codex-login/{session_id}")
    async def cancel_codex_login(session_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await login_operation(codex_login_service.cancel, owner, "owned", session_id)

    @router.post("/chat-login")
    async def start_chat_login(body: ChatLoginStart, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await chat_login_operation(
            chat_login_service.start, owner, "owned", body.mode,
            body.client_request_id, body.account_ref,
        )

    @router.get("/chat-login/{session_id}")
    async def get_chat_login(session_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await chat_login_operation(chat_login_service.get, owner, "owned", session_id)

    @router.post("/chat-login/{session_id}/callback")
    async def submit_chat_login_callback(session_id: str, body: ChatLoginCallback, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await chat_login_operation(
            chat_login_service.submit_callback, owner, "owned", session_id,
            body.callback_url.get_secret_value(),
        )

    @router.delete("/chat-login/{session_id}")
    async def cancel_chat_login(session_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await chat_login_operation(chat_login_service.cancel, owner, "owned", session_id)

    @router.post("/pool/codex-login")
    async def start_pool_codex_login(body: CodexLoginStart, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await login_operation(
            codex_login_service.start,
            owner,
            "pool",
            body.mode,
            body.client_request_id,
            body.account_ref,
        )

    @router.get("/pool/codex-login/{session_id}")
    async def get_pool_codex_login(session_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await login_operation(codex_login_service.get, owner, "pool", session_id)

    @router.delete("/pool/codex-login/{session_id}")
    async def cancel_pool_codex_login(session_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await login_operation(codex_login_service.cancel, owner, "pool", session_id)

    @router.post("/pool/chat-login")
    async def start_pool_chat_login(body: ChatLoginStart, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await chat_login_operation(
            chat_login_service.start, owner, "pool", body.mode,
            body.client_request_id, body.account_ref,
        )

    @router.get("/pool/chat-login/{session_id}")
    async def get_pool_chat_login(session_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await chat_login_operation(chat_login_service.get, owner, "pool", session_id)

    @router.post("/pool/chat-login/{session_id}/callback")
    async def submit_pool_chat_login_callback(session_id: str, body: ChatLoginCallback, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await chat_login_operation(
            chat_login_service.submit_callback, owner, "pool", session_id,
            body.callback_url.get_secret_value(),
        )

    @router.delete("/pool/chat-login/{session_id}")
    async def cancel_pool_chat_login(session_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return await chat_login_operation(chat_login_service.cancel, owner, "pool", session_id)

    @router.post("/accounts/{account_id}/refresh")
    async def refresh_account(account_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return {"item": await account_operation(account_service.refresh_owned_account, owner, account_id)}

    @router.post("/accounts/{account_id}/enabled")
    async def enable_account(account_id: str, body: EnabledAccount, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return {"item": await account_operation(account_service.set_owned_account_enabled, owner, account_id, body.enabled)}

    @router.get("/models")
    async def models(authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner_scope(authorization, x_workbench_account_owner)
        from services.codex_service import codex_service
        return await run_in_threadpool(codex_service.management_models)

    @router.get("/keys")
    async def keys(authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        return {"items": await run_in_threadpool(auth_service.list_owned_keys, owner)}

    @router.post("/keys")
    async def create_key(body: KeyName, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        try:
            item, key = await run_in_threadpool(auth_service.create_key, role="user", name=body.name, owner_subject=owner, routes=body.routes)
        except PolicyError as exc:
            raise HTTPException(422, detail={"code": exc.code}) from None
        except ValueError:
            raise HTTPException(409, detail={"error": "key name already exists"}) from None
        return {"item": item, "key": key}

    @router.patch("/keys/{key_id}/policy")
    async def update_key_policy(key_id: str, body: KeyPolicyUpdate, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        try:
            item = await run_in_threadpool(auth_service.update_owned_policy, owner, key_id, body.routes, body.expected_revision)
        except PolicyError as exc:
            raise HTTPException(409 if exc.code == "KEY_POLICY_REVISION_CONFLICT" else 422, detail={"code": exc.code}) from None
        if item is None:
            raise HTTPException(404, detail={"error": "key not found"})
        return {"item": item}

    @router.delete("/keys/{key_id}")
    async def revoke_key(key_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        if not await run_in_threadpool(auth_service.revoke_owned_key, owner, key_id):
            raise HTTPException(404, detail={"error": "key not found"})
        return {"revoked": True}

    return router
