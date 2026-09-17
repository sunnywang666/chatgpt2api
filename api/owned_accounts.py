"""Internal Workbench management bridge; never exposed by public AI ingress."""
from fastapi import APIRouter, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from api.support import require_admin
from services.account_service import account_service
from services.auth_service import auth_service


class ImportAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    access_token: SecretStr
    refresh_token: SecretStr | None = None
    id_token: SecretStr | None = None
    source_type: str = "web"
    account_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,200}$")


class EnabledAccount(BaseModel):
    enabled: bool


class KeyName(BaseModel):
    name: str = ""


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
    except ValueError:
        raise HTTPException(409, detail={"error": "account import conflicts with existing account scope or has invalid input"}) from None


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

    @router.post("/accounts")
    async def import_account(body: ImportAccount, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        payload = {key: value.get_secret_value() if isinstance(value, SecretStr) else value for key, value in body if value is not None}
        return {"item": await account_operation(account_service.import_owned_account, owner, payload)}

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
            item, key = await run_in_threadpool(auth_service.create_key, role="user", name=body.name, owner_subject=owner)
        except ValueError:
            raise HTTPException(409, detail={"error": "key name already exists"}) from None
        return {"item": item, "key": key}

    @router.delete("/keys/{key_id}")
    async def revoke_key(key_id: str, authorization: str | None = Header(default=None), x_workbench_account_owner: str | None = Header(default=None)):
        owner = owner_scope(authorization, x_workbench_account_owner)
        if not await run_in_threadpool(auth_service.revoke_owned_key, owner, key_id):
            raise HTTPException(404, detail={"error": "key not found"})
        return {"revoked": True}

    return router
