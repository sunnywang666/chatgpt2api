"""Company account import on the existing trusted Workbench management bridge.

Listing, refresh, enable, labels and account authorization attachment reuse the
already registered /api/workbench/ai/pool endpoints. No duplicate routes, queue,
account store or ordinary-program-key management authority are introduced.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import Field, SecretStr, model_validator

from api.owned_accounts import ImportAccount, account_operation, owner_scope
from services.account_service import account_service
from services.company_account_service import (
    CompanyAccountReadbackRequired,
    import_company_account,
)
from services.storage.base import AccountCommitUncertain


class CompanyAccountImport(ImportAccount):
    access_token: SecretStr = Field(min_length=1, max_length=20_000)
    refresh_token: SecretStr | None = Field(default=None, max_length=20_000)
    id_token: SecretStr | None = Field(default=None, max_length=20_000)
    source_type: Literal["web", "codex"] = "web"

    @model_validator(mode="after")
    def complete_codex_authorization(self):
        if self.source_type == "codex":
            values = (self.refresh_token, self.id_token)
            if (any(value is None or not value.get_secret_value().strip() for value in values)
                    or not self.account_id):
                raise ValueError("complete Codex authorization is required")
        return self


def _private_no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "private, no-store"


def create_router() -> APIRouter:
    router = APIRouter(
        prefix="/api/workbench/ai/pool", dependencies=[Depends(_private_no_store)],
    )

    @router.post("/accounts")
    async def import_account(
        body: CompanyAccountImport,
        authorization: str | None = Header(default=None),
        x_workbench_account_owner: str | None = Header(default=None),
    ):
        # The BFF checks company-management authority. The Provider checks its
        # existing trusted admin bridge, never an ordinary caller's program Key.
        submitted_by = owner_scope(authorization, x_workbench_account_owner)
        payload = {
            key: value.get_secret_value() if isinstance(value, SecretStr) else value
            for key, value in body if value is not None
        }
        try:
            return await account_operation(
                import_company_account, account_service, submitted_by, payload,
            )
        except (CompanyAccountReadbackRequired, AccountCommitUncertain):
            # Import is not replayed to obtain a nicer receipt. The company
            # inventory/original authorization must be reconciled first.
            raise HTTPException(
                503,
                detail={"code": "COMPANY_ACCOUNT_READBACK_REQUIRED", "outcome": "unknown"},
                headers={"Cache-Control": "private, no-store"},
            ) from None

    return router
