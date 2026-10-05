"""Company-facing import over the existing account authority, not a new pool.

The legacy importer records a submitting actor in managed_owner. That field is
retained as provenance for existing records; it must not select the company
account inventory or confer exclusive access to an upstream account.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from services.owned_accounts import public_pool_account

if TYPE_CHECKING:
    from services.account_service import AccountService


_POOL_REF = re.compile(r"car_[A-Za-z0-9_-]{43}")


class CompanyAccountReadbackRequired(RuntimeError):
    """Import may have committed, but a unique company receipt is unconfirmed."""


def import_company_account(
    accounts: AccountService, submitted_by: str, payload: dict,
) -> dict:
    """Reuse verified import/deduplication and return the canonical pool row.

    No pre-existing account is reassigned, copied, enabled or quota-adjusted by
    this adapter. The submitting actor is supplied only by the authenticated
    bridge. Never hold the account lock over the importer's upstream I/O.
    """
    expected_ref = None
    requested_ref = payload.get("account_ref")
    if requested_ref:
        with accounts.admission_transaction():
            _token, target = accounts._pool_account_locked(requested_ref)
            expected_ref = accounts.pool_account_ref(target)

    if payload.get("source_type") == "codex":
        required = ("access_token", "refresh_token", "id_token", "account_id")
        if any(not isinstance(payload.get(name), str) or not payload[name].strip() for name in required):
            raise ValueError("complete Codex authorization is required")

    if expected_ref is not None and payload.get("source_type") == "codex":
        # The generic Codex importer selects by credential identity rather than
        # account_ref. For a named target use its existing attachment writer,
        # which verifies that identity before changing the selected account.
        credentials = {name: payload[name] for name in required}
        accounts.attach_codex_authorization(credentials, expected_ref)
        reference = expected_ref
    else:
        # The legacy name denotes the existing writer, not a personal pool. It
        # already verifies authorization identity, deduplicates and persists.
        receipt = accounts.import_owned_account(submitted_by, payload)
        reference = receipt.get("authorization_ref") if isinstance(receipt, dict) else None
    if not isinstance(reference, str) or not _POOL_REF.fullmatch(reference):
        raise CompanyAccountReadbackRequired("company_account_readback_required")

    try:
        with accounts.admission_transaction():
            _token, target = accounts._pool_account_locked(reference)
            actual_ref = accounts.pool_account_ref(target)
            if expected_ref is not None and actual_ref != expected_ref:
                raise CompanyAccountReadbackRequired("company_account_readback_required")
            # Project from the persisted authority rather than forwarding an
            # owned receipt or arbitrary fields returned by an import handler.
            item = public_pool_account(target)
    except (KeyError, ValueError):
        raise CompanyAccountReadbackRequired("company_account_readback_required") from None

    # Stable company ref is also the ID used by existing pool refresh/enable
    # endpoints. Never return the submitting actor's managed_account_id here.
    if item.get("id") != actual_ref or item.get("account_ref") != actual_ref:
        raise CompanyAccountReadbackRequired("company_account_readback_required")
    return {"item": item, "resource_scope": "company"}
