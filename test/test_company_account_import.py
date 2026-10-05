"""Company import adapter regressions, defined for later unified validation.

Controlled import and projection fixtures exercise this adapter only. They are
not evidence for real upstream authorization, persistence or browser behavior.
"""
from __future__ import annotations

from copy import deepcopy
from threading import RLock

import pytest

from services import company_account_service as subject


REF = "car_" + "a" * 43
OTHER_REF = "car_" + "b" * 43


class Accounts:
    def __init__(self):
        self.lock = RLock()
        self.row = {
            "ref": REF, "managed_owner": "workbench:org:first-importer",
            "managed_account_id": "old-managed-id", "quota": 7,
            "access_token": "fixture-only-private-material",
            "original_receipts": {"fixture-original": "unknown"},
        }
        self.calls = []
        self.attachments = []
        self.attach_error = None
        self.receipt = {"authorization_ref": REF, "id": "old-managed-id"}
        self.missing_after_import = False

    def admission_transaction(self):
        return self.lock

    def pool_account_ref(self, row):
        return row["ref"]

    def _pool_account_locked(self, ref):
        assert self.lock._is_owned()
        if ref not in {REF, OTHER_REF} or (self.calls and self.missing_after_import):
            raise KeyError("fixture missing")
        return "fixture-token", {**self.row, "ref": ref}

    def import_owned_account(self, actor, payload):
        assert not self.lock._is_owned(), "import I/O must not hold the pool lock"
        self.calls.append((actor, deepcopy(payload)))
        return deepcopy(self.receipt)


    def attach_codex_authorization(self, credentials, ref):
        assert not self.lock._is_owned(), "attachment I/O must not hold the pool lock"
        if self.attach_error is not None:
            raise self.attach_error
        self.attachments.append((deepcopy(credentials), ref))
        return {"attached": True}


@pytest.fixture
def accounts(monkeypatch):
    # Existing production projection has separate coverage. Here the adapter
    # must use that projection, never pass through the legacy import response.
    monkeypatch.setattr(subject, "public_pool_account", lambda row: {
        "id": row["ref"], "account_ref": row["ref"],
        "authorization_ref": row["ref"], "label": "Company account",
    })
    return Accounts()


def test_other_manager_gets_the_same_company_reference_without_reassignment(accounts):
    before = deepcopy(accounts.row)
    actor = "workbench:org:second-manager"
    result = subject.import_company_account(accounts, actor, {"access_token": "fixture-input"})
    assert result == {"resource_scope": "company", "item": {
        "id": REF, "account_ref": REF, "authorization_ref": REF, "label": "Company account",
    }}
    assert accounts.calls[0][0] == actor
    assert accounts.row == before
    assert "managed_owner" not in result["item"]
    assert "access_token" not in result["item"]


def test_another_manager_may_name_a_company_target(accounts):
    result = subject.import_company_account(accounts, "workbench:org:other", {
        "access_token": "fixture-input", "account_ref": REF,
    })
    assert result["item"]["id"] == REF
    assert len(accounts.calls) == 1


def test_unresolved_target_is_rejected_before_import(accounts):
    with pytest.raises(KeyError):
        subject.import_company_account(accounts, "workbench:org:other", {
            "access_token": "fixture-input", "account_ref": "missing",
        })
    assert accounts.calls == []


@pytest.mark.parametrize("receipt", [None, {}, {"authorization_ref": "old-managed-id"}])
def test_missing_import_receipt_does_not_replay_the_import(accounts, receipt):
    accounts.receipt = receipt
    with pytest.raises(subject.CompanyAccountReadbackRequired):
        subject.import_company_account(accounts, "workbench:org:other", {"access_token": "fixture-input"})
    assert len(accounts.calls) == 1


def test_post_import_target_disappearance_requires_readback_not_second_import(accounts):
    accounts.missing_after_import = True
    with pytest.raises(subject.CompanyAccountReadbackRequired):
        subject.import_company_account(accounts, "workbench:org:other", {"access_token": "fixture-input"})
    assert len(accounts.calls) == 1


def test_named_target_cannot_silently_become_another_account(accounts):
    accounts.receipt = {"authorization_ref": OTHER_REF}
    with pytest.raises(subject.CompanyAccountReadbackRequired):
        subject.import_company_account(accounts, "workbench:org:other", {
            "access_token": "fixture-input", "account_ref": REF,
        })
    assert len(accounts.calls) == 1


def test_canonical_receipt_does_not_forward_legacy_private_fields(accounts):
    accounts.receipt.update(access_token="fixture-should-not-pass-through", managed_owner="old-owner")
    result = subject.import_company_account(accounts, "workbench:org:other", {"access_token": "fixture-input"})
    assert set(result["item"]) == {"id", "account_ref", "authorization_ref", "label"}


def codex_payload():
    return {"source_type": "codex", "access_token": "fixture-access",
            "refresh_token": "fixture-refresh", "id_token": "fixture-id",
            "account_id": "fixture-workspace", "account_ref": REF}


def test_named_codex_account_uses_identity_checked_attachment_not_generic_import(accounts):
    result = subject.import_company_account(accounts, "workbench:org:other", codex_payload())
    assert result["item"]["id"] == REF
    assert accounts.calls == [] and len(accounts.attachments) == 1
    assert accounts.attachments[0][1] == REF
    assert "account_ref" not in accounts.attachments[0][0]


def test_rejected_codex_attachment_never_falls_back_to_import(accounts):
    accounts.attach_error = ValueError("fixture identity conflict")
    with pytest.raises(ValueError, match="identity conflict"):
        subject.import_company_account(accounts, "workbench:org:other", codex_payload())
    assert accounts.calls == [] and accounts.attachments == []


@pytest.mark.parametrize("missing", ["access_token", "refresh_token", "id_token", "account_id"])
def test_partial_codex_credentials_fail_before_any_import_or_attachment(accounts, missing):
    payload = codex_payload()
    payload.pop(missing)
    with pytest.raises(ValueError, match="complete Codex"):
        subject.import_company_account(accounts, "workbench:org:other", payload)
    assert accounts.calls == [] and accounts.attachments == []
