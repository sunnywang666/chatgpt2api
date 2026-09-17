from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from copy import deepcopy
from threading import Lock
from typing import Literal

from services.config import config
from services.storage.base import StorageBackend

from services.program_key_policy import ProgramKeyPolicy, PolicyError, make_policy


AuthRole = Literal["admin", "user"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()



class AuthService:
    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self._lock = Lock()
        self._items = self._load()
        self._last_used_flush_at: dict[str, datetime] = {}

    @staticmethod
    def _clean(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _default_name(role: object) -> str:
        return "管理员密钥" if str(role or "").strip().lower() == "admin" else "普通用户"

    def _normalize_item(self, raw: object) -> dict[str, object] | None:
        if not isinstance(raw, dict):
            return None
        role = self._clean(raw.get("role")).lower()
        if role not in {"admin", "user"}:
            return None
        key_hash = self._clean(raw.get("key_hash"))
        if not key_hash:
            return None
        item_id = self._clean(raw.get("id")) or uuid.uuid4().hex[:12]
        name = self._clean(raw.get("name")) or self._default_name(role)
        created_at = self._clean(raw.get("created_at")) or _now_iso()
        last_used_at = self._clean(raw.get("last_used_at")) or None
        return {
            "id": item_id,
            "owner_subject": self._clean(raw.get("owner_subject")),
            "name": name,
            "role": role,
            "key_hash": key_hash,
            "enabled": bool(raw.get("enabled", True)),
            "created_at": created_at,
            "last_used_at": last_used_at,
            "policy": deepcopy(raw.get("policy")),
        }

    def _load(self) -> list[dict[str, object]]:
        items = self.storage.load_auth_keys()
        if not isinstance(items, list):
            return []
        return [normalized for item in items if (normalized := self._normalize_item(item)) is not None]

    @contextmanager
    def _transaction(self, *, persist: bool = True):
        with self._lock, self.storage.auth_keys_transaction() as records:
            self._items = [normalized for raw in records
                           if (normalized := self._normalize_item(raw)) is not None]
            # Never silently destroy malformed records during an unrelated update.
            if len(self._items) != len(records):
                raise ValueError("invalid auth key storage")
            yield
            if persist:
                records[:] = self._items

    def _reload_locked(self) -> None:
        self._items = self._load()

    @staticmethod
    def _public_item(item: dict[str, object]) -> dict[str, object]:
        try:
            policy = ProgramKeyPolicy.from_record(item.get("policy")).to_record()
        except PolicyError:
            policy = None
        return {
            "id": item.get("id"),
            "name": item.get("name"),
            "role": item.get("role"),
            "enabled": bool(item.get("enabled", True)),
            "created_at": item.get("created_at"),
            "last_used_at": item.get("last_used_at"),
            "policy": policy,
            "policy_state": "active" if policy is not None else "reconciliation_required",
        }

    def list_keys(self, role: AuthRole | None = None) -> list[dict[str, object]]:
        with self._lock:
            self._reload_locked()
            items = [item for item in self._items if role is None or item.get("role") == role]
            return [self._public_item(item) for item in items]

    def _has_key_hash_locked(self, key_hash: str, *, exclude_id: str = "") -> bool:
        for item in self._items:
            item_id = self._clean(item.get("id"))
            if exclude_id and item_id == exclude_id:
                continue
            stored_hash = self._clean(item.get("key_hash"))
            if stored_hash and hmac.compare_digest(stored_hash, key_hash):
                return True
        return False

    def _build_key_hash_locked(self, raw_key: str, *, exclude_id: str = "") -> str:
        candidate = self._clean(raw_key)
        if not candidate:
            raise ValueError("请输入新的专用密钥")
        admin_key = self._clean(config.auth_key)
        if admin_key and hmac.compare_digest(candidate, admin_key):
            raise ValueError("这个密钥和管理员密钥冲突了，请换一个新的密钥")
        key_hash = _hash_key(candidate)
        if self._has_key_hash_locked(key_hash, exclude_id=exclude_id):
            raise ValueError("这个专用密钥已经存在，请换一个新的密钥")
        return key_hash

    def _has_name_locked(self, name: str, *, role: AuthRole | None = None, exclude_id: str = "") -> bool:
        candidate = self._clean(name)
        if not candidate:
            return False
        for item in self._items:
            item_id = self._clean(item.get("id"))
            if exclude_id and item_id == exclude_id:
                continue
            if role is not None and item.get("role") != role:
                continue
            if self._clean(item.get("name")) == candidate:
                return True
        return False

    def _build_default_name_locked(self, role: AuthRole, *, exclude_id: str = "") -> str:
        base_name = self._default_name(role)
        if not self._has_name_locked(base_name, role=role, exclude_id=exclude_id):
            return base_name
        suffix = 2
        while True:
            candidate = f"{base_name} {suffix}"
            if not self._has_name_locked(candidate, role=role, exclude_id=exclude_id):
                return candidate
            suffix += 1

    def _build_name_locked(self, name: str, *, role: AuthRole, exclude_id: str = "") -> str:
        candidate = self._clean(name)
        if not candidate:
            return self._build_default_name_locked(role, exclude_id=exclude_id)
        if self._has_name_locked(candidate, role=role, exclude_id=exclude_id):
            raise ValueError("这个名称已经在使用中了，换一个更容易区分的名称吧")
        return candidate

    def create_key(self, *, role: AuthRole, name: str = "", owner_subject: str = "", routes: list[str] | None = None) -> tuple[dict[str, object], str]:
        policy = make_policy(routes if routes is not None else ["chat"], revision=1) if role == "user" else None
        with self._transaction():
            normalized_name = self._build_name_locked(name, role=role)
            while True:
                raw_key = f"sk-{secrets.token_urlsafe(24)}"
                try:
                    key_hash = self._build_key_hash_locked(raw_key)
                    break
                except ValueError:
                    continue
            item = {
                "id": uuid.uuid4().hex[:12],
                "owner_subject": owner_subject,
                "name": normalized_name,
                "role": role,
                "key_hash": key_hash,
                "enabled": True,
                "created_at": _now_iso(),
                "last_used_at": None,
                "policy": policy.to_record() if policy else None,
            }
            self._items.append(item)
            return self._public_item(item), raw_key

    def list_owned_keys(self, owner: str) -> list[dict[str, object]]:
        with self._lock:
            self._reload_locked()
            return [self._public_item(item) for item in self._items
                    if item.get("role") == "user" and item.get("owner_subject") == owner]

    def revoke_owned_key(self, owner: str, key_id: str) -> bool:
        with self._transaction():
            for item in self._items:
                if item.get("id") == key_id and item.get("role") == "user" and item.get("owner_subject") == owner:
                    item["enabled"] = False
                    return True
        return False

    def update_key(
        self,
        key_id: str,
        updates: dict[str, object],
        *,
        role: AuthRole | None = None,
    ) -> dict[str, object] | None:
        normalized_id = self._clean(key_id)
        if not normalized_id:
            return None
        with self._transaction():
            for index, item in enumerate(self._items):
                if item.get("id") != normalized_id:
                    continue
                if role is not None and item.get("role") != role:
                    return None
                next_item = dict(item)
                next_role = "admin" if str(next_item.get("role") or "").strip().lower() == "admin" else "user"
                if "name" in updates and updates.get("name") is not None:
                    next_item["name"] = self._build_name_locked(
                        str(updates.get("name") or ""),
                        role=next_role,
                        exclude_id=normalized_id,
                    )
                if "enabled" in updates and updates.get("enabled") is not None:
                    next_item["enabled"] = bool(updates.get("enabled"))
                if "key" in updates and updates.get("key") is not None:
                    next_item["key_hash"] = self._build_key_hash_locked(str(updates.get("key") or ""), exclude_id=normalized_id)
                self._items[index] = next_item
                return self._public_item(next_item)
        return None

    def delete_key(self, key_id: str, *, role: AuthRole | None = None) -> bool:
        normalized_id = self._clean(key_id)
        if not normalized_id:
            return False
        with self._transaction():
            before = len(self._items)
            self._items = [
                item
                for item in self._items
                if not (item.get("id") == normalized_id and (role is None or item.get("role") == role))
            ]
            if len(self._items) == before:
                return False
            return True

    def update_owned_policy(self, owner: str, key_id: str, routes: list[str],
                            expected_revision: int) -> dict[str, object] | None:
        with self._transaction():
            for item in self._items:
                if item.get("id") != key_id or item.get("owner_subject") != owner or item.get("role") != "user":
                    continue
                existing = item.get("policy")
                revision = ProgramKeyPolicy.from_record(existing).revision if existing is not None else 0
                if type(expected_revision) is not int or expected_revision != revision:
                    raise PolicyError("KEY_POLICY_REVISION_CONFLICT")
                item["policy"] = make_policy(routes, revision=revision + 1).to_record()
                return self._public_item(item)
        return None

    def reconcile_legacy_policies(self, assignments: list[dict], *, apply: bool = False) -> list[dict]:
        """Operator-only cutover step, never exposed to ordinary API callers.

        Explicit key IDs/routes come from consumer/receipt reconciliation.
        Hashes, owners, enabled state and tasks are unchanged. All enabled legacy
        keys must be accounted for so guard cutover cannot silently strand one.
        """
        if not isinstance(assignments, list):
            raise PolicyError("LEGACY_ASSIGNMENTS_INVALID")
        by_id = {}
        for assignment in assignments:
            if not isinstance(assignment, dict) or set(assignment) - {"id", "routes"}:
                raise PolicyError("LEGACY_ASSIGNMENTS_INVALID")
            key_id = assignment.get("id")
            if not isinstance(key_id, str) or not key_id or key_id in by_id:
                raise PolicyError("LEGACY_ASSIGNMENTS_INVALID")
            by_id[key_id] = assignment
        with self._transaction(persist=apply):
            legacy = {str(item["id"]): item for item in self._items
                      if item["role"] == "user" and item["enabled"] and item.get("policy") is None}
            if set(legacy) != set(by_id):
                raise PolicyError("LEGACY_KEY_SET_CHANGED")
            result = []
            for key_id, assignment in by_id.items():
                policy = make_policy(assignment.get("routes"), revision=1)
                candidate = {**legacy[key_id], "policy": policy.to_record()}
                result.append(self._public_item(candidate))
                if apply:
                    legacy[key_id].update(policy=candidate["policy"])
            return result

    def authenticate(self, raw_key: str) -> dict[str, object] | None:
        candidate = self._clean(raw_key)
        if not candidate:
            return None
        candidate_hash = _hash_key(candidate)
        # Authentication and last_used updates share the same cross-process
        # transaction as revocation/policy changes. No stale snapshot is saved.
        with self._transaction():
            for item in self._items:
                if not bool(item.get("enabled", True)):
                    continue
                stored_hash = self._clean(item.get("key_hash"))
                if not stored_hash or not hmac.compare_digest(stored_hash, candidate_hash):
                    continue
                now = datetime.now(timezone.utc)
                item_id = self._clean(item.get("id"))
                last_flush_at = self._last_used_flush_at.get(item_id)
                if last_flush_at is None or (now - last_flush_at).total_seconds() >= 60:
                    item["last_used_at"] = now.isoformat()
                    self._last_used_flush_at[item_id] = now
                return self._public_item(item)
        return None


auth_service = AuthService(config.get_storage_backend())
