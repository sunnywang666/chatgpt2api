from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.account_service import AccountService
from services.storage.base import AccountCommitUncertain
from services.storage.json_storage import JSONStorageBackend


class JSONAccountStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "accounts.json"
        self.backend = JSONStorageBackend(self.path)
        self.original = [{"access_token": "fixture-chat", "managed_owner": "owner", "quota": 3}]
        self.backend.save_accounts(self.original)

    def test_partial_temp_write_failure_preserves_original_bytes_and_mode(self):
        before = self.path.read_bytes()
        def partial(_items, target, **_kwargs):
            target.write('[{"partial":')
            raise OSError("injected partial write")
        with patch("services.storage.json_storage.json.dump", side_effect=partial):
            with self.assertRaises(OSError):
                self.backend.save_accounts([{"access_token": "new"}])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.path.parent.glob(".accounts-*")), [])
        self.assertEqual(JSONStorageBackend(self.path).load_accounts(), self.original)

    def test_file_fsync_failure_does_not_replace_live_accounts(self):
        with patch("services.storage.json_storage.os.fsync", side_effect=OSError("file sync")):
            with self.assertRaises(OSError):
                self.backend.save_accounts([])
        self.assertEqual(self.backend.load_accounts(), self.original)

    def test_replace_error_before_or_after_commit_is_read_back_without_rewrite(self):
        original_replace = os.replace
        for replaced in (False, True):
            with self.subTest(replaced=replaced):
                self.backend.save_accounts(self.original)
                def uncertain(source, target):
                    if replaced:
                        original_replace(source, target)
                    raise OSError("replace response lost")
                candidate = [{"access_token": "new"}]
                with patch("services.storage.json_storage.os.replace", side_effect=uncertain):
                    with self.assertRaises(AccountCommitUncertain):
                        self.backend.save_accounts(candidate)
                self.assertEqual(self.backend.confirm_accounts_commit(), candidate if replaced else self.original)

    def test_directory_fsync_failure_leaves_complete_document_for_restart(self):
        candidate = self.original + [{"access_token": "new"}]
        with patch.object(JSONStorageBackend, "_sync_directory", side_effect=OSError("directory sync")):
            with self.assertRaises(AccountCommitUncertain):
                self.backend.save_accounts(candidate)
        self.assertEqual(JSONStorageBackend(self.path).confirm_accounts_commit(), candidate)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_corrupt_or_wrong_shape_pool_fails_closed_and_cannot_be_overwritten(self):
        for value in ('{broken', '{}', '[null]', '[1]', 'null'):
            with self.subTest(value=value):
                self.path.write_text(value)
                with self.assertRaises(RuntimeError):
                    self.backend.load_accounts()
                with self.assertRaises(RuntimeError):
                    self.backend.save_accounts([])
                with self.assertRaises(RuntimeError):
                    AccountService(self.backend)
                self.assertEqual(self.path.read_text(), value)
                self.assertEqual(self.backend.health_check()["status"], "unhealthy")

    def test_unreadable_pool_is_not_empty_and_new_pool_can_start_empty(self):
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(RuntimeError, "unreadable"):
                self.backend.load_accounts()
        fresh = JSONStorageBackend(self.path.with_name("new.json"))
        self.assertEqual(fresh.load_accounts(), [])
