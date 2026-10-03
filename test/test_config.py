import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_CONFIG_FILE = ROOT_DIR / "config.json"


class ConfigLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._created_root_config = False
        if not ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")
            cls._created_root_config = True

        from services import config as config_module

        cls.config_module = config_module

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._created_root_config and ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.unlink()

    def test_load_settings_ignores_directory_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            data_dir = base_dir / "data"
            config_dir = base_dir / "config.json"
            os_auth_key = "env-auth"

            config_dir.mkdir()

            module = self.config_module
            old_base_dir = module.BASE_DIR
            old_data_dir = module.DATA_DIR
            old_config_file = module.CONFIG_FILE
            old_env_auth_key = module.os.environ.get("CHATGPT2API_AUTH_KEY")
            try:
                module.BASE_DIR = base_dir
                module.DATA_DIR = data_dir
                module.CONFIG_FILE = config_dir
                module.os.environ["CHATGPT2API_AUTH_KEY"] = os_auth_key

                settings = module._load_settings()

                self.assertEqual(settings.auth_key, os_auth_key)
                self.assertEqual(settings.refresh_account_interval_minute, 5)
            finally:
                module.BASE_DIR = old_base_dir
                module.DATA_DIR = old_data_dir
                module.CONFIG_FILE = old_config_file
                if old_env_auth_key is None:
                    module.os.environ.pop("CHATGPT2API_AUTH_KEY", None)
                else:
                    module.os.environ["CHATGPT2API_AUTH_KEY"] = old_env_auth_key


class PacingSettingsTests(unittest.TestCase):
    def setUp(self):
        from services.config import ConfigStore
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.json"
        self.path.write_text(json.dumps({"auth-key": "test-pacing-settings-secret", "proxy": "original"}))
        self.store = ConfigStore(self.path)

    def test_invalid_pacing_rejects_entire_update_without_saving(self):
        for key, bad in [
            ("account_request_interval_secs", [0, .5, 61, True, None, "bad", float("nan"), float("inf")]),
            ("account_message_interval_secs", [1, 2, 4.99, 301, False, None, "-inf"]),
            ("account_conversation_read_interval_secs", [-1, 301, False, None, "bad", float("nan")]),
        ]:
            for value in bad:
                with self.subTest(key=key, value=value):
                    before = self.path.read_bytes()
                    with self.assertRaises(ValueError), patch.object(self.store, "_save") as save:
                        self.store.update({key: value, "proxy": "must-not-save"})
                    save.assert_not_called()
                    self.assertEqual(self.path.read_bytes(), before)
                    self.assertEqual(self.store.data["proxy"], "original")

        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.store.update({"account_request_interval_secs": 1,
                               "account_message_interval_secs": 2, "proxy": "must-not-save"})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertNotIn("account_request_interval_secs", self.store.data)

    def test_valid_boundaries_persist_and_match_runtime_after_reopen(self):
        from services.config import ConfigStore
        for http, message in [(1, 5), (60, 300), ("2.5", "7.5")]:
            with self.subTest(http=http, message=message):
                settings = self.store.update({"account_request_interval_secs": http,
                                              "account_message_interval_secs": message})
                reopened = ConfigStore(self.path)
                self.assertEqual(settings["account_request_interval_secs"], reopened.account_request_interval_secs)
                self.assertEqual(settings["account_message_interval_secs"], reopened.account_message_interval_secs)
                self.assertEqual(settings["account_message_interval_secs"], float(message))

    def test_legacy_invalid_get_shows_effective_values_without_rewriting_file(self):
        from services.config import ConfigStore
        raw = {"auth-key": "test-pacing-settings-secret", "account_request_interval_secs": .5,
               "account_message_interval_secs": 2}
        self.path.write_text(json.dumps(raw))
        self.store = ConfigStore(self.path)
        before = self.path.read_bytes()
        settings = self.store.get()
        self.assertEqual(settings["account_request_interval_secs"], 5)
        self.assertEqual(settings["account_message_interval_secs"], 30)
        self.assertNotIn("auth-key", settings)
        self.assertEqual(self.path.read_bytes(), before)
        self.store.update({"proxy": "unrelated-allowed"})
        self.assertEqual(self.store.data["account_message_interval_secs"], 2)
        self.assertEqual(self.store.get()["account_message_interval_secs"], 30)

    def test_conversation_read_setting_is_independent_and_defaults_to_no_extra_floor(self):
        from services.config import ConfigStore
        self.assertEqual(self.store.get()["account_conversation_read_interval_secs"], 0)
        for value in (0, 15, 300, "2.5"):
            settings = self.store.update({"account_conversation_read_interval_secs": value})
            reopened = ConfigStore(self.path)
            self.assertEqual(settings["account_conversation_read_interval_secs"], float(value))
            self.assertEqual(reopened.account_conversation_read_interval_secs, float(value))
            self.assertEqual(reopened.account_request_interval_secs, 5)
            self.assertEqual(reopened.account_message_interval_secs, 30)

    def test_settings_api_rejects_bad_pacing_and_reads_effective_legacy_value(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import api.system as system
        app = FastAPI()
        app.include_router(system.create_router("test"))
        with patch.object(system, "config", self.store), patch.object(system, "require_admin"):
            with TestClient(app) as client:
                response = client.post("/api/settings", json={"account_message_interval_secs": 2,
                                                              "proxy": "must-not-save"})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(self.store.data["proxy"], "original")
                self.store.data["account_message_interval_secs"] = 2
                response = client.get("/api/settings")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["config"]["account_message_interval_secs"], 30)
                response = client.post("/api/settings", json={"account_request_interval_secs": 1,
                                                              "account_message_interval_secs": 5})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["config"]["account_message_interval_secs"], 5)


if __name__ == "__main__":
    unittest.main()
