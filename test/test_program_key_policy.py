"""Contract tests only; these do NOT exercise the real HTTP or storage wiring."""
from __future__ import annotations
import copy
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from services.program_key_policy import (
    Capability as C, Route as R, PolicyError, ProgramKeyPolicy, RequestUse, TaskBinding,
    authorize_read, authorize_submission, codex_use, image_use, make_policy,
    shared_limit_identity,
)

# Readiness is an injected TEST fixture, not a claim about deployed features.
READY = frozenset({C.CHAT_IMAGE, C.CODEX_CODING})
MODELS = {"gpt-image-2": R.CHAT, "codex-gpt-image-2": R.CODEX}
TOOLS = {"image_generation": frozenset({C.CODEX_IMAGE}),
         "web_search": frozenset({C.CODEX_CODING})}
CHAT_USE = RequestUse(R.CHAT, frozenset({C.CHAT_IMAGE}))
CODE_USE = RequestUse(R.CODEX, frozenset({C.CODEX_CODING}))

class PolicyTests(unittest.TestCase):
    def denied(self, code, function, *args, **kwargs):
        with self.assertRaises(PolicyError) as result:
            function(*args, **kwargs)
        self.assertEqual(code, result.exception.code)

    def policy(self, *capabilities):
        return make_policy(list(capabilities or ["chat_image"]), revision=1, ready=READY)

    def authorize(self, policy, use, **kwargs):
        return authorize_submission(policy, key_enabled=True, use=use, ready=READY, **kwargs)

    def test_chat_image_allowed(self):
        self.assertEqual(CHAT_USE, self.authorize(self.policy(), CHAT_USE))

    def test_chat_key_cannot_use_codex(self):
        self.denied("KEY_CAPABILITY_DENIED", self.authorize, self.policy(), CODE_USE)

    def test_coding_key_cannot_use_chat(self):
        self.denied("KEY_CAPABILITY_DENIED", self.authorize, self.policy("codex_coding"), CHAT_USE)

    def test_explicit_combination_not_default(self):
        policy = self.policy("codex_coding", "chat_image")
        self.assertEqual(CODE_USE, self.authorize(policy, CODE_USE))
        self.assertEqual(CHAT_USE, self.authorize(policy, CHAT_USE))

    def test_disabled_denies_new_request(self):
        self.denied("KEY_DISABLED", authorize_submission, self.policy(), key_enabled=False,
                    use=CHAT_USE, ready=READY)

    def test_truthy_string_is_not_enabled(self):
        self.denied("KEY_DISABLED", authorize_submission, self.policy(), key_enabled="true",
                    use=CHAT_USE, ready=READY)

    def test_unready_chat_planning_not_selectable(self):
        self.denied("CAPABILITY_NOT_READY", self.policy, "chat_text")

    def test_unready_codex_images_not_selectable(self):
        self.denied("CAPABILITY_NOT_READY", self.policy, "codex_image")

    def test_existing_policy_rechecks_runtime_readiness(self):
        self.denied("CAPABILITY_NOT_READY", authorize_submission, self.policy(),
                    key_enabled=True, use=CHAT_USE, ready=frozenset())

    def test_policy_json_roundtrip(self):
        original = self.policy("codex_coding", "chat_image")
        self.assertEqual(original, ProgramKeyPolicy.from_record(json.loads(json.dumps(original.to_record()))))

    def test_policy_output_mutation_cannot_change_policy(self):
        policy = self.policy()
        row = policy.to_record()
        row["capabilities"].append("codex_coding")
        self.assertEqual(frozenset({C.CHAT_IMAGE}), policy.capabilities)

    def test_legacy_missing_not_unlimited(self):
        self.denied("KEY_POLICY_RECONCILIATION_REQUIRED", ProgramKeyPolicy.from_record, None)

    def test_unknown_capability_denied(self):
        self.denied("KEY_POLICY_UNKNOWN_CAPABILITY", self.policy, "anything")

    def test_extra_budget_field_rejected(self):
        row = self.policy().to_record()
        row["budget"] = 500
        self.denied("KEY_POLICY_INVALID_RECORD", ProgramKeyPolicy.from_record, row)

    def test_no_boolean_version(self):
        row = self.policy().to_record(); row["version"] = True
        self.denied("KEY_POLICY_UNSUPPORTED_VERSION", ProgramKeyPolicy.from_record, row)

    def test_unknown_version_denied(self):
        row = self.policy().to_record(); row["version"] = 2
        self.denied("KEY_POLICY_UNSUPPORTED_VERSION", ProgramKeyPolicy.from_record, row)

    def test_invalid_revision_denied(self):
        for revision in (0, -1, True, 1.0, "1"):
            with self.subTest(revision=revision):
                self.denied("KEY_POLICY_INVALID_REVISION", make_policy,
                            ["chat_image"], revision=revision, ready=READY)

    def test_empty_capabilities_denied(self):
        self.denied("KEY_POLICY_INVALID_CAPABILITIES", make_policy, [], revision=1, ready=READY)

    def test_string_not_capability_list(self):
        self.denied("KEY_POLICY_INVALID_CAPABILITIES", make_policy, "chat_image", revision=1, ready=READY)

    def test_duplicate_capability_rejected(self):
        self.denied("KEY_POLICY_DUPLICATE_CAPABILITY", self.policy, "chat_image", "chat_image")

    def test_wrong_route_cannot_be_claimed(self):
        self.denied("REQUEST_ROUTE_MISMATCH", RequestUse, R.CHAT, frozenset({C.CODEX_CODING}))

    def test_mixed_route_cannot_be_claimed(self):
        self.denied("REQUEST_ROUTE_MISMATCH", RequestUse, R.CHAT, frozenset({C.CHAT_IMAGE, C.CODEX_IMAGE}))

    def test_typed_route_required(self):
        self.denied("REQUEST_USE_INVALID", RequestUse, "chat", frozenset({C.CHAT_IMAGE}))

    def test_image_model_chat_route(self):
        self.assertEqual(CHAT_USE, image_use("gpt-image-2", model_routes=MODELS))

    def test_image_model_codex_alias_rejected_for_chat_key(self):
        use = image_use("codex-gpt-image-2", model_routes=MODELS)
        self.denied("KEY_CAPABILITY_DENIED", self.authorize, self.policy(), use)

    def test_unknown_model_alias_never_defaults_to_chat(self):
        self.denied("MODEL_ROUTE_UNKNOWN", image_use, "gpt-image-2-fake", model_routes=MODELS)

    def test_model_registry_values_must_be_typed(self):
        self.denied("MODEL_ROUTE_UNKNOWN", image_use, "image", model_routes={"image": "chat"})

    def test_codex_plain_request(self):
        self.assertEqual(CODE_USE, codex_use({"input": "hello"}, native_tool_capabilities=TOOLS))

    def test_codex_function_image_name_is_not_native_image(self):
        payload = {"tools": [{"type": "function", "name": "image_generation"}]}
        self.assertEqual(CODE_USE, codex_use(payload, native_tool_capabilities=TOOLS))

    def test_codex_custom_tool_kept(self):
        self.assertEqual(CODE_USE, codex_use({"tools": [{"type": "custom", "name": "apply_patch"}]},
                                           native_tool_capabilities=TOOLS))

    def test_codex_native_image_needs_both_capabilities(self):
        use = codex_use({"tools": [{"type": "image_generation"}]}, native_tool_capabilities=TOOLS)
        self.assertEqual(frozenset({C.CODEX_CODING, C.CODEX_IMAGE}), use.capabilities)
        self.denied("KEY_CAPABILITY_DENIED", self.authorize, self.policy("codex_coding"), use)

    def test_codex_forced_image_also_detected(self):
        use = codex_use({"tool_choice": {"type": "image_generation"}}, native_tool_capabilities=TOOLS)
        self.denied("KEY_CAPABILITY_DENIED", self.authorize, self.policy("codex_coding"), use)

    def test_codex_namespace_nested_image_detected(self):
        use = codex_use({"tools": [{"type": "namespace", "tools": [{"type": "image_generation"}]}]},
                        native_tool_capabilities=TOOLS)
        self.denied("KEY_CAPABILITY_DENIED", self.authorize, self.policy("codex_coding"), use)

    def test_codex_unknown_builtin_denied(self):
        self.denied("NATIVE_TOOL_NOT_CLASSIFIED", codex_use,
                    {"tools": [{"type": "future_native_tool"}]}, native_tool_capabilities=TOOLS)

    def test_codex_tool_registry_cannot_claim_chat(self):
        self.denied("NATIVE_TOOL_NOT_CLASSIFIED", codex_use,
                    {"tools": [{"type": "image_generation"}]},
                    native_tool_capabilities={"image_generation": frozenset({C.CHAT_IMAGE})})

    def test_codex_builtin_search_kept(self):
        self.assertEqual(CODE_USE, codex_use({"tools": [{"type": "web_search"}]}, native_tool_capabilities=TOOLS))

    def test_invalid_tools_denied(self):
        for payload in ({"tools": None}, {"tools": "image_generation"}, {"tools": [None]},
                        {"tools": [{"type": "namespace"}]}, {"tool_choice": "image_generation"}):
            with self.subTest(payload=payload):
                self.denied("REQUEST_TOOL_INVALID", codex_use, payload, native_tool_capabilities=TOOLS)

    def test_codex_does_not_mutate_native_payload(self):
        payload = {"input": [{"type": "function_call_output", "call_id": "a", "output": "image_generation"}],
                   "tools": [{"type": "function", "name": "run"}], "tool_choice": "auto"}
        before = copy.deepcopy(payload)
        codex_use(payload, native_tool_capabilities=TOOLS)
        self.assertEqual(before, payload)

    def test_check_own_task_keeps_exact_binding(self):
        binding = TaskBinding("key", "account", "task", CHAT_USE)
        self.assertIs(binding, authorize_read(key_id="key", key_enabled=True, original=binding))

    def test_read_other_key_task_hidden(self):
        binding = TaskBinding("other-key", "account", "task", CHAT_USE)
        self.denied("TASK_NOT_FOUND", authorize_read, key_id="key", key_enabled=True, original=binding)

    def test_revoked_key_cannot_read_but_binding_unchanged(self):
        binding = TaskBinding("key", "account", "task", CHAT_USE)
        self.denied("KEY_DISABLED", authorize_read, key_id="key", key_enabled=False, original=binding)
        self.assertEqual("task", binding.task_id)

    def test_narrowing_policy_keeps_old_read_not_new_write(self):
        binding = TaskBinding("key", "account", "task", CHAT_USE)
        self.assertIs(binding, authorize_read(key_id="key", key_enabled=True, original=binding))
        self.denied("KEY_CAPABILITY_DENIED", self.authorize, self.policy("codex_coding"), CHAT_USE,
                    key_id="key", original=binding)

    def test_existing_task_cannot_switch_route(self):
        binding = TaskBinding("key", "account", "task", CHAT_USE)
        self.denied("TASK_ROUTE_CHANGE_FORBIDDEN", self.authorize,
                    self.policy("chat_image", "codex_coding"), CODE_USE, key_id="key", original=binding)

    def test_existing_task_cannot_switch_key(self):
        binding = TaskBinding("old-key", "account", "task", CHAT_USE)
        self.denied("TASK_NOT_FOUND", self.authorize, self.policy(), CHAT_USE, key_id="new-key", original=binding)

    def test_existing_task_unchanged_for_allowed_request(self):
        binding = TaskBinding("key", "account", "task", CHAT_USE)
        self.assertEqual(CHAT_USE, self.authorize(self.policy(), CHAT_USE, key_id="key", original=binding))
        self.assertEqual("account", binding.account_ref)

    def test_unknown_binding_not_accepted(self):
        self.denied("TASK_BINDING_INVALID", TaskBinding, "key", "", "task", CHAT_USE)

    def test_codex_images_and_coding_share_limit_identity(self):
        self.assertEqual(shared_limit_identity("a", C.CODEX_CODING, "limit", "week"),
                         shared_limit_identity("a", C.CODEX_IMAGE, "limit", "week"))

    def test_chat_image_and_model_not_one_balance(self):
        self.assertNotEqual(shared_limit_identity("a", C.CHAT_IMAGE, "limit", "week"),
                            shared_limit_identity("a", C.CHAT_TEXT, "limit", "week"))

    def test_accounts_and_windows_never_merge(self):
        identifiers = {shared_limit_identity(a, C.CODEX_CODING, "limit", w)
                       for a in ("a", "b") for w in ("5h", "week")}
        self.assertEqual(4, len(identifiers))

    def test_unknown_limit_cannot_be_called_zero(self):
        self.denied("LIMIT_IDENTITY_UNKNOWN", shared_limit_identity, "a", C.CODEX_IMAGE, "", "week")

    def test_nonsequence_capabilities_rejected(self):
        for value in (None, 7, True, {"chat_image"}):
            with self.subTest(value=value):
                self.denied("KEY_POLICY_INVALID_CAPABILITIES", make_policy,
                            value, revision=1, ready=READY)

    def test_codex_nonmapping_payload_rejected(self):
        self.denied("REQUEST_TOOL_INVALID", codex_use, None, native_tool_capabilities=TOOLS)

    def test_codex_excessive_tool_nesting_rejected(self):
        tool = {"type": "image_generation"}
        for _ in range(18):
            tool = {"type": "namespace", "tools": [tool]}
        self.denied("REQUEST_TOOL_NESTING_EXCEEDED", codex_use,
                    {"tools": [tool]}, native_tool_capabilities=TOOLS)

    def test_pure_policy_parallel_use(self):
        policy = self.policy()
        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(lambda _: self.authorize(policy, CHAT_USE), range(64)))
        self.assertTrue(all(outcome == CHAT_USE for outcome in outcomes))
        self.assertEqual(frozenset({C.CHAT_IMAGE}), policy.capabilities)

if __name__ == "__main__":
    unittest.main(verbosity=2)
