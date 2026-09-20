from types import SimpleNamespace

import pytest

from services import editable_file_task_service as files
from services.request_context import current_request
from test.test_reliable_pool_routes import runtime


@pytest.fixture
def export_runtime(runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(files, "account_service", runtime.accounts)
    monkeypatch.setattr(runtime.accounts, "refresh_access_token", lambda token, **kwargs: token)
    monkeypatch.setattr(files, "EDITABLE_FILE_ROOT", tmp_path / "exports")
    monkeypatch.setattr(files.EditableFileTaskService, "_log_call", lambda *args, **kwargs: None)
    runtime.files = files.EditableFileTaskService(tmp_path / "file-tasks.json", text_tasks=runtime.service)
    runtime.exports = []
    class Backend:
        def __init__(self, token):
            assert token == runtime.account["access_token"]
        def export_ppt_zip(self, images, prompt, output_dir):
            context = current_request.get()
            context.before_send()
            runtime.exports.append((images, prompt, context.receipt()["provider_account_identity"]))
            if prompt == "unknown":
                raise TimeoutError("unknown model response")
            return SimpleNamespace(conversation_id="original-file-conversation", primary_path=output_dir / "result.pptx", zip_path=output_dir / "result.zip")
        export_psd_zip = export_ppt_zip
        def close(self):
            pass
    monkeypatch.setattr(files, "OpenAIBackendAPI", Backend)
    return runtime


@pytest.mark.parametrize("kind", ["ppt", "psd"])
def test_file_task_waits_in_same_pool_and_keeps_input_across_restart(export_runtime, kind):
    r = export_runtime
    r.service.submit("other", {"client_request_id": "busy", "client_conversation_id": "busy-session", "model": "fixture-text"})
    busy = r.admission.claim_next()
    identity = {"id": "file-user", "role": "user"}
    submit = getattr(r.files, "submit_" + kind)
    original = submit(identity, client_task_id="original-file", prompt="original prompt", base64_images=["original-image-bytes"])
    assert original["status"] == "queued"
    assert r.admission.claim_next() is None
    restarted = files.EditableFileTaskService(r.files.path, text_tasks=r.service)
    assert restarted.list_tasks(identity, ["original-file"])["items"][0]["status"] == "queued"
    r.admission.update_claim(busy, status="succeeded", _turn_reserved=False, _executing=False)
    claim = r.admission.claim_next()
    r.admission.execute(claim)
    assert restarted.list_tasks(identity, ["original-file"])["items"][0]["status"] == "success"
    assert r.exports == [(["original-image-bytes"], "original prompt", "account-0")]
    assert submit(identity, client_task_id="original-file", prompt="original prompt", base64_images=["original-image-bytes"])["status"] == "success"
    assert r.admission.claim_next() is None


def test_unknown_file_task_does_not_reexport_on_same_id_or_account_change(export_runtime):
    r = export_runtime
    who = {"id": "file-user", "role": "user"}
    r.files.submit_ppt(who, client_task_id="original", prompt="unknown")
    r.admission.execute(r.admission.claim_next())
    for _ in range(2):
        result = r.files.submit_ppt(who, client_task_id="original", prompt="unknown")
        assert result["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
        assert r.admission.claim_next() is None
    assert len(r.exports) == 1


def test_invalid_file_input_is_terminal_without_a_model_send(export_runtime):
    r = export_runtime
    who = {"id": "file-user", "role": "user"}
    r.files.submit_psd(who, client_task_id="invalid", base64_images=[])
    r.admission.execute(r.admission.claim_next())
    assert r.files.list_tasks(who, ["invalid"])["items"][0]["error_code"] == "EDITABLE_FILE_INPUT_INVALID"
    assert r.admission.claim_next() is None
    assert r.exports == []
