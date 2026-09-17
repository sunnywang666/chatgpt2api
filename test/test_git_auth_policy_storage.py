import json
import multiprocessing
import os
from pathlib import Path

import pytest
from git import Actor, Repo
from git.exc import GitCommandError

from services.storage.git_storage import GitStorageBackend


def _git_identity() -> None:
    os.environ.setdefault("GIT_AUTHOR_NAME", "Provider Test")
    os.environ.setdefault("GIT_AUTHOR_EMAIL", "provider-test@example.invalid")
    os.environ.setdefault("GIT_COMMITTER_NAME", "Provider Test")
    os.environ.setdefault("GIT_COMMITTER_EMAIL", "provider-test@example.invalid")


def _backend(remote: str, cache: str) -> GitStorageBackend:
    _git_identity()
    return GitStorageBackend(remote, "", branch="main", local_cache_dir=Path(cache))


def _hold_auth_transaction(remote, cache, entered, release, result):
    try:
        backend = _backend(remote, cache)
        with backend.auth_keys_transaction() as items:
            items.append({"id": "key-1", "enabled": True})
            entered.set()
            if not release.wait(15):
                raise RuntimeError("test did not release auth transaction")
        result.put(("ok", ""))
    except Exception as exc:
        result.put(("error", str(exc)))


def _checkout_operation(remote, cache, operation, started, finished, result):
    try:
        backend = _backend(remote, cache)
        started.set()
        if operation == "load":
            backend.load_accounts()
        elif operation == "health":
            state = backend.health_check()
            if state["status"] != "healthy":
                raise RuntimeError(state["error"])
        elif operation == "save":
            backend.save_accounts([{"access_token": "account-1"}])
        else:
            raise AssertionError(operation)
        result.put(("ok", ""))
    except Exception as exc:
        result.put(("error", str(exc)))
    finally:
        finished.set()


@pytest.fixture
def bare_remote(tmp_path, monkeypatch):
    _git_identity()
    for name in (
        "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.setenv(name, os.environ[name])
    remote_path = tmp_path / "remote.git"
    seed_path = tmp_path / "seed"
    remote = Repo.init(remote_path, bare=True)
    seed = Repo.init(seed_path)
    (seed_path / "accounts.json").write_text("[]\n", encoding="utf-8")
    (seed_path / "auth_keys.json").write_text('{"items": []}\n', encoding="utf-8")
    seed.index.add(["accounts.json", "auth_keys.json"])
    actor = Actor("Provider Test", "provider-test@example.invalid")
    seed.index.commit("seed", author=actor, committer=actor)
    seed.create_remote("origin", str(remote_path)).push(refspec="HEAD:refs/heads/main")
    remote.git.symbolic_ref("HEAD", "refs/heads/main")
    return remote_path


@pytest.mark.parametrize("operation", ["load", "save", "health"])
def test_shared_checkout_operations_wait_for_auth_transaction(bare_remote, tmp_path, operation):
    context = multiprocessing.get_context("spawn")
    cache = str(tmp_path / f"shared-{operation}")
    entered, release = context.Event(), context.Event()
    started, finished = context.Event(), context.Event()
    auth_result, operation_result = context.Queue(), context.Queue()
    auth = context.Process(
        target=_hold_auth_transaction,
        args=(str(bare_remote), cache, entered, release, auth_result),
    )
    other = context.Process(
        target=_checkout_operation,
        args=(str(bare_remote), cache, operation, started, finished, operation_result),
    )
    auth.start()
    try:
        assert entered.wait(15)
        other.start()
        assert started.wait(15)
        assert not finished.wait(0.3), "shared checkout operation escaped the process lock"
        release.set()
        auth.join(15)
        other.join(15)
        assert auth.exitcode == 0 and other.exitcode == 0
        assert auth_result.get(timeout=2) == ("ok", "")
        assert operation_result.get(timeout=2) == ("ok", "")
    finally:
        release.set()
        for process in (auth, other):
            if process.pid and process.is_alive():
                process.terminate()
                process.join()

    checkout = Repo.clone_from(str(bare_remote), tmp_path / f"verify-{operation}", branch="main")
    auth_keys = json.loads((Path(checkout.working_dir) / "auth_keys.json").read_text())
    assert auth_keys["items"] == [{"id": "key-1", "enabled": True}]
    if operation == "save":
        accounts = json.loads((Path(checkout.working_dir) / "accounts.json").read_text())
        assert accounts == [{"access_token": "account-1"}]
        account_commit = checkout.head.commit
        assert {diff.b_path for diff in account_commit.parents[0].diff(account_commit)} == {"accounts.json"}
        auth_commit = account_commit.parents[0]
        assert {diff.b_path for diff in auth_commit.parents[0].diff(auth_commit)} == {"auth_keys.json"}


def test_rejected_push_keeps_unpublished_checkout_and_fails_closed(bare_remote, tmp_path):
    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    first = _backend(str(bare_remote), str(cache_a))
    second = _backend(str(bare_remote), str(cache_b))
    first.load_accounts()
    second.load_accounts()

    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Event()
    result = context.Queue()
    writer = context.Process(
        target=_hold_auth_transaction,
        args=(str(bare_remote), str(cache_b), entered, release, result),
    )
    writer.start()
    try:
        assert entered.wait(15)
        first.save_accounts([{"access_token": "remote-advance"}])
        release.set()
        writer.join(15)
        assert writer.exitcode == 0
        state, message = result.get(timeout=2)
        assert state == "error"
        assert "reconciliation" in message
    finally:
        release.set()
        if writer.is_alive():
            writer.terminate()
            writer.join()

    local_repo_path = cache_b / "repo"
    unpublished_head = Repo(local_repo_path).head.commit.hexsha
    health = second.health_check()
    assert health["status"] == "unhealthy"
    assert "unpublished or diverged" in health["error"]
    assert local_repo_path.is_dir()
    assert Repo(local_repo_path).head.commit.hexsha == unpublished_head
    with pytest.raises(RuntimeError, match="unpublished or diverged"):
        second.load_auth_keys()

    remote_checkout = Repo.clone_from(str(bare_remote), tmp_path / "remote-readback", branch="main")
    remote_auth = json.loads((Path(remote_checkout.working_dir) / "auth_keys.json").read_text())
    remote_accounts = json.loads((Path(remote_checkout.working_dir) / "accounts.json").read_text())
    assert remote_auth == {"items": []}
    assert remote_accounts == [{"access_token": "remote-advance"}]


def test_transport_errors_do_not_expose_credential_urls(bare_remote, tmp_path, monkeypatch, capsys):
    backend = _backend(str(bare_remote), str(tmp_path / "credential-error"))
    secret = "credential-must-not-escape"
    failure = GitCommandError(
        "fetch",
        128,
        stderr=f"fatal: unable to access https://{secret}@example.invalid/private.git",
    )
    monkeypatch.setattr(backend, "_ready_repo_locked", lambda: (_ for _ in ()).throw(failure))

    health = backend.health_check()
    assert health == {
        "status": "unhealthy",
        "backend": "git",
        "error": "git storage operation failed (GitCommandError)",
    }
    with pytest.raises(RuntimeError, match=r"git storage operation failed \(GitCommandError\)"):
        backend.load_accounts()
    with pytest.raises(RuntimeError, match=r"git storage operation failed \(GitCommandError\)"):
        with backend.auth_keys_transaction():
            pass
    output = capsys.readouterr().out
    assert secret not in output
    assert "example.invalid" not in output
