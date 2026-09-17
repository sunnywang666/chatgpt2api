from __future__ import annotations

import json
import fcntl
from contextlib import contextmanager
import tempfile
from pathlib import Path
from typing import Any

from git import Repo
from git.exc import GitCommandError

from services.storage.base import StorageBackend


class GitStorageStateError(RuntimeError):
    """A fixed, credential-free storage state error safe for logs and APIs."""


class GitStorageBackend(StorageBackend):
    """Git 私有仓库存储后端"""

    def __init__(
        self,
        repo_url: str,
        token: str,
        branch: str = "main",
        file_path: str = "accounts.json",
        auth_keys_file_path: str = "auth_keys.json",
        local_cache_dir: Path | None = None,
    ):
        self.repo_url = repo_url
        self.token = token
        self.branch = branch
        self.file_path = file_path
        self.auth_keys_file_path = auth_keys_file_path
        
        # 本地缓存目录
        if local_cache_dir is None:
            local_cache_dir = Path(tempfile.gettempdir()) / "chatgpt2api_git_cache"
        self.local_cache_dir = local_cache_dir
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建带认证的 Git URL
        self.auth_repo_url = self._build_auth_url(repo_url, token)

    @staticmethod
    def _build_auth_url(repo_url: str, token: str) -> str:
        """构建带认证的 Git URL"""
        if not token:
            return repo_url
        
        # 支持 HTTPS 格式：https://github.com/user/repo.git
        if repo_url.startswith("https://"):
            # 插入 token
            return repo_url.replace("https://", f"https://{token}@")
        
        # 支持 git@ 格式：git@github.com:user/repo.git
        # 转换为 HTTPS 格式
        if repo_url.startswith("git@"):
            repo_url = repo_url.replace("git@", "https://")
            repo_url = repo_url.replace(".com:", ".com/")
            return repo_url.replace("https://", f"https://{token}@")
        
        return repo_url

    @contextmanager
    def _checkout_lock(self):
        """Serialize every operation using the shared checkout."""
        lock_path = self.local_cache_dir / "checkout.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _ready_repo_locked(self) -> Repo:
        """Return a clean checkout at the fetched remote head.

        The caller must hold ``_checkout_lock``. Only a remote fast-forward is
        applied automatically. Dirty, unpublished or diverged state is kept for
        operator reconciliation instead of being deleted or merged.
        """
        repo_path = self.local_cache_dir / "repo"

        if repo_path.exists() and (repo_path / ".git").exists():
            repo = Repo(repo_path)
            if repo.head.is_detached or repo.active_branch.name != self.branch:
                raise GitStorageStateError("git checkout branch requires reconciliation")
            if repo.is_dirty(untracked_files=True):
                raise GitStorageStateError("git checkout is dirty; reconciliation required")
            origin = repo.remote("origin")
            try:
                origin.fetch()
            except Exception as exc:
                raise GitStorageStateError(self._safe_error(exc)) from None
            remote = repo.commit(f"origin/{self.branch}")
            head = repo.head.commit
            if head != remote:
                merge_bases = repo.merge_base(head, remote)
                if len(merge_bases) == 1 and merge_bases[0] == head:
                    repo.git.merge("--ff-only", f"origin/{self.branch}")
                else:
                    raise GitStorageStateError("git checkout has unpublished or diverged commits; reconciliation required")
            if repo.is_dirty(untracked_files=True) or repo.head.commit != repo.commit(f"origin/{self.branch}"):
                raise GitStorageStateError("git checkout requires reconciliation")
            return repo

        if repo_path.exists():
            raise GitStorageStateError("git checkout path exists but is not a repository")
        try:
            repo = Repo.clone_from(
                self.auth_repo_url,
                repo_path,
                branch=self.branch,
            )
        except Exception as exc:
            raise GitStorageStateError(self._safe_error(exc)) from None
        if repo.is_dirty(untracked_files=True):
            raise GitStorageStateError("new git checkout is dirty; reconciliation required")
        return repo

    @classmethod
    def _push_checked(cls, repo: Repo, branch: str) -> None:
        try:
            results = repo.remote("origin").push(branch)
        except Exception as exc:
            raise GitStorageStateError(cls._safe_error(exc)) from None
        if not results or any(
            result.flags & (result.ERROR | result.REJECTED | result.REMOTE_REJECTED)
            for result in results
        ):
            raise GitStorageStateError("git storage write conflicted; checkout requires reconciliation")

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, GitStorageStateError):
            return str(exc)
        return f"git storage operation failed ({type(exc).__name__})"

    def load_accounts(self) -> list[dict[str, Any]]:
        """从 Git 仓库加载账号数据"""
        try:
            return self._load_json_file(self.file_path)
        except Exception as e:
            message = self._safe_error(e)
            print(f"[git-storage] load failed: {message}")
            raise GitStorageStateError(message) from None

    def save_accounts(self, accounts: list[dict[str, Any]]) -> None:
        """保存账号数据到 Git 仓库"""
        try:
            self._save_json_file(self.file_path, accounts, "Update accounts data")
        except Exception as e:
            message = self._safe_error(e)
            print(f"[git-storage] save failed: {message}")
            raise GitStorageStateError(message) from None

    def load_auth_keys(self) -> list[dict[str, Any]]:
        """从 Git 仓库加载鉴权密钥数据"""
        try:
            data = self._load_json_value(self.auth_keys_file_path)
            if isinstance(data, dict):
                data = data.get("items")
            return data if isinstance(data, list) else []
        except Exception as e:
            message = self._safe_error(e)
            print(f"[git-storage] load failed: {message}")
            raise GitStorageStateError(message) from None

    def save_auth_keys(self, auth_keys: list[dict[str, Any]]) -> None:
        """保存鉴权密钥数据到 Git 仓库"""
        with self.auth_keys_transaction() as items:
            items[:] = auth_keys

    @contextmanager
    def auth_keys_transaction(self):
        # Remote writers still compete through ordinary fast-forward push. A
        # rejection leaves the local commit intact and makes later calls fail
        # closed until an operator reconciles it.
        try:
            with self._checkout_lock():
                repo = self._ready_repo_locked()
                path = Path(repo.working_dir) / self.auth_keys_file_path
                raw = json.loads(path.read_text()) if path.exists() else []
                items = raw.get("items") if isinstance(raw, dict) else raw
                if not isinstance(items, list):
                    raise ValueError("invalid auth key storage")
                before = json.dumps(items, sort_keys=True)
                yield items
                if json.dumps(items, sort_keys=True) != before:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2) + "\n")
                    repo.index.add([self.auth_keys_file_path])
                    repo.index.commit("Update auth key policy or usage")
                    self._push_checked(repo, self.branch)
        except GitCommandError as exc:
            raise GitStorageStateError(self._safe_error(exc)) from None

    def _load_json_file(self, file_path: str) -> list[dict[str, Any]]:
        data = self._load_json_value(file_path)
        return data if isinstance(data, list) else []

    def _load_json_value(self, file_path: str) -> Any:
        with self._checkout_lock():
            repo = self._ready_repo_locked()
            file_full_path = Path(repo.working_dir) / file_path
            if not file_full_path.exists():
                return None
            return json.loads(file_full_path.read_text(encoding="utf-8"))

    def _save_json_file(self, file_path: str, items: Any, message: str) -> None:
        with self._checkout_lock():
            repo = self._ready_repo_locked()
            file_full_path = Path(repo.working_dir) / file_path
            file_full_path.parent.mkdir(parents=True, exist_ok=True)
            file_full_path.write_text(
                json.dumps(items, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            repo.index.add([file_path])
            if repo.is_dirty(untracked_files=True):
                staged_paths = {item.a_path or item.b_path for item in repo.index.diff("HEAD")}
                if staged_paths != {file_path}:
                    raise GitStorageStateError("git checkout contains unrelated staged changes; reconciliation required")
                repo.index.commit(message)
                self._push_checked(repo, self.branch)

    def health_check(self) -> dict[str, Any]:
        """健康检查"""
        try:
            with self._checkout_lock():
                repo = self._ready_repo_locked()
                last_commit = repo.head.commit.hexsha[:8]
            return {
                "status": "healthy",
                "backend": "git",
                "repo_url": self._mask_token(self.repo_url),
                "branch": self.branch,
                "file_path": self.file_path,
                "auth_keys_file_path": self.auth_keys_file_path,
                "last_commit": last_commit,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "git",
                "error": self._safe_error(e),
            }

    def get_backend_info(self) -> dict[str, Any]:
        """获取存储后端信息"""
        return {
            "type": "git",
            "description": "Git 私有仓库存储",
            "repo_url": self._mask_token(self.repo_url),
            "branch": self.branch,
            "file_path": self.file_path,
            "auth_keys_file_path": self.auth_keys_file_path,
        }

    @staticmethod
    def _mask_token(url: str) -> str:
        """隐藏 URL 中的 token"""
        if "@" in url and "://" in url:
            protocol, rest = url.split("://", 1)
            if "@" in rest:
                _, host = rest.split("@", 1)
                return f"{protocol}://****@{host}"
        return url
