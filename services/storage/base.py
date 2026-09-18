from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any


class AccountCommitUncertain(RuntimeError):
    """Account replacement may have happened; reconcile before any further save."""


class StorageBackend(ABC):
    """抽象存储后端基类"""

    @abstractmethod
    def load_accounts(self) -> list[dict[str, Any]]:
        """加载所有账号数据"""
        pass

    @abstractmethod
    def save_accounts(self, accounts: list[dict[str, Any]]) -> None:
        """保存所有账号数据"""
        pass

    def confirm_accounts_commit(self) -> list[dict[str, Any]]:
        """Read the committed account snapshot for an interrupted operation."""
        return self.load_accounts()

    @abstractmethod
    def load_auth_keys(self) -> list[dict[str, Any]]:
        """加载所有鉴权密钥数据"""
        pass

    @abstractmethod
    def save_auth_keys(self, auth_keys: list[dict[str, Any]]) -> None:
        """保存所有鉴权密钥数据"""
        pass

    @abstractmethod
    def auth_keys_transaction(self) -> AbstractContextManager[list[dict[str, Any]]]:
        """Lock, load and atomically persist key records across service processes.

        Changes are committed only when the context exits normally.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """健康检查，返回存储后端状态"""
        pass

    @abstractmethod
    def get_backend_info(self) -> dict[str, Any]:
        """获取存储后端信息"""
        pass
