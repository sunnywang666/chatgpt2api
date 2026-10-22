"""Credential request boundaries must fail closed on certificate errors.

Controlled HTTP fixtures only; no credentials or real authorization requests.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from curl_cffi.requests.exceptions import SSLError


class CertificateCheckingSession:
    def __init__(self, fail_at, **kwargs):
        self.verify = kwargs.get("verify", True)
        self.fail_at = fail_at
        self.calls = []
        self.cookies = SimpleNamespace(set=Mock())
        self.close = Mock()

    def _request(self, method, url, **kwargs):
        assert kwargs.get("verify", self.verify) is True, "credential request disabled TLS verification"
        self.calls.append((method, url))
        if self.fail_at in url:
            raise SSLError("certificate verify failed", code=60)
        assert "password/verify" in url or "authorize?" in url
        return SimpleNamespace(status_code=200, url=url, text="{}", json=lambda: {
            "continue_url": "https://platform.openai.com/auth/callback?code=fixture-code",
        })

    def get(self, url, **kwargs):
        return self._request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._request("POST", url, **kwargs)


@pytest.mark.parametrize("failure_path,request_count", [("authorize?", 1), ("oauth/token", 3)])
def test_password_login_does_not_accept_an_untrusted_server(monkeypatch, failure_path, request_count):
    from curl_cffi import requests
    from services.account_service import AccountService, config
    from utils import sentinel
    sessions = []

    def factory(**kwargs):
        session = CertificateCheckingSession(failure_path, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(requests, "Session", factory)
    monkeypatch.setattr(config, "get_proxy_settings", lambda: "")
    monkeypatch.setattr(sentinel, "build_sentinel_token", lambda *a, **k: ("fixture", ""))
    service = object.__new__(AccountService)
    with pytest.raises(SSLError):
        service._login_with_password("fixture@example.invalid", "fixture-password")
    assert len(sessions) == 1
    assert len(sessions[0].calls) == request_count
    sessions[0].close.assert_called_once()


def test_sentinel_does_not_override_session_verification():
    from utils.sentinel import build_sentinel_token
    session = CertificateCheckingSession("sentinel/req", verify=True)
    with pytest.raises(SSLError):
        build_sentinel_token(session, "fixture-device", "password_verify")
    assert len(session.calls) == 1


def test_manual_oauth_keeps_original_session_on_certificate_failure(monkeypatch):
    from services import oauth_login_service as module
    sessions = []

    def factory(**kwargs):
        session = CertificateCheckingSession("oauth/token", **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(module.requests, "Session", factory)
    monkeypatch.setattr(module.proxy_settings, "build_session_kwargs", lambda **kwargs: {
        **kwargs, "verify": False,
    })
    service = module.OAuthLoginService()
    started = service.start()
    sid = started["session_id"]
    original = dict(service._sessions[sid])
    with pytest.raises(module.OAuthLoginError, match="certificate verify failed"):
        service.finish(sid, "fixture-code")
    assert service._sessions[sid] == original
    assert len(sessions) == 1 and len(sessions[0].calls) == 1
    sessions[0].close.assert_called_once()
