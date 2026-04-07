from unittest.mock import patch

import inference


def test_proxy_credentials_available(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("API_KEY", "secret")

    assert inference.proxy_credentials_available() is True


def test_proxy_credentials_missing_key(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")
    monkeypatch.delenv("API_KEY", raising=False)

    assert inference.proxy_credentials_available() is False


def test_build_proxy_client_uses_injected_env(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("API_KEY", "secret")

    with patch("inference.OpenAI") as mock_openai:
        inference.build_proxy_client()

    mock_openai.assert_called_once_with(
        base_url="https://proxy.example/v1",
        api_key="secret",
    )


def test_get_env_server_url_default(monkeypatch):
    monkeypatch.delenv("ENV_API_URL", raising=False)

    assert inference.get_env_server_url() == "http://localhost:7860"
