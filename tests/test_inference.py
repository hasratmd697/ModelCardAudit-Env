import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


def test_call_llm_proxy_returns_fallback_on_exception():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(side_effect=RuntimeError("proxy unavailable"))
            )
        )
    )
    obs = {
        "task_description": "Audit this model card.",
        "model_card_metadata": {"model_name": "DemoModel"},
    }

    with patch("inference.time.sleep"):
        response = inference.call_llm_proxy(client, "basic_completeness", obs)

    payload = json.loads(response)
    assert payload["task"] == "basic_completeness"
    assert payload["model"] == "DemoModel"
    assert payload["status"] == "proxy_unavailable"


def test_run_task_deterministic_returns_zero_when_reset_fails():
    with patch("inference.request_json", return_value=(None, "connection failed")):
        score = inference.run_task_deterministic("basic_completeness")

    assert score == 0.0
