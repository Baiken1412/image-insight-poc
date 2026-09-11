"""Tests for the QWEN_BACKEND local/remote selection added to
image_insight/config.py — image_insight/api/main.py's lifespan reads these
to decide whether to build a LocalQwenTransport or a RemoteQwenTransport.
"""
from __future__ import annotations

import pytest

from image_insight.config import load_qwen_backend, load_remote_qwen_config


def test_load_qwen_backend_defaults_to_local(monkeypatch):
    monkeypatch.delenv("QWEN_BACKEND", raising=False)

    assert load_qwen_backend() == "local"


def test_load_qwen_backend_accepts_remote(monkeypatch):
    monkeypatch.setenv("QWEN_BACKEND", "remote")

    assert load_qwen_backend() == "remote"


def test_load_qwen_backend_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("QWEN_BACKEND", "REMOTE")

    assert load_qwen_backend() == "remote"


def test_load_qwen_backend_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("QWEN_BACKEND", "cloud")

    with pytest.raises(ValueError):
        load_qwen_backend()


def test_load_remote_qwen_config_returns_none_when_base_url_unset(monkeypatch):
    monkeypatch.delenv("QWEN_REMOTE_BASE_URL", raising=False)

    assert load_remote_qwen_config() is None


def test_load_remote_qwen_config_reads_all_fields(monkeypatch):
    monkeypatch.setenv("QWEN_REMOTE_BASE_URL", "http://gpu-host:8000/v1")
    monkeypatch.setenv("QWEN_REMOTE_MODEL_ID", "custom-model")
    monkeypatch.setenv("QWEN_REMOTE_API_KEY", "secret-token")
    monkeypatch.setenv("QWEN_REMOTE_REQUEST_TIMEOUT", "45")

    config = load_remote_qwen_config()

    assert config.base_url == "http://gpu-host:8000/v1"
    assert config.model_id == "custom-model"
    assert config.api_key == "secret-token"
    assert config.request_timeout == 45


def test_load_remote_qwen_config_falls_back_to_qwen_model_id(monkeypatch):
    monkeypatch.setenv("QWEN_REMOTE_BASE_URL", "http://gpu-host:8000/v1")
    monkeypatch.delenv("QWEN_REMOTE_MODEL_ID", raising=False)
    monkeypatch.setenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")

    config = load_remote_qwen_config()

    assert config.model_id == "Qwen/Qwen3-VL-4B-Instruct"


def test_load_remote_qwen_config_defaults_api_key_to_none(monkeypatch):
    monkeypatch.setenv("QWEN_REMOTE_BASE_URL", "http://gpu-host:8000/v1")
    monkeypatch.delenv("QWEN_REMOTE_API_KEY", raising=False)

    config = load_remote_qwen_config()

    assert config.api_key is None
