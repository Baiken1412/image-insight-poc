"""Tests for RemoteQwenTransport (image_insight/vlm/qwen_client.py) — the
opt-in QWEN_BACKEND=remote path that calls an OpenAI-compatible
chat-completions endpoint instead of loading local GPU weights.

requests.post is monkeypatched rather than hitting a real HTTP server —
these tests only verify the request this transport builds and how it maps
a response (or a network failure) back into the ChatCompletionTransport
shape QwenVisionAnalyzer expects, mirroring how tests/test_qwen_client.py's
FakeTransport substitutes for LocalQwenTransport.
"""
from __future__ import annotations

import base64
import io

import pytest
import requests
from PIL import Image

from image_insight.vlm.qwen_client import RemoteQwenTransport


def _tiny_image_base64() -> str:
    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self) -> dict:
        return self._json_data


def _ok_response(content: str = "{}", finish_reason: str = "stop", usage: dict | None = None) -> FakeResponse:
    return FakeResponse(
        {
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )


def test_complete_posts_openai_style_payload_and_returns_message_content(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return _ok_response(content='{"items": []}')

    monkeypatch.setattr(requests, "post", fake_post)

    transport = RemoteQwenTransport(
        base_url="http://localhost:8000/v1",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        api_key="secret-token",
        max_new_tokens=512,
        request_timeout=60,
    )

    result = transport.complete(image_base64=_tiny_image_base64(), system_prompt="describe this photo")

    assert result == '{"items": []}'
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["timeout"] == 60
    body = captured["json"]
    assert body["model"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert body["max_tokens"] == 512
    content_parts = body["messages"][0]["content"]
    assert content_parts[0]["type"] == "image_url"
    assert content_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content_parts[1] == {"type": "text", "text": "describe this photo"}


def test_complete_records_last_usage_from_response(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: _ok_response(usage={"prompt_tokens": 123, "completion_tokens": 45}, finish_reason="length"),
    )
    transport = RemoteQwenTransport(base_url="http://host/v1", model_id="m")

    transport.complete(image_base64=_tiny_image_base64(), system_prompt="x")

    assert transport.last_usage.prompt_tokens == 123
    assert transport.last_usage.completion_tokens == 45
    assert transport.last_usage.finish_reason == "length"


def test_complete_per_call_overrides_take_priority_over_constructor_defaults(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(json=json, timeout=timeout)
        return _ok_response()

    monkeypatch.setattr(requests, "post", fake_post)
    transport = RemoteQwenTransport(base_url="http://host/v1", model_id="m", max_new_tokens=100, request_timeout=30)

    transport.complete(
        image_base64=_tiny_image_base64(),
        system_prompt="x",
        max_new_tokens=999,
        timeout_seconds=5,
        repetition_penalty=1.2,
    )

    assert captured["json"]["max_tokens"] == 999
    assert captured["json"]["repetition_penalty"] == 1.2
    assert captured["timeout"] == 5


def test_complete_omits_authorization_header_when_no_api_key(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["headers"] = headers
        return _ok_response()

    monkeypatch.setattr(requests, "post", fake_post)
    transport = RemoteQwenTransport(base_url="http://host/v1", model_id="m")

    transport.complete(image_base64=_tiny_image_base64(), system_prompt="x")

    assert "Authorization" not in captured["headers"]


def test_complete_raises_timeout_error_on_requests_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout("boom")

    monkeypatch.setattr(requests, "post", fake_post)
    transport = RemoteQwenTransport(base_url="http://host/v1", model_id="m", request_timeout=30)

    with pytest.raises(TimeoutError):
        transport.complete(image_base64=_tiny_image_base64(), system_prompt="x")


def test_complete_raises_runtime_error_on_connection_failure(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", fake_post)
    transport = RemoteQwenTransport(base_url="http://host/v1", model_id="m")

    with pytest.raises(RuntimeError):
        transport.complete(image_base64=_tiny_image_base64(), system_prompt="x")


def test_complete_raises_runtime_error_on_http_error_status(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({}, status_code=500))
    transport = RemoteQwenTransport(base_url="http://host/v1", model_id="m")

    with pytest.raises(RuntimeError):
        transport.complete(image_base64=_tiny_image_base64(), system_prompt="x")
