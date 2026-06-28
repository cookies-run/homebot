"""LLM 客户端适配器单元测试"""
import os
from unittest.mock import MagicMock, patch

import pytest

from services.llm_service.llm_client import (
    OpenAICompatibleClient,
    OpenAIOfficialClient,
    get_llm_client,
    reset_llm_client,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """每个测试前重置 LLM 客户端单例，避免状态泄漏"""
    reset_llm_client()
    yield
    reset_llm_client()


def _make_mock_openai_response(content: str = "hi", tool_calls=None):
    """构造一个模拟的 OpenAI SDK 响应对象"""
    response = MagicMock()
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    response.choices = [choice]
    return response


def test_openai_official_client_omits_extra_body():
    """官方 OpenAI 客户端不应发送 thinking 等 extra_body 参数"""
    with patch("services.llm_service.llm_client.OpenAI") as MockOpenAI:
        mock_instance = MagicMock()
        MockOpenAI.return_value = mock_instance
        mock_instance.chat.completions.create.return_value = _make_mock_openai_response(
            content="你好"
        )

        client = OpenAIOfficialClient(api_key="sk-test")
        result = client.chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=256,
            top_p=0.9,
        )

        kwargs = mock_instance.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs
        assert result["choices"][0]["message"]["content"] == "你好"


def test_openai_compatible_client_includes_extra_body():
    """OpenAI 兼容客户端应发送 thinking 禁用参数"""
    with patch("services.llm_service.llm_client.OpenAI") as MockOpenAI:
        mock_instance = MagicMock()
        MockOpenAI.return_value = mock_instance
        mock_instance.chat.completions.create.return_value = _make_mock_openai_response(
            content="你好"
        )

        client = OpenAICompatibleClient(
            api_key="sk-test", base_url="https://api.minimax.chat/v1"
        )
        client.chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            model="MiniMax-M2.7",
            temperature=0.1,
            max_tokens=256,
            top_p=0.9,
        )

        kwargs = mock_instance.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_openai_compatible_client_with_tools():
    """OpenAI 兼容客户端支持工具调用"""
    with patch("services.llm_service.llm_client.OpenAI") as MockOpenAI:
        mock_instance = MagicMock()
        MockOpenAI.return_value = mock_instance
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.type = "function"
        tool_call.function.name = "move_forward"
        tool_call.function.arguments = '{"distance": 0.5}'
        mock_instance.chat.completions.create.return_value = _make_mock_openai_response(
            content="好的", tool_calls=[tool_call]
        )

        client = OpenAICompatibleClient(
            api_key="sk-test", base_url="https://api.minimax.chat/v1"
        )
        result = client.chat_completion(
            messages=[{"role": "user", "content": "前进"}],
            model="MiniMax-M2.7",
            temperature=0.1,
            max_tokens=256,
            top_p=0.9,
            tools=[{"type": "function", "function": {"name": "move_forward"}}],
            tool_choice="auto",
        )

        kwargs = mock_instance.chat.completions.create.call_args.kwargs
        assert kwargs["tools"] is not None
        assert kwargs["tool_choice"] == "auto"
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "move_forward"


@patch("services.llm_service.llm_client.get_config")
def test_get_llm_client_returns_openai(mock_get_config):
    """工厂根据 provider=openai 返回 OpenAIOfficialClient"""
    cfg = MagicMock()
    cfg.llm.provider = "openai"
    cfg.llm.api_key = "sk-test"
    cfg.llm.api_url = ""
    cfg.llm.model = "gpt-4o-mini"
    mock_get_config.return_value = cfg

    client = get_llm_client()
    assert isinstance(client, OpenAIOfficialClient)


@patch("services.llm_service.llm_client.get_config")
def test_get_llm_client_returns_minimax(mock_get_config):
    """工厂根据 provider=minimax 返回 OpenAICompatibleClient"""
    cfg = MagicMock()
    cfg.llm.provider = "minimax"
    cfg.llm.api_key = "sk-test"
    cfg.llm.api_url = ""
    cfg.llm.model = "MiniMax-M2.7"
    mock_get_config.return_value = cfg

    client = get_llm_client()
    assert isinstance(client, OpenAICompatibleClient)


@patch("services.llm_service.llm_client.get_config")
def test_get_llm_client_caches_singleton(mock_get_config):
    """工厂应缓存单例实例"""
    cfg = MagicMock()
    cfg.llm.provider = "openai"
    cfg.llm.api_key = "sk-test"
    cfg.llm.api_url = ""
    cfg.llm.model = "gpt-4o-mini"
    mock_get_config.return_value = cfg

    client1 = get_llm_client()
    client2 = get_llm_client()
    assert client1 is client2


@patch("services.llm_service.llm_client.get_config")
def test_get_llm_client_unknown_provider_raises(mock_get_config):
    """未知 provider 应抛出 ValueError"""
    cfg = MagicMock()
    cfg.llm.provider = "unknown"
    cfg.llm.api_key = "sk-test"
    cfg.llm.model = "some-model"
    mock_get_config.return_value = cfg

    with pytest.raises(ValueError, match="未知的 LLM 提供商"):
        get_llm_client()


@patch("services.llm_service.llm_client.get_config")
def test_get_llm_client_missing_api_key_raises(mock_get_config):
    """缺少 API Key 应抛出 ValueError"""
    cfg = MagicMock()
    cfg.llm.provider = "openai"
    cfg.llm.api_key = ""
    cfg.llm.model = "gpt-4o-mini"
    mock_get_config.return_value = cfg

    with pytest.raises(ValueError, match="API Key 未配置"):
        get_llm_client()


@patch("services.llm_service.llm_client.get_config")
def test_get_llm_client_missing_model_raises(mock_get_config):
    """缺少模型名称应抛出 ValueError"""
    cfg = MagicMock()
    cfg.llm.provider = "openai"
    cfg.llm.api_key = "sk-test"
    cfg.llm.model = ""
    mock_get_config.return_value = cfg

    with pytest.raises(ValueError, match="模型未配置"):
        get_llm_client()


@patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "gpt-4o-mini"}, clear=False)
def test_config_reads_llm_provider_from_env():
    """LLMConfig 应能从环境变量读取 provider"""
    # 先重置配置和密钥单例，否则可能读到 .env.local 的缓存
    from configs.config import set_config
    from configs.ai_config import reload_ai_credentials
    set_config(None)
    reload_ai_credentials()

    from configs.config import get_config
    cfg = get_config().llm
    assert cfg.provider == "openai"
    assert cfg.api_key == "sk-test"
    assert cfg.model == "gpt-4o-mini"

    # 恢复默认配置，避免影响其他测试
    set_config(None)
    reload_ai_credentials()
