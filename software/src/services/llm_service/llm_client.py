"""集中式 LLM 客户端适配器

支持多提供商文本 LLM 调用，按配置加载、不轮番尝试。

使用方法:
    from services.llm_service.llm_client import get_llm_client

    client = get_llm_client()
    response = client.chat_completion(
        messages=[{"role": "user", "content": "你好"}],
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=256,
        top_p=0.9,
    )
    print(response["choices"][0]["message"]["content"])
"""
from typing import Optional, Protocol

from openai import OpenAI

from common.logging import get_logger
from configs.config import get_config

logger = get_logger(__name__)


class LLMClient(Protocol):
    """LLM 客户端协议接口

    所有 LLM 提供商必须实现此接口，供上层统一调用。
    """

    def chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
    ) -> dict:
        """发起聊天补全请求并返回统一格式的 dict。

        Returns:
            {"choices": [{"message": {"content": str, "tool_calls": [...]}}]}
        """
        ...


class OpenAICompatibleClient:
    """OpenAI 兼容端点客户端

    适用于 MiniMax / 火山 Ark / DeepSeek / 通义千问 等 OpenAI 兼容接口。
    调用时会附加 thinking 禁用参数，避免这些平台返回思考过程。
    """

    def __init__(self, api_key: str, base_url: str):
        """初始化客户端

        Args:
            api_key: API 密钥
            base_url: API 基础地址
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
    ) -> dict:
        """调用 OpenAI 兼容接口"""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        # 显式禁用思考功能，提升响应速度并避免 TTS 读出思考过程
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = self._client.chat.completions.create(**kwargs)
        return self._normalize_response(response)

    @staticmethod
    def _normalize_response(response) -> dict:
        """将 OpenAI SDK 响应转换为统一格式"""
        message = response.choices[0].message
        normalized = {
            "choices": [
                {
                    "message": {
                        "content": message.content,
                        "tool_calls": [],
                    }
                }
            ]
        }
        if message.tool_calls:
            for tool_call in message.tool_calls:
                normalized["choices"][0]["message"]["tool_calls"].append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
        return normalized


class OpenAIOfficialClient:
    """OpenAI 官方客户端

    调用官方 OpenAI / GPT 接口，不发送 thinking 等兼容参数。
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        """初始化官方 OpenAI 客户端

        Args:
            api_key: OpenAI API 密钥
            base_url: API 基础地址，默认识别 OpenAI 官方地址
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
    ) -> dict:
        """调用 OpenAI 官方接口"""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        # 官方 OpenAI 不支持 thinking 参数，不附加 extra_body

        response = self._client.chat.completions.create(**kwargs)
        return OpenAICompatibleClient._normalize_response(response)


# 提供商默认配置映射：provider -> (client_class, default_base_url)
_PROVIDER_REGISTRY = {
    "openai": (OpenAIOfficialClient, "https://api.openai.com/v1"),
    "cctq": (OpenAIOfficialClient, "https://www.cctq.ai/v1"),
    "minimax": (OpenAICompatibleClient, "https://api.minimax.chat/v1"),
    "volcano": (OpenAICompatibleClient, "https://ark.cn-beijing.volces.com/api/v3"),
    "ark": (OpenAICompatibleClient, "https://ark.cn-beijing.volces.com/api/v3"),
    "deepseek": (OpenAICompatibleClient, "https://api.deepseek.com/v1"),
    "qwen": (OpenAICompatibleClient, "https://dashscope.aliyuncs.com/compatible-mode/v1"),
}

# 全局 LLM 客户端单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端实例（工厂模式）

    根据 config.llm.provider 自动创建对应提供商的客户端。
    首次调用时创建并缓存，后续调用复用同一实例。

    Returns:
        LLMClient: LLM 客户端实例

    Raises:
        ValueError: 当 provider 未知或缺少 api_key / model 时
    """
    global _llm_client
    if _llm_client is None:
        config = get_config().llm
        provider = (config.provider or "minimax").lower()

        if provider not in _PROVIDER_REGISTRY:
            raise ValueError(
                f"未知的 LLM 提供商: {provider}。"
                f"支持的提供商: {list(_PROVIDER_REGISTRY.keys())}"
            )

        if not config.api_key:
            raise ValueError(
                f"LLM 提供商 {provider} 的 API Key 未配置，"
                f"请在 .env.local 中设置对应的环境变量"
            )
        if not config.model:
            raise ValueError(
                f"LLM 提供商 {provider} 的模型未配置，"
                f"请在 .env.local 中设置模型名称"
            )

        client_class, default_url = _PROVIDER_REGISTRY[provider]
        base_url = config.api_url or default_url
        _llm_client = client_class(api_key=config.api_key, base_url=base_url)
        logger.info(f"LLM 客户端初始化完成: provider={provider}, model={config.model}, base_url={base_url}")

    return _llm_client


def reset_llm_client() -> None:
    """重置全局 LLM 客户端单例

    主要用于测试，允许在运行时重新加载配置后创建新客户端。
    """
    global _llm_client
    _llm_client = None
    logger.debug("LLM 客户端单例已重置")
