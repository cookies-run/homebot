# -*- coding: utf-8 -*-
"""外部模型服务配置模块 - 加载 TTS / LLM / Vision 各服务的接入配置

本模块本身不存储任何密钥值：provider / api_key / api_url / model 等
均在运行时从环境变量或 .env.local 文件加载（确保密钥不会提交到版本控制）。
"""
import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from common.logging import get_logger

logger = get_logger(__name__)


# 项目根目录（software/）
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_env_file(env_path: Path) -> None:
    """加载.env文件到环境变量
    
    Args:
        env_path: .env文件路径
    """
    if not env_path.exists():
        return
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                # 解析KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')  # 去除引号
                    # 只设置尚未存在的环境变量（允许用户通过系统环境变量覆盖）
                    if key not in os.environ:
                        os.environ[key] = value
        logger.debug(f"已加载环境变量文件: {env_path}")
    except Exception as e:
        logger.warning(f"加载环境变量文件失败: {e}")


def _load_all_env_files() -> None:
    """按优先级加载所有环境变量文件"""
    # 优先级：.env.local > .env.development > .env.production > .env
    env_files = [
        PROJECT_ROOT / ".env.local",
        PROJECT_ROOT / ".env.development",
        PROJECT_ROOT / ".env.production",
        PROJECT_ROOT / ".env",
    ]
    
    for env_file in env_files:
        if env_file.exists():
            _load_env_file(env_file)


# 模块加载时自动执行
_load_all_env_files()


@dataclass
class TTSCredentials:
    """TTS 服务配置（支持多提供商）"""
    # 火山引擎专用
    appid: str = ""
    access_token: str = ""
    # 通用字段（MiniMax 等复用）
    api_key: str = ""
    api_url: str = ""
    model: str = ""
    # 非敏感配置覆盖
    resource_id: str = "seed-tts-2.0"
    voice_type: str = "zh_female_vv_uranus_bigtts"


@dataclass
class LLMCredentials:
    """LLM 服务配置"""
    provider: str = ""                        # 提供商: minimax/volcano/deepseek/openai/qwen
    api_key: str = ""
    api_url: str = ""                         # 空时由 LLMConfig 按 provider 填充默认值
    model: str = ""  # 火山Ark需要填写模型ID，如 ep-20250324123456-abcdef


@dataclass
class VisionCredentials:
    """图片理解/Vision 服务配置"""
    provider: str = "deepseek"  # deepseek/qwen/openai
    api_key: str = ""
    api_url: str = ""
    model: str = ""


@dataclass
class AICredentials:
    """所有外部模型服务配置"""
    tts: TTSCredentials = field(default_factory=TTSCredentials)
    llm: LLMCredentials = field(default_factory=LLMCredentials)
    vision: VisionCredentials = field(default_factory=VisionCredentials)


# 全局服务配置实例
_ai_credentials_instance: Optional[AICredentials] = None


def _get_env(key: str, default: str = "") -> str:
    """安全获取环境变量
    
    Args:
        key: 环境变量名
        default: 默认值
        
    Returns:
        环境变量值或默认值
    """
    return os.environ.get(key, default)


def _mask_key(key: str, visible_start: int = 4, visible_end: int = 4) -> str:
    """脱敏显示API密钥
    
    Args:
        key: 原始密钥
        visible_start: 开头显示的字符数
        visible_end: 结尾显示的字符数
        
    Returns:
        脱敏后的密钥，如 sk-****1234
    """
    if len(key) <= visible_start + visible_end:
        return "*" * len(key) if key else "(未设置)"
    return f"{key[:visible_start]}****{key[-visible_end:]}"


def load_ai_credentials() -> AICredentials:
    """加载所有密钥配置
    
    从环境变量加载敏感配置，支持多种命名方式
    
    Returns:
        AICredentials: 密钥配置对象
    """
    # TTS 配置 - 支持多种环境变量名
    tts = TTSCredentials(
        appid=_get_env("VOLCANO_APPID", _get_env("TTS_APPID", "")),
        access_token=_get_env("VOLCANO_ACCESS_TOKEN", _get_env("TTS_ACCESS_TOKEN", "")),
        api_key=_get_env("MINIMAX_TTS_API_KEY", _get_env("TTS_API_KEY", "")),
        api_url=_get_env("MINIMAX_TTS_API_URL", _get_env("TTS_API_URL", "")),
        model=_get_env("MINIMAX_TTS_MODEL", _get_env("TTS_MODEL", "")),
        resource_id=_get_env("VOLCANO_RESOURCE_ID", "seed-tts-2.0"),
        voice_type=_get_env("VOLCANO_VOICE_TYPE", _get_env("TTS_VOICE_TYPE", "zh_female_vv_uranus_bigtts")),
    )
    
    # LLM 配置 - 优先使用火山Ark，兼容DeepSeek / OpenAI / MiniMax
    llm = LLMCredentials(
        provider=_get_env("LLM_PROVIDER", ""),
        api_key=_get_env("ARK_API_KEY", _get_env("VOLCANO_API_KEY", _get_env("DEEPSEEK_API_KEY", _get_env("OPENAI_API_KEY", _get_env("LLM_API_KEY", ""))))),
        api_url=_get_env("ARK_API_URL", _get_env("OPENAI_API_URL", _get_env("LLM_API_URL", "https://ark.cn-beijing.volces.com/api/v3"))),
        model=_get_env("ARK_MODEL_ID", _get_env("VOLCANO_MODEL_ID", _get_env("DEEPSEEK_MODEL", _get_env("OPENAI_MODEL", _get_env("LLM_MODEL", ""))))),
    )
    
    # Vision 配置
    vision_provider = _get_env("VISION_PROVIDER", "deepseek")
    
    # 根据provider获取对应的配置
    vision_api_key = _get_env("VISION_API_KEY", "")
    vision_api_url = _get_env("VISION_API_URL", "")
    vision_model = _get_env("VISION_MODEL", "")
    
    # 如果没有单独设置VISION配置，使用LLM的配置
    if not vision_api_key and vision_provider == "deepseek":
        vision_api_key = llm.api_key
        vision_api_url = llm.api_url or "https://api.deepseek.com/v1"
        vision_model = vision_model or "deepseek-chat"
    
    vision = VisionCredentials(
        provider=vision_provider,
        api_key=vision_api_key,
        api_url=vision_api_url,
        model=vision_model,
    )
    
    return AICredentials(tts=tts, llm=llm, vision=vision)


def get_ai_credentials() -> AICredentials:
    """获取全局密钥实例
    
    Returns:
        AICredentials: 全局密钥配置实例
    """
    global _ai_credentials_instance
    if _ai_credentials_instance is None:
        _ai_credentials_instance = load_ai_credentials()
    return _ai_credentials_instance


def reload_ai_credentials() -> AICredentials:
    """重新加载密钥配置
    
    用于在运行时重新读取环境变量
    
    Returns:
        AICredentials: 新的密钥配置实例
    """
    global _ai_credentials_instance
    _load_all_env_files()
    _ai_credentials_instance = load_ai_credentials()
    return _ai_credentials_instance


def check_ai_credentials(verbose: bool = True) -> dict:
    """检查密钥配置状态
    
    检查各项密钥是否已配置，并返回状态报告
    
    Args:
        verbose: 是否打印详细信息
        
    Returns:
        dict: 各服务的配置状态
    """
    creds = get_ai_credentials()
    
    status = {
        "tts": {
            "configured": bool(creds.tts.appid and creds.tts.access_token),
            "appid": creds.tts.appid[:6] + "..." if creds.tts.appid else "(未设置)",
            "access_token": _mask_key(creds.tts.access_token),
        },
        "llm": {
            "configured": bool(creds.llm.api_key),
            "provider": creds.llm.provider or "(未设置)",
            "api_key": _mask_key(creds.llm.api_key),
            "model": creds.llm.model,
        },
        "vision": {
            "configured": bool(creds.vision.api_key),
            "provider": creds.vision.provider,
            "api_key": _mask_key(creds.vision.api_key),
        },
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print("HomeBot API Key Configuration Status")
        print("=" * 50)
        
        # TTS 状态
        tts_ok = status["tts"]["configured"]
        status_icon = "[OK]" if tts_ok else "[MISSING]"
        print(f"\n[TTS] 火山引擎 TTS: {status_icon}")
        if tts_ok:
            print(f"   AppID: {status['tts']['appid']}")
            print(f"   Access Token: {status['tts']['access_token']}")
        else:
            print("   [提示] 设置 VOLCANO_APPID 和 VOLCANO_ACCESS_TOKEN 环境变量")
        
        # LLM 状态
        llm_ok = status["llm"]["configured"]
        status_icon = "[OK]" if llm_ok else "[MISSING]"
        print(f"\n[LLM] 大语言模型: {status_icon}")
        if llm_ok:
            print(f"   Provider: {status['llm']['provider']}")
            print(f"   API Key: {status['llm']['api_key']}")
            print(f"   Model: {status['llm']['model'] if status['llm']['model'] else '(未设置，必填)'}")
        else:
            print("   [提示] 设置 LLM_API_KEY 和 LLM_MODEL 环境变量")
            print("   支持: OpenAI / MiniMax / 火山Ark / DeepSeek / 通义千问 等 OpenAI 兼容接口")
        
        # Vision 状态
        vision_ok = status["vision"]["configured"]
        status_icon = "[OK]" if vision_ok else "[MISSING]"
        print(f"\n[Vision] 图片理解: {status_icon}")
        print(f"   Provider: {status['vision']['provider']}")
        if vision_ok:
            print(f"   API Key: {status['vision']['api_key']}")
        elif status["vision"]["provider"] == "deepseek" and llm_ok:
            print("   [提示] 将复用 DeepSeek LLM 的配置")
            status["vision"]["configured"] = True  # 复用LLM配置
        else:
            print("   [提示] 设置 VISION_API_KEY 环境变量")
        
        # 配置文件路径提示
        local_env = PROJECT_ROOT / ".env.local"
        print(f"\n[文件] 配置文件路径: {local_env}")
        if not local_env.exists():
            example_env = PROJECT_ROOT / ".env.example"
            print(f"[提示] 复制 {example_env.name} 为 {local_env.name} 并填入密钥")
        
        print("=" * 50 + "\n")
    
    return status


def require_ai_credentials(service: str) -> None:
    """检查特定服务的密钥是否已配置
    
    如果未配置，打印帮助信息并退出程序
    
    Args:
        service: 服务名称 (tts/llm/vision)
    """
    creds = get_ai_credentials()
    
    if service == "tts":
        if not (creds.tts.appid and creds.tts.access_token):
            logger.error("火山引擎 TTS 密钥未配置")
            print("\n[错误] 火山引擎 TTS 密钥未配置")
            print("\n请设置以下环境变量之一:")
            print("  1. VOLCANO_APPID and VOLCANO_ACCESS_TOKEN")
            print("  2. TTS_APPID and TTS_ACCESS_TOKEN")
            print(f"\n或创建 {PROJECT_ROOT / '.env.local'} 文件，格式如下:")
            print("  VOLCANO_APPID=your_appid")
            print("  VOLCANO_ACCESS_TOKEN=your_token")
            sys.exit(1)
    
    elif service == "llm":
        if not creds.llm.api_key:
            logger.error("LLM API Key 未配置")
            print("\n[错误] LLM API Key 未配置")
            print("\n请设置以下环境变量之一:")
            print("  OPENAI_API_KEY=sk-your_api_key")
            print("  ARK_API_KEY=your_api_key")
            print("  DEEPSEEK_API_KEY=your_api_key")
            print("  LLM_API_KEY=your_api_key")
            print(f"\n或创建 {PROJECT_ROOT / '.env.local'} 文件，格式如下:")
            print("  LLM_PROVIDER=openai")
            print("  OPENAI_API_KEY=sk-your_api_key")
            print("  OPENAI_API_URL=https://api.openai.com/v1")
            print("  OPENAI_MODEL=gpt-4o-mini")
            sys.exit(1)
        if not creds.llm.model:
            logger.error("LLM 模型ID未配置")
            print("\n[错误] LLM 模型ID未配置")
            print("\n请在 .env.local 文件中设置 OPENAI_MODEL 或 LLM_MODEL:")
            print("  OPENAI_MODEL=gpt-4o-mini")
            print("  LLM_MODEL=MiniMax-M2.7-highspeed")
            sys.exit(1)
    
    elif service == "vision":
        # Vision 可以复用 DeepSeek 的配置
        if not creds.vision.api_key:
            if creds.llm.api_key and creds.vision.provider == "deepseek":
                return  # 允许复用LLM配置
            logger.error("Vision API Key 未配置")
            print("\n[错误] Vision API Key 未配置")
            print("\n请设置以下环境变量:")
            print("  VISION_API_KEY")
            print(f"\n或创建 {PROJECT_ROOT / '.env.local'} 文件")
            sys.exit(1)


if __name__ == "__main__":
    # 命令行检查密钥配置
    check_ai_credentials(verbose=True)


# ==================== 模型行为配置（behavior + 凭证合并） ====================

@dataclass
class TTSConfig:
    """TTS 配置（支持多提供商）

    支持火山引擎(volcano)和 MiniMax 等开放式 TTS 提供商。
    敏感信息从环境变量/.env.local 加载（见 get_ai_credentials），不在此硬编码。
    """
    provider: str = "volcano"                 # 提供商: volcano / minimax / ...
    # 火山引擎专用配置
    appid: str = ""                           # 应用ID
    access_token: str = ""                    # 访问令牌
    resource_id: str = "seed-tts-2.0"         # 资源ID
    endpoint: str = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
    # 通用配置（MiniMax 等也复用）
    api_key: str = ""                         # 通用 API Key
    api_url: str = ""                         # 通用 API 地址
    model: str = ""                           # 通用模型名称
    voice_type: str = "zh_female_vv_uranus_bigtts"  # 音色类型 / voice_id
    encoding: str = "pcm"                     # 音频编码
    sample_rate: int = 16000                  # 输出采样率

    def __post_init__(self):
        """从密钥管理加载敏感配置"""
        import os
        creds = get_ai_credentials()
        # provider 支持从环境变量覆盖
        env_provider = os.environ.get("TTS_PROVIDER")
        if env_provider:
            self.provider = env_provider
        # 火山引擎密钥
        if not self.appid:
            self.appid = creds.tts.appid
        if not self.access_token:
            self.access_token = creds.tts.access_token
        # 通用密钥（MiniMax 等）
        if not self.api_key:
            self.api_key = creds.tts.api_key
        if not self.api_url:
            self.api_url = creds.tts.api_url
        if not self.model:
            self.model = creds.tts.model
        # 非敏感配置也可以从环境变量覆盖
        if creds.tts.resource_id:
            self.resource_id = creds.tts.resource_id
        if creds.tts.voice_type:
            self.voice_type = creds.tts.voice_type


@dataclass
class LLMConfig:
    """LLM API配置

    敏感信息（api_key）从环境变量/.env.local 加载（见 get_ai_credentials）
    支持火山Ark、DeepSeek、MiniMax 等 OpenAI 兼容接口
    如需修改，请在 .env.local 文件中设置
    """
    provider: str = "minimax"                 # 提供商: minimax/volcano/deepseek/qwen/openai
    api_key: str = ""                         # API密钥
    api_url: str = ""                         # API地址（空时按 provider 使用默认值）
    model: str = ""                           # 模型名称
    temperature: float = 0.1                  # 温度参数（低温度=更确定性回复，响应更快）
    max_tokens: int = 256                     # 最大token数（限制回复长度，提升速度）
    top_p: float = 0.9                        # 核采样（控制输出多样性）

    def __post_init__(self):
        """从密钥管理加载敏感配置，并应用提供商特定默认值"""
        import os
        creds = get_ai_credentials()
        # provider 支持从环境变量覆盖
        env_provider = os.environ.get("LLM_PROVIDER") or creds.llm.provider
        if env_provider:
            self.provider = env_provider
        if not self.api_key:
            self.api_key = creds.llm.api_key
        # 非敏感配置可以从环境变量/密钥覆盖
        if creds.llm.api_url:
            self.api_url = creds.llm.api_url
        if creds.llm.model:
            self.model = creds.llm.model
        # 如果 api_url 仍为空，按 provider 使用默认地址
        if not self.api_url:
            provider_defaults = {
                "openai": "https://api.openai.com/v1",
                "minimax": "https://api.minimax.chat/v1",
                "volcano": "https://ark.cn-beijing.volces.com/api/v3",
                "ark": "https://ark.cn-beijing.volces.com/api/v3",
                "deepseek": "https://api.deepseek.com/v1",
                "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }
            self.api_url = provider_defaults.get(
                self.provider, "https://api.minimax.chat/v1"
            )
        else:
            # 避免切换 provider 时，旧的通用 LLM_API_URL 指向另一家平台导致请求失败
            provider_defaults = {
                "openai": "https://api.openai.com/v1",
                "minimax": "https://api.minimax.chat/v1",
                "volcano": "https://ark.cn-beijing.volces.com/api/v3",
                "ark": "https://ark.cn-beijing.volces.com/api/v3",
                "deepseek": "https://api.deepseek.com/v1",
                "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }
            url_to_providers = {
                "https://api.minimax.chat/v1": ["minimax"],
                "https://ark.cn-beijing.volces.com/api/v3": ["volcano", "ark"],
                "https://api.deepseek.com/v1": ["deepseek"],
                "https://dashscope.aliyuncs.com/compatible-mode/v1": ["qwen"],
                "https://api.openai.com/v1": ["openai"],
            }
            normalized_url = self.api_url.rstrip("/")
            matched_providers = None
            for known_url, providers in url_to_providers.items():
                if normalized_url == known_url.rstrip("/"):
                    matched_providers = providers
                    break
            if matched_providers and self.provider not in matched_providers:
                new_url = provider_defaults.get(
                    self.provider, "https://api.minimax.chat/v1"
                )
                logger.warning(
                    f"检测到 api_url ({self.api_url}) 与 provider ({self.provider}) "
                    f"不匹配，已自动切换为默认地址: {new_url}"
                )
                self.api_url = new_url
        # 如果没有配置model，给出警告
        if not self.model:
            logger.warning("LLM模型未配置，请在.env.local中设置 LLM_MODEL 或 ARK_MODEL_ID")


@dataclass
class VisionConfig:
    """图片理解/Vision API配置
    
    支持多提供商: deepseek/qwen/openai
    敏感信息从环境变量/.env.local 加载（见 get_ai_credentials）
    """
    provider: str = "deepseek"                # 提供商
    api_key: str = ""                         # API密钥
    api_url: str = ""                         # API地址
    model: str = ""                           # 模型名称
    temperature: float = 0.7                  # 温度参数
    max_tokens: int = 1024                    # 最大token数
    
    def __post_init__(self):
        """从密钥管理加载配置"""
        creds = get_ai_credentials()
        
        # 如果未指定provider，使用环境变量的配置
        env_provider = creds.vision.provider
        if env_provider:
            self.provider = env_provider
        
        # 加载密钥和URL
        if creds.vision.api_key:
            self.api_key = creds.vision.api_key
        if creds.vision.api_url:
            self.api_url = creds.vision.api_url
        if creds.vision.model:
            self.model = creds.vision.model
        
        # 如果没有单独配置Vision，复用DeepSeek LLM配置
        if self.provider == "deepseek":
            if not self.api_key:
                self.api_key = creds.llm.api_key
            if not self.api_url:
                self.api_url = creds.llm.api_url or "https://api.deepseek.com/v1"
            if not self.model:
                self.model = "deepseek-chat"
        
        # 提供商特定的默认配置
        elif self.provider == "qwen":
            if not self.api_url:
                self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if not self.model:
                self.model = "qwen-vl-plus"
        
        elif self.provider == "openai":
            if not self.api_url:
                self.api_url = "https://api.openai.com/v1"
            if not self.model:
                self.model = "gpt-4o"
