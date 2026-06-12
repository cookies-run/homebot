"""开放式 TTS 客户端

支持多提供商流式语音合成：
- volcano: 火山引擎 WebSocket TTS
- minimax: MiniMax Speech 2.8 (OpenAI 兼容接口)

未来接入新 provider 只需：
1. 实现 TTSClient Protocol
2. 在 get_tts_client() 工厂中注册
"""
import asyncio
import copy
import json
import typing
import uuid
from typing import AsyncGenerator, Optional, Protocol

import httpx
import websockets
from openai import AsyncOpenAI

from common.logging import get_logger
from configs.config import get_config
from services.speech_service.protocols import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)

logger = get_logger(__name__)


class TTSClient(Protocol):
    """TTS 客户端协议接口

    所有 TTS 提供商必须实现此接口，供 VoiceEngine 统一调用。
    """
    is_connected: bool

    async def connect(self) -> bool:
        """建立连接，返回是否成功"""
        ...

    async def disconnect(self) -> None:
        """断开连接，释放资源"""
        ...

    async def synthesize(self, text: str) -> bytes:
        """完整合成（一次性返回全部音频数据）"""
        ...

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """流式合成（逐块返回音频数据）"""
        ...


class VolcanoTTSClient:
    """火山引擎流式 TTS 客户端"""
    
    def __init__(self):
        """初始化火山引擎 TTS 客户端"""
        tts_config = get_config().tts
        self.appid = tts_config.appid
        self.access_token = tts_config.access_token
        self.resource_id = tts_config.resource_id
        self.voice_type = tts_config.voice_type
        self.encoding = tts_config.encoding
        self.endpoint = tts_config.endpoint

        self.websocket = None
        self.session_id = None
        self.is_connected = False

    def _get_resource_id(self) -> str:
        """获取资源 ID"""
        if self.resource_id:
            return self.resource_id
        if self.voice_type.startswith("S_"):
            return "volc.megatts.default"
        return "volc.service_type.10029"

    async def connect(self) -> bool:
        """连接到火山引擎 TTS 服务器并启动会话

        Returns:
            bool: 连接是否成功
        """
        try:
            headers = {
                "X-Api-App-Key": self.appid,
                "X-Api-Access-Key": self.access_token,
                "X-Api-Resource-Id": self._get_resource_id(),
                "X-Api-Connect-Id": str(uuid.uuid4()),
            }

            logger.info(f"正在连接到火山引擎 TTS 服务器...")
            self.websocket = await websockets.connect(
                self.endpoint,
                additional_headers=headers,
                max_size=10 * 1024 * 1024
            )

            logid = self.websocket.response.headers.get('x-tt-logid', 'unknown')
            logger.info(f"连接成功, Logid: {logid}")

            # 启动连接
            await start_connection(self.websocket)
            await wait_for_event(
                self.websocket, MsgType.FullServerResponse, EventType.ConnectionStarted
            )

            # 启动会话
            base_request = {
                "user": {"uid": str(uuid.uuid4())},
                "namespace": "BidirectionalTTS",
                "req_params": {
                    "speaker": self.voice_type,
                    "audio_params": {
                        "format": self.encoding,
                        "sample_rate": 16000,
                        "enable_timestamp": True,
                    },
                    "additions": json.dumps({
                        "disable_markdown_filter": False,
                    }),
                },
            }
            start_session_request = copy.deepcopy(base_request)
            start_session_request["event"] = EventType.StartSession
            self.session_id = str(uuid.uuid4())

            await start_session(
                self.websocket,
                json.dumps(start_session_request).encode(),
                self.session_id
            )
            await wait_for_event(
                self.websocket, MsgType.FullServerResponse, EventType.SessionStarted
            )

            logger.info(f"TTS 会话启动成功, Session ID: {self.session_id}")
            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"连接火山引擎 TTS 服务器失败: {e}")
            self.is_connected = False
            return False
    
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """流式合成语音
        
        Args:
            text: 要合成的文本
            
        Yields:
            bytes: 音频数据块
        """
        if not self.websocket or not self.session_id:
            logger.error("未连接或未启动会话")
            return
        
        try:
            # 发送文本数据
            async def send_text():
                for char in text:
                    synthesis_request = {
                        "user": {"uid": str(uuid.uuid4())},
                        "namespace": "BidirectionalTTS",
                        "event": EventType.TaskRequest,
                        "req_params": {
                            "speaker": self.voice_type,
                            "text": char,
                        },
                    }
                    await task_request(
                        self.websocket, 
                        json.dumps(synthesis_request).encode(), 
                        self.session_id
                    )
                    await asyncio.sleep(0.005)  # 5ms 延迟
                
                await finish_session(self.websocket, self.session_id)
            
            # 启动发送任务
            send_task = asyncio.create_task(send_text())
            
            # 接收音频数据
            audio_data = bytearray()
            while True:
                msg = await receive_message(self.websocket)
                
                if msg is None:
                    break
                    
                if msg.type == MsgType.FullServerResponse:
                    if msg.event == EventType.SessionFinished:
                        break
                elif msg.type == MsgType.AudioOnlyServer:
                    if hasattr(msg, 'payload') and msg.payload:
                        chunk = msg.payload
                        yield chunk
                        audio_data.extend(chunk)
                else:
                    logger.warning(f"未知消息类型: {msg.type}")
                    break
            
            # 等待发送完成
            await send_task
            
            logger.info(f"流式合成完成, 总共返回 {len(audio_data)} 字节音频数据")
            
        except Exception as e:
            logger.error(f"流式合成失败: {e}")
    
    async def synthesize(self, text: str) -> bytes:
        """完整合成语音（一次性返回所有音频数据）

        Args:
            text: 要合成的文本

        Returns:
            bytes: 完整的音频数据
        """
        if not self.is_connected:
            if not await self.connect():
                return b""

        audio_chunks = []
        async for chunk in self.synthesize_stream(text):
            audio_chunks.append(chunk)

        await self.disconnect()

        return b"".join(audio_chunks)
    
    async def disconnect(self):
        """断开连接，释放资源"""
        try:
            if self.websocket:
                await finish_connection(self.websocket)
                await self.websocket.close()
                self.websocket = None
                self.session_id = None
                self.is_connected = False
                logger.info("已断开与火山引擎 TTS 服务器的连接")
        except Exception as e:
            logger.error(f"断开连接时出错: {e}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.disconnect()


class MiniMaxTTSClient:
    """MiniMax Speech 2.8 TTS 客户端（原生 HTTP /v1/t2a_v2 接口）"""

    def __init__(self):
        """初始化 MiniMax TTS 客户端"""
        tts_config = get_config().tts
        self.api_key = tts_config.api_key
        self.api_url = tts_config.api_url or "https://api.minimax.chat/v1"
        self.model = tts_config.model or "speech-2.8-hd"
        self.voice_id = tts_config.voice_type or "Chinese (Mandarin)_Gentle_Senior"
        self.audio_format = tts_config.encoding or "pcm"
        self.sample_rate = tts_config.sample_rate or 16000
        self.is_connected = True  # HTTP 无需显式连接

    async def connect(self) -> bool:
        """HTTP 无需显式连接"""
        return True

    async def disconnect(self) -> None:
        """HTTP 无需显式断开"""
        pass

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """流式合成语音（MiniMax 原生 t2a_v2 接口）

        当前使用非流式请求，一次性返回完整音频后分块 yield，
        保证与 VoiceEngine 的流式播放接口兼容。

        Args:
            text: 要合成的文本

        Yields:
            bytes: 音频数据块
        """
        try:
            logger.info(f"MiniMax TTS 合成: {text}")
            url = f"{self.api_url.rstrip('/')}/t2a_v2"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "text": text,
                "stream": False,
                "voice_setting": {
                    "voice_id": self.voice_id,
                    "speed": 1,
                    "vol": 1,
                    "pitch": 0,
                },
                "audio_setting": {
                    "sample_rate": self.sample_rate,
                    "format": self.audio_format,
                    "channel": 1,
                },
            }
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()

            hex_audio = result.get("data", {}).get("audio", "")
            if not hex_audio:
                logger.warning("MiniMax TTS 返回空音频")
                return

            audio_bytes = bytes.fromhex(hex_audio)
            # 分块 yield，模拟流式效果，与 VoiceEngine 兼容
            chunk_size = 4096
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]

            extra = result.get("extra_info", {})
            logger.info(
                f"MiniMax TTS 完成: {len(audio_bytes)} 字节, "
                f"采样率 {extra.get('audio_sample_rate', 'unknown')}, "
                f"字数 {extra.get('usage_characters', 'unknown')}"
            )
        except Exception as e:
            logger.error(f"MiniMax TTS 合成失败: {e}")
            raise

    async def synthesize(self, text: str) -> bytes:
        """完整合成语音（一次性返回所有音频数据）

        Args:
            text: 要合成的文本

        Returns:
            bytes: 完整的音频数据
        """
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# 全局 TTS 客户端实例
_tts_client: Optional[TTSClient] = None


async def get_tts_client() -> TTSClient:
    """获取全局 TTS 客户端实例（工厂模式）

    根据 config.tts.provider 自动创建对应提供商的客户端。

    Returns:
        TTSClient: TTS 客户端实例
    """
    global _tts_client
    if _tts_client is None:
        config = get_config().tts
        logger.info(f"TTS Provider: {config.provider}")
        if config.provider == "volcano":
            _tts_client = VolcanoTTSClient()
        elif config.provider == "minimax":
            _tts_client = MiniMaxTTSClient()
        else:
            raise ValueError(f"未知的 TTS 提供商: {config.provider}")
    return _tts_client


async def tts_synthesize_stream(text: str) -> AsyncGenerator[bytes, None]:
    """流式 TTS 合成接口

    Args:
        text: 要合成的文本

    Yields:
        bytes: 音频数据块
    """
    client = await get_tts_client()
    if not client.is_connected:
        if not await client.connect():
            return

    async for chunk in client.synthesize_stream(text):
        yield chunk


async def tts_synthesize(text: str) -> bytes:
    """完整 TTS 合成接口

    Args:
        text: 要合成的文本

    Returns:
        bytes: 完整的音频数据
    """
    client = await get_tts_client()
    if '**' in text:
        text = text.replace('**', '')
    return await client.synthesize(text)


def tts_to_file(text: str, audio_file: str) -> None:
    """将文本转换为音频文件

    Args:
        text: 要合成的文本
        audio_file: 输出音频文件路径
    """
    audio_data = asyncio.run(tts_synthesize(text))
    with open(audio_file, "wb") as f:
        f.write(audio_data)


async def tts_connect() -> bool:
    """建立 TTS 连接

    Returns:
        bool: 连接是否成功
    """
    client = await get_tts_client()
    if client.is_connected:
        return True
    return await client.connect()


async def tts_disconnect():
    """断开 TTS 连接"""
    global _tts_client
    if _tts_client:
        await _tts_client.disconnect()
        _tts_client = None


if __name__ == "__main__":
    # 测试代码
    async def test_tts():
        logger.info("测试 TTS 合成功能...")
        
        text = "你好，我是HomeBot语音助手"
        logger.info(f"合成文本: {text}")
        
        client = await get_tts_client()
        if await client.connect() and await client.start_session():
            audio_chunks = []
            async for chunk in client.synthesize_stream(text):
                audio_chunks.append(chunk)
                logger.info(f"收到音频块: {len(chunk)} 字节")
            
            await client.disconnect()
            
            if audio_chunks:
                total_audio = b"".join(audio_chunks)
                logger.info(f"合成完成，总音频大小: {len(total_audio)} 字节")
            else:
                logger.warning("未收到任何音频数据")
        else:
            logger.error("连接失败")
    
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_tts())
