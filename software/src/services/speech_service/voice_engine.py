"""语音引擎 - 提供唤醒、ASR、TTS功能

基于 sherpa-onnx 实现语音唤醒和语音识别
支持火山引擎流式 TTS
"""
import asyncio
import io
import time
import wave
import numpy as np
import sounddevice as sd
import sherpa_onnx
from pathlib import Path
from common.logging import get_logger
from configs.config import get_config
from services.speech_service.tts_client import (
    tts_connect,
    tts_disconnect,
    tts_synthesize_stream,
)

logger = get_logger(__name__)


class VoiceEngine:
    """语音引擎类
    
    支持三种模式:
    - full: 完整模式（唤醒 + ASR + TTS）
    - tts_only: 仅 TTS 模式（用于语音交互应用，避免重复加载模型）
    """
    
    def __init__(self, mode: str = "full"):
        """初始化语音引擎
        
        Args:
            mode: 运行模式，"full" 或 "tts_only"
        """
        config = get_config()
        self.speech_config = config.speech
        self.tts_config = config.tts
        self.mode = mode
        
        self.is_initialized = False
        
        # 唤醒词检测相关（仅 full 模式）
        self.wakeup_keyword = self.speech_config.wakeup_keyword
        self.sample_rate = self.speech_config.sample_rate
        self.block_size = 1600  # 100ms
        self.confidence_threshold = self.speech_config.wakeup_sensitivity
        self.wakeup_detector = None
        self.wakeup_stream = None
        
        # ASR相关（仅 full 模式）
        self.asr_recognizer = None
        self.asr_stream = None
        self.listen_timeout = self.speech_config.listen_timeout
        
        # 状态管理（仅 full 模式）
        self.in_listening_mode = False
        self.last_speech_time = 0
        self.current_text = ""

        # 模拟唤醒冷却（避免模型缺失时频繁误触发）
        self._simulated_wakeup_cooldown = 5.0  # 秒
        self._last_simulated_wakeup_time = 0.0

        # 麦克风相关（仅 full 模式）
        # 优先使用设备名称（mic_name）匹配，找不到再回退到索引（mic_index）
        mic_name = getattr(self.speech_config, 'mic_name', None)
        if mic_name:
            self.mic_index = self._resolve_mic_index_by_name(mic_name)
        else:
            self.mic_index = self.speech_config.mic_index
        self.mic_stream = None
        self.samples_per_read = int(0.1 * self.sample_rate)  # 0.1秒

        # TTS相关
        self.tts_connected = False
        self.tts_sample_rate = config.tts.sample_rate  # 从配置读取采样率
        self.tts_channels = 1
        self.tts_bytes_per_sample = 2  # 16-bit PCM

        # 音频播放相关
        self.audio_stream = sd.OutputStream(
            channels=self.tts_channels,
            samplerate=self.tts_sample_rate,
            dtype="int16",
            blocksize=2048
        )

        if mode == "tts_only":
            self._initialize_tts_only()
        else:
            self._initialize()

    def _resolve_mic_index_by_name(self, name_pattern: str) -> int:
        """根据设备名称查找 sounddevice 麦克风索引

        按名称匹配可以避免设备插拔后索引变化的问题。

        Args:
            name_pattern: 设备名称或名称中的关键词（不区分大小写）

        Returns:
            sounddevice 设备索引，未找到时回退到 mic_index 配置
        """
        if not name_pattern:
            return self.speech_config.mic_index

        pattern = name_pattern.lower()
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev.get('max_input_channels', 0) > 0:
                    dev_name = dev.get('name', '').lower()
                    if pattern in dev_name:
                        logger.info(f"麦克风名称匹配: '{name_pattern}' -> 索引 {i} ({dev.get('name')})")
                        return i
            logger.warning(
                f"未找到麦克风设备名称包含 '{name_pattern}'，"
                f"回退到索引 {self.speech_config.mic_index}"
            )
        except Exception as e:
            logger.error(f"查询麦克风设备失败: {e}")
        return self.speech_config.mic_index

    def _initialize_tts_only(self):
        """仅初始化 TTS 相关资源（轻量级模式）"""
        logger.info("语音引擎以 TTS-only 模式初始化")
        # 仅初始化音频播放流，不加载唤醒和 ASR 模型
        try:
            self.audio_stream.start()
            self.audio_stream.stop()
            logger.info("TTS-only 模式初始化成功")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"TTS-only 模式初始化失败: {e}")
            raise
    
    def _initialize(self):
        """初始化语音引擎"""
        try:
            # 计算模型文件目录
            # voice_engine.py 路径: software/src/services/speech_service/voice_engine.py
            base_path = Path(__file__).parent.parent.parent.parent  # 项目根目录 (software/)
            
            wakeup_model_dir = (base_path / self.speech_config.wakeup_model_path).resolve()
            asr_model_dir = (base_path / self.speech_config.asr_model_path).resolve()
            
            logger.info(f"模型根目录: {base_path}")
            logger.info(f"唤醒模型目录: {wakeup_model_dir}")
            logger.info(f"ASR模型目录: {asr_model_dir}")

            if not wakeup_model_dir.exists():
                logger.warning(f"唤醒模型目录不存在: {wakeup_model_dir}，使用模拟语音引擎")
                self.is_initialized = True
                return
                
            if not asr_model_dir.exists():
                logger.warning(f"ASR模型目录不存在: {asr_model_dir}，使用模拟语音引擎")
                self.is_initialized = True
                return
            
            # 检查关键模型文件是否存在
            required_files = {
                "wakeup_encoder": wakeup_model_dir / self.speech_config.wakeup_encoder_file,
                "wakeup_decoder": wakeup_model_dir / self.speech_config.wakeup_decoder_file,
                "wakeup_joiner": wakeup_model_dir / self.speech_config.wakeup_joiner_file,
                "wakeup_tokens": wakeup_model_dir / "tokens.txt",
                "wakeup_keywords": wakeup_model_dir / self.speech_config.wakeup_keyword_file,
            }
            
            for name, path in required_files.items():
                if not path.exists():
                    logger.warning(f"模型文件缺失: {name} = {path}")
                    logger.warning("使用模拟语音引擎")
                    self.is_initialized = True
                    return
                else:
                    logger.debug(f"模型文件就绪: {name} = {path} ({path.stat().st_size} bytes)")
            
            # 初始化唤醒引擎
            logger.info("正在加载唤醒模型...")
            try:
                self.wakeup_detector = sherpa_onnx.KeywordSpotter(
                    tokens=str(wakeup_model_dir / "tokens.txt"),
                    encoder=str(wakeup_model_dir / self.speech_config.wakeup_encoder_file),
                    decoder=str(wakeup_model_dir / self.speech_config.wakeup_decoder_file),
                    joiner=str(wakeup_model_dir / self.speech_config.wakeup_joiner_file),
                    keywords_file=str(wakeup_model_dir / self.speech_config.wakeup_keyword_file),
                    keywords_score=1,
                    keywords_threshold=self.confidence_threshold,
                    num_threads=1,
                    provider="cpu",
                )
                logger.info("唤醒模型加载成功")
            except Exception as e:
                logger.error(f"唤醒模型加载失败: {e}")
                raise
            
            # 初始化语音识别引擎
            logger.info("正在加载语音识别模型...")
            try:
                self.asr_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(asr_model_dir / "tokens.txt"),
                    encoder=str(asr_model_dir / self.speech_config.asr_encoder_file),
                    decoder=str(asr_model_dir / self.speech_config.asr_decoder_file),
                    joiner=str(asr_model_dir / self.speech_config.asr_joiner_file),
                    num_threads=1,
                    provider="cpu",
                    sample_rate=self.sample_rate,
                    feature_dim=80,
                )
                logger.info("ASR模型加载成功")
            except Exception as e:
                logger.error(f"ASR模型加载失败: {e}")
                raise
            
            # 创建唤醒流
            self.wakeup_stream = self.wakeup_detector.create_stream()
            
            logger.info("语音引擎初始化成功")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"语音引擎初始化失败: {e}")
            logger.warning("使用模拟语音引擎")
            self.is_initialized = True
    
    def wakeup(self) -> bool:
        """语音唤醒检测
        
        Returns:
            bool: 是否检测到唤醒词
        """
        if not self.is_initialized:
            logger.error("语音引擎未初始化")
            return False
        
        try:
            # 如果没有加载真实模型，使用模拟唤醒（带冷却，避免频繁误触发）
            if not self.wakeup_detector:
                import random
                current_time = time.time()
                if current_time - self._last_simulated_wakeup_time < self._simulated_wakeup_cooldown:
                    return False
                is_wakeup = random.random() < 0.005  # 0.5% 概率，大幅降低误触发
                if is_wakeup:
                    self._last_simulated_wakeup_time = current_time
                    logger.info(f"[模拟] 检测到唤醒词: {self.wakeup_keyword}")
                return is_wakeup
            
            # 真实唤醒检测
            if self.mic_stream is None:
                # 初始化麦克风流
                sd.default.device = self.mic_index
                devices = sd.query_devices()
                logger.info(f"使用麦克风设备: {devices[self.mic_index]['name']}")
                self.mic_stream = sd.InputStream(
                    channels=1, 
                    dtype="float32", 
                    samplerate=self.sample_rate
                )
                self.mic_stream.start()
            
            # 读取音频数据
            samples, _ = self.mic_stream.read(self.samples_per_read)
            samples = samples.reshape(-1)
            
            # 喂给唤醒模型
            self.wakeup_stream.accept_waveform(self.sample_rate, samples)
            
            # 检测唤醒词
            while self.wakeup_detector.is_ready(self.wakeup_stream):
                self.wakeup_detector.decode_stream(self.wakeup_stream)
                result = self.wakeup_detector.get_result(self.wakeup_stream)
                if result:
                    logger.info(f"检测到唤醒词: {result}")
                    self.wakeup_detector.reset_stream(self.wakeup_stream)
                    # 切换到语音识别模式
                    self.in_listening_mode = True
                    self.last_speech_time = time.time()
                    self.current_text = ""
                    self.asr_stream = self.asr_recognizer.create_stream()
                    logger.info(f"已唤醒，开始聆听... (静默{self.listen_timeout}秒自动退出)")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"语音唤醒检测失败: {e}")
            # 降级为模拟唤醒（带冷却，避免频繁误触发）
            import random
            current_time = time.time()
            if current_time - self._last_simulated_wakeup_time < self._simulated_wakeup_cooldown:
                return False
            is_wakeup = random.random() < 0.005  # 0.5% 概率，大幅降低误触发
            if is_wakeup:
                self._last_simulated_wakeup_time = current_time
                logger.info(f"[模拟] 检测到唤醒词: {self.wakeup_keyword}")
            return is_wakeup
    
    def recognize(self) -> str:
        """ASR语音识别
        
        Returns:
            str: 识别结果文本
        """
        if not self.is_initialized:
            logger.error("语音引擎未初始化")
            return ""
        
        try:
            if not self.in_listening_mode:
                logger.warning("未处于聆听模式，无法进行ASR识别")
                return ""

            start_time = time.time()
            max_listen_time = 10.0  # 最大聆听总时长（秒）
            speech_start_timeout = 5.0  # 等待语音开始的超时（秒）

            while True:
                # 读取音频数据
                samples, _ = self.mic_stream.read(self.samples_per_read)
                samples = samples.reshape(-1)

                current_time = time.time()
                self.asr_stream.accept_waveform(self.sample_rate, samples)

                # 解码音频
                while self.asr_recognizer.is_ready(self.asr_stream):
                    self.asr_recognizer.decode_stream(self.asr_stream)

                # 获取当前识别结果
                result = self.asr_recognizer.get_result(self.asr_stream)
                if result != self.current_text:
                    self.current_text = result
                    self.last_speech_time = current_time
                    logger.info(f"正在识别: {self.current_text}")

                # 检查超时
                if result and current_time - self.last_speech_time > self.listen_timeout:
                    break

                # 总超时保护：防止 ASR 始终无结果导致死循环
                elapsed = current_time - start_time
                if elapsed > max_listen_time:
                    logger.warning(f"聆听总超时 ({max_listen_time}s)，退出识别")
                    break

                # 语音开始超时：长时间没识别到任何文字，提前退出
                if not self.current_text and elapsed > speech_start_timeout:
                    logger.warning(f"未检测到语音输入 ({speech_start_timeout}s)，退出识别")
                    break

                time.sleep(0.01)
            
            final_result = self.current_text
            logger.info(f"ASR识别结果: {final_result}")
            self.in_listening_mode = False
            logger.info("结束聆听")
            return final_result
        except Exception as e:
            logger.error(f"ASR识别失败: {e}")
            return ""
    
    async def synthesize_streaming(self, text: str) -> None:
        """流式 TTS 合成并播放音频
        
        Args:
            text: 需要合成的文本
        """
        if not text:
            logger.warning("TTS合成文本为空")
            return
        
        try:
            text = text.replace("**", "")
            logger.info(f"开始 TTS 流式合成: {text}")
            self.audio_stream.start()
            
            # 流式接收并播放音频
            audio_buffer = bytearray()
            async for chunk in tts_synthesize_stream(text):
                if chunk:
                    audio_buffer.extend(chunk)
                    
                    # 将音频数据转换为 numpy 数组并播放
                    try:
                        audio_data = np.frombuffer(bytes(chunk), dtype=np.int16)
                        if len(audio_data) > 0:
                            self.audio_stream.write(audio_data)
                    except Exception as e:
                        logger.warning(f"播放音频块失败: {e}")
            
            # 确保所有音频播放完成
            self.audio_stream.stop()
            await tts_disconnect()

            logger.info(f"TTS 流式合成完成, 总共 {len(audio_buffer)} 字节")
        except Exception as e:
            logger.error(f"TTS 流式合成失败: {e}")
            await tts_disconnect()
    
    async def synthesize(self, text: str) -> None:
        """TTS文本转语音（异步接口）

        支持多提供商（火山引擎 / MiniMax 等），配置缺失或失败时自动降级为模拟 TTS。

        Args:
            text: 需要合成的文本
        """
        if not text:
            logger.warning("TTS合成文本为空")
            return

        try:
            await self.synthesize_streaming(text)
        except Exception as e:
            logger.error(f"TTS合成失败: {e}")
            logger.info(f"[模拟TTS] {text}")
    
    def play_audio_data(self, audio_data: bytes) -> None:
        """播放音频数据
        
        Args:
            audio_data: 音频数据字节流
        """
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            self.audio_stream.start()
            if len(audio_array) > 0:
                self.audio_stream.write(audio_array)
            self.audio_stream.stop()
        except Exception as e:
            logger.error(f"播放音频数据失败: {e}")
    
    def play_pcm_file(self, name: str) -> None:
        """播放PCM音频文件
        
        Args:
            name: PCM音频文件名称
        """
        try:
            base_path = Path(__file__).parent.parent.parent.parent
            pcm_file = base_path / self.speech_config.cache_dir / f"{name}.pcm"
            with open(pcm_file, "rb") as f:
                audio_data = f.read()
                self.play_audio_data(audio_data)
        except Exception as e:
            logger.error(f"播放PCM文件失败: {e}")
    
    def release(self):
        """释放语音引擎资源"""
        try:
            # 关闭音频播放流
            if self.audio_stream:
                try:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                except:
                    pass
                self.audio_stream = None
            
            # 关闭麦克风流
            if self.mic_stream:
                self.mic_stream.stop()
                self.mic_stream.close()
                self.mic_stream = None
            
            # 释放唤醒流
            if self.wakeup_stream:
                self.wakeup_stream = None
            
            # 释放ASR流
            if self.asr_stream:
                self.asr_stream = None
            
            # 断开 TTS 连接
            if self.tts_connected:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(tts_disconnect())
                    else:
                        loop.run_until_complete(tts_disconnect())
                except:
                    pass
                self.tts_connected = False
            
            self.is_initialized = False
            logger.info("语音引擎资源已释放")
        except Exception as e:
            logger.error(f"释放语音引擎资源失败: {e}")
