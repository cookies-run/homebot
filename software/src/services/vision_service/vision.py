"""VisionService - 图像采集、处理和发布服务.

其他应用通过订阅此服务发布的图像帧来获取视频流.
"""

import zmq
import time
from common.logging import get_logger

logger = get_logger(__name__)


class VisionService:
    """视觉服务: 直接采集图像,处理后发布给其他应用订阅."""
    
    def __init__(self, pub_addr: str = "tcp://*:5560", config=None):
        """
        初始化视觉服务.
        
        Args:
            pub_addr: ZMQ PUB socket 绑定地址
            config: 配置对象,包含 camera 和 zmq 配置
        """
        from common.zmq_helper import create_socket
        from configs.config import get_config
        
        # 加载配置
        if config is None:
            config = get_config()
        self._config = config

        # 从配置获取地址（仅在未显式传入 pub_addr 时）
        # 优先级: 传入的 pub_addr 参数 > config 中的配置
        if pub_addr == "tcp://*:5560":  # 只有使用默认值时才从 config 读取
            if hasattr(config, 'zmq') and hasattr(config.zmq, 'vision_pub_addr'):
                pub_addr = config.zmq.vision_pub_addr
            elif hasattr(config, 'network') and hasattr(config.network, 'zmq') and hasattr(config.network.zmq, 'vision_pub_addr'):
                pub_addr = config.network.zmq.vision_pub_addr
            
        # 创建 PUB socket 用于发布图像帧
        self._pub_socket = create_socket(zmq.PUB, bind=True, address=pub_addr)
        logger.info(f"VisionService PUB socket bound to {pub_addr}")
        
        # 相机配置
        if hasattr(config, 'camera') and config.camera:
            import platform
            device_name = getattr(config.camera, 'device_name', '')
            unique_id = getattr(config.camera, 'unique_id', '')
            if platform.system() == 'Darwin' and (device_name or unique_id):
                # macOS 使用 AVFoundation 原生驱动，按名称/uniqueID 匹配，
                # 不经过 OpenCV 的整数索引，因此跳过 resolve_device_id。
                self._cam_device = -1
            else:
                from hal.camera.enumeration import resolve_device_id
                self._cam_device = resolve_device_id(config)
            self._fps = getattr(config.camera, 'fps', 30)
            self._width = getattr(config.camera, 'width', 640)
            self._height = getattr(config.camera, 'height', 480)
        else:
            self._cam_device = 0
            self._fps = 30
            self._width = 640
            self._height = 480
        
        # 相机驱动 (延迟初始化)
        self._cam = None
        
        # 运行标志
        self._running = False

    def _init_camera(self):
        """初始化相机驱动.

        macOS 上如果配置了 device_name，使用原生的 AVFoundation 驱动以绕过
        OpenCV 不稳定的整数索引；否则回退到 OpenCV CameraDriver。
        """
        import platform
        camera_cfg = getattr(self._config, 'camera', None)
        device_name = getattr(camera_cfg, 'device_name', '') if camera_cfg else ''
        unique_id = getattr(camera_cfg, 'unique_id', '') if camera_cfg else ''

        if platform.system() == 'Darwin' and (device_name or unique_id):
            from hal.camera.avfoundation_driver import AVFoundationCameraDriver
            # unique_id 优先；如果配置了 unique_id，按硬件 ID 直接匹配。
            lookup_name = unique_id if unique_id else device_name
            self._cam = AVFoundationCameraDriver(
                device_name=lookup_name,
                width=self._width,
                height=self._height,
                fps=self._fps,
                flip_horizontal=True,
            )
            resolved_uid = getattr(self._cam, 'unique_id', '')
            logger.info(
                f"AVFoundation camera initialized: name='{device_name}', unique_id={resolved_uid}, "
                f"fps={self._fps}, resolution={self._width}x{self._height}, flip_horizontal=True"
            )
        else:
            from hal.camera.driver import CameraDriver
            self._cam = CameraDriver(
                self._cam_device,
                flip_horizontal=True,
                width=self._width,
                height=self._height,
            )
            logger.info(
                f"OpenCV camera initialized: device={self._cam_device}, fps={self._fps}, "
                f"resolution={self._width}x{self._height}, flip_horizontal=True"
            )

    def process_frame(self, frame):
        """处理图像帧 (子类可重写此方法添加视觉处理逻辑).
        
        Args:
            frame: OpenCV 图像帧 (numpy array)
            
        Returns:
            处理后的图像帧,如果不需要修改则返回原帧
        """
        # placeholder for detection logic (e.g., object detection, tracking)
        return frame

    def start(self, display: bool = False):
        """启动图像采集和发布循环.
        
        Args:
            display: 是否显示图像窗口 (调试用)
        """
        import cv2
        import numpy as np
        
        self._init_camera()
        self._running = True
        
        frame_id = 0
        tgt_int = 1.0 / self._fps if self._fps > 0 else 0
        
        logger.info(f"VisionService started publishing at {self._fps} FPS")

        consecutive_failures = 0
        MAX_FAILURES_BEFORE_REOPEN = 5

        try:
            perf = time.perf_counter
            while self._running:
                t0 = perf()

                # 1. 采集图像
                frame = self._cam.capture_frame()
                if frame is None:
                    consecutive_failures += 1
                    logger.warning(f"no frame captured (consecutive failures: {consecutive_failures})")
                    if consecutive_failures >= MAX_FAILURES_BEFORE_REOPEN:
                        logger.warning("too many consecutive failures, attempting camera reopen...")
                        if self._cam.reopen():
                            consecutive_failures = 0
                            logger.info("camera reopened successfully")
                        else:
                            logger.error("camera reopen failed, will retry")
                    time.sleep(0.1)
                    continue
                else:
                    consecutive_failures = 0
                
                # 2. 处理图像 (子类可扩展)
                processed_frame = self.process_frame(frame)
                
                # 3. 编码并发布
                ret, buf = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    logger.warning("failed to encode frame")
                    continue
                    
                self._pub_socket.send_multipart([str(frame_id).encode(), buf.tobytes()])
                frame_id += 1
                
                # 4. 可选显示
                if display and processed_frame is not None:
                    cv2.imshow('VisionService', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # FPS 控制
                elapsed = perf() - t0
                rem = tgt_int - elapsed
                if rem > 0:
                    time.sleep(rem)
                    
        except KeyboardInterrupt:
            logger.info("VisionService interrupted by user")
        except Exception as e:
            logger.error(f"VisionService error: {e}")
        finally:
            self.stop()
            if display:
                cv2.destroyAllWindows()

    def stop(self):
        """停止服务并释放资源."""
        self._running = False
        if self._cam:
            self._cam.release()
            self._cam = None
        logger.info("VisionService stopped")

    # 保持向后兼容: listen 方法重命名为 start
    listen = start


class VisionSubscriber:
    """视觉订阅者 - 用于其他应用订阅 VisionService 发布的图像帧.
    
    使用后台线程持续接收图像，始终保持最新帧，避免延迟累积。
    """
    
    def __init__(self, sub_addr: str = "tcp://localhost:5560"):
        """
        初始化视觉订阅者.
        
        Args:
            sub_addr: VisionService 的 PUB 地址
        """
        from common.zmq_helper import create_socket
        from threading import Lock, Thread
        
        self._sub_socket = create_socket(zmq.SUB, bind=False, address=sub_addr)
        self._sub_socket.setsockopt(zmq.RCVHWM, 1)
        self._sub_socket.setsockopt(zmq.CONFLATE, 1)
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub_socket.setsockopt(zmq.RCVTIMEO, 1000)
        
        # 后台线程相关
        self._latest_frame = None
        self._frame_lock = Lock()
        self._running = False
        self._thread = None
        self._frame_id = 0
        
        logger.info(f"VisionSubscriber connected to {sub_addr}")

    def start(self):
        """启动后台接收线程."""
        if self._running:
            return
        self._running = True
        from threading import Thread
        self._thread = Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info("VisionSubscriber started background receiver")

    def stop(self):
        """停止后台接收线程."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sub_socket:
            self._sub_socket.close()
        logger.info("VisionSubscriber stopped")

    def _receive_loop(self):
        """后台线程持续接收图像帧."""
        import cv2
        import numpy as np
        
        while self._running:
            try:
                parts = self._sub_socket.recv_multipart()
                if len(parts) == 2:
                    frame_id_str = parts[0].decode()
                    buf = parts[1]
                    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with self._frame_lock:
                            self._latest_frame = img
                            try:
                                self._frame_id = int(frame_id_str)
                            except ValueError:
                                self._frame_id += 1
            except zmq.Again:
                # 超时，继续循环
                continue
            except Exception as e:
                logger.warning(f"VisionSubscriber receive error: {e}")

    def read_frame(self, timeout: int = 1000):
        """读取最新帧图像（非阻塞，从内存直接获取）.
        
        Args:
            timeout: 保留参数，兼容性用途（实际不阻塞等待）
            
        Returns:
            (frame_id, frame) 元组,如果没有帧返回 (None, None)
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None, None
            return self._frame_id, self._latest_frame.copy()

    def read_loop(self, callback=None, display: bool = False):
        """持续读取图像帧.
        
        注意：需要先用 start() 启动后台线程，否则 read_frame 会返回空。
        
        Args:
            callback: 每帧的回调函数,接收 (frame_id, frame) 参数
            display: 是否显示图像
        """
        import cv2
        import time
        
        if not self._running:
            self.start()
        
        logger.info("VisionSubscriber read_loop started")
        
        try:
            while True:
                frame_id, frame = self.read_frame()
                if frame is not None:
                    if callback:
                        callback(frame_id, frame)
                    if display:
                        cv2.imshow('VisionSubscriber', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    # 没有帧时短暂休眠避免CPU占用过高
                    time.sleep(0.001)
        except KeyboardInterrupt:
            logger.info("VisionSubscriber interrupted")
        finally:
            if display:
                cv2.destroyAllWindows()


# from services.vision_service import VisionSubscriber
# vs=VisionSubscriber()
# vs.read_loop(display=True)
