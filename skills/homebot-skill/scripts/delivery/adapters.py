"""服务适配层（技能自包含版本）。

封装与 HomeBot ZeroMQ 服务（底盘、机械臂、视觉）的通信。
此文件从 applications/delivery_agent/adapter.py 下沉到技能目录，使 homebot-skill
不再依赖已废弃的 applications.delivery_agent 包（任务拆解已上移到 openclaw）。
所有实现均为通信封装，不修改原有服务代码。
"""
import os
import sys
import time
from typing import Optional

# 支持独立运行：将 software/src 加入路径以导入 common/configs/services
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../software/src"))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import zmq

from common.logging import get_logger
from common.zmq_helper import create_socket
from configs.config import get_config
from services.vision_service.vision import VisionSubscriber

logger = get_logger(__name__)


class ChassisAdapter:
    """底盘控制适配器。"""

    DEFAULT_TIMEOUT_MS = 3000

    def __init__(self, addr: Optional[str] = None, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        config = get_config()
        self.addr = addr or config.zmq.chassis_service_addr.replace("*", "localhost")
        self.timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket: Optional[zmq.Socket] = None

    def _get_socket(self) -> zmq.Socket:
        if self._socket is None:
            self._socket = create_socket(zmq.REQ, bind=False, address=self.addr, context=self._context)
            self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.LINGER, 0)
        return self._socket

    def _reset_socket(self):
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._socket = None

    def send_velocity(self, vx: float, vy: float, vz: float, duration_ms: int = 1000) -> dict:
        """发送速度指令持续一段时间。"""
        try:
            socket = self._get_socket()
            command = {
                "source": "delivery_agent",
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "priority": 3,
            }
            start = time.time()
            interval = 0.2
            last_response = None
            while (time.time() - start) * 1000 < duration_ms:
                try:
                    socket.send_json(command, flags=zmq.NOBLOCK)
                except zmq.Again:
                    self._reset_socket()
                    return {"status": "error", "message": "发送底盘命令超时"}
                try:
                    last_response = socket.recv_json()
                except zmq.Again:
                    self._reset_socket()
                    return {"status": "error", "message": "接收底盘响应超时"}
                elapsed = (time.time() - start) * 1000
                remaining = duration_ms - elapsed
                if remaining > interval * 1000:
                    time.sleep(interval)
                elif remaining > 0:
                    time.sleep(remaining / 1000.0)

            # 停止
            stop_cmd = {"source": "delivery_agent", "vx": 0, "vy": 0, "vz": 0, "priority": 3}
            try:
                socket.send_json(stop_cmd, flags=zmq.NOBLOCK)
                socket.recv_json()
            except zmq.Again:
                logger.warning("底盘停止命令超时")
            return {"status": "success", "data": last_response}
        except Exception as e:
            self._reset_socket()
            logger.error(f"底盘控制异常: {e}")
            return {"status": "error", "message": str(e)}

    def move_forward_cm(self, cm: float, speed: float = 0.2) -> dict:
        """前进指定厘米。"""
        m = cm / 100.0
        duration_ms = int((m / speed) * 1000) if speed > 0 else 1000
        return self.send_velocity(speed, 0, 0, duration_ms)

    def move_backward_cm(self, cm: float, speed: float = 0.2) -> dict:
        """后退指定厘米。"""
        m = cm / 100.0
        duration_ms = int((m / speed) * 1000) if speed > 0 else 1000
        return self.send_velocity(-speed, 0, 0, duration_ms)

    def rotate_left_deg(self, deg: float, speed: float = 0.5) -> dict:
        """左转指定角度。"""
        rad = deg * 3.14159 / 180.0
        duration_ms = int((abs(rad) / speed) * 1000)
        return self.send_velocity(0, 0, -speed, duration_ms)

    def rotate_right_deg(self, deg: float, speed: float = 0.5) -> dict:
        """右转指定角度。"""
        rad = deg * 3.14159 / 180.0
        duration_ms = int((abs(rad) / speed) * 1000)
        return self.send_velocity(0, 0, speed, duration_ms)

    def stop(self) -> dict:
        """停止底盘。"""
        return self.send_velocity(0, 0, 0, 200)

    def close(self):
        if self._socket:
            self._socket.close()
        self._context.term()


class ArmAdapter:
    """机械臂控制适配器。"""

    DEFAULT_TIMEOUT_MS = 5000

    def __init__(self, addr: Optional[str] = None, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        config = get_config()
        self.addr = addr or config.zmq.arm_service_addr.replace("*", "localhost")
        self.timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket: Optional[zmq.Socket] = None

    def _get_socket(self) -> zmq.Socket:
        if self._socket is None:
            self._socket = create_socket(zmq.REQ, bind=False, address=self.addr, context=self._context)
            self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.LINGER, 0)
        return self._socket

    def _reset_socket(self):
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._socket = None

    def set_joint_angles(self, joint_angles: dict, speed: int = 800) -> dict:
        """设置关节角度。"""
        try:
            socket = self._get_socket()
            command = {
                "source": "delivery_agent",
                "priority": 3,
                "speed": speed,
                "joints": joint_angles,
            }
            socket.send_json(command, flags=zmq.NOBLOCK)
            response = socket.recv_json()
            success = response.get("success", False)
            return {
                "status": "success" if success else "failed",
                "data": response,
                "message": response.get("message", ""),
            }
        except zmq.Again:
            self._reset_socket()
            return {"status": "error", "message": "机械臂通信超时"}
        except Exception as e:
            self._reset_socket()
            logger.error(f"机械臂控制异常: {e}")
            return {"status": "error", "message": str(e)}

    def set_gripper(self, angle: float) -> dict:
        """设置夹爪角度。"""
        return self.set_joint_angles({"gripper": angle})

    def open_gripper(self) -> dict:
        return self.set_gripper(90)

    def close_gripper(self) -> dict:
        return self.set_gripper(0)

    def get_joint_states(self) -> dict:
        """获取关节状态。"""
        try:
            socket = self._get_socket()
            command = {"source": "delivery_agent", "priority": 3, "speed": 0, "joints": {}, "query": True}
            socket.send_json(command, flags=zmq.NOBLOCK)
            response = socket.recv_json()
            return response.get("joint_states", {})
        except Exception as e:
            self._reset_socket()
            logger.error(f"获取机械臂状态异常: {e}")
            return {}

    def close(self):
        if self._socket:
            self._socket.close()
        self._context.term()


class VisionAdapter:
    """视觉订阅适配器。"""

    def __init__(self, addr: Optional[str] = None):
        config = get_config()
        self.addr = addr or config.zmq.vision_pub_addr.replace("*", "localhost")
        self._subscriber: Optional[VisionSubscriber] = None

    def start(self):
        """启动订阅。"""
        if self._subscriber is None:
            self._subscriber = VisionSubscriber(self.addr)
            self._subscriber.start()
            # 等待第一帧
            time.sleep(0.5)

    def read_frame(self) -> tuple:
        """读取最新帧。

        Returns:
            (frame_id, frame) 或 (None, None)
        """
        if self._subscriber is None:
            self.start()
        return self._subscriber.read_frame()

    def stop(self):
        if self._subscriber:
            self._subscriber.stop()
            self._subscriber = None
