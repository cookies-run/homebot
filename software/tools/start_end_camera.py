import sys
import platform
sys.path.insert(0, '/Users/yi/works/智绘屿/yu-freetime/robots/HomeBot/software/src')
from services.vision_service import VisionService
from dataclasses import dataclass, field

@dataclass
class CamCfg:
    device_id: int = 0
    device_name: str = ""
    unique_id: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 30

    def __post_init__(self):
        # On macOS use AVFoundation name/uniqueID matching for stable camera selection.
        if platform.system() == 'Darwin' and not self.device_name and not self.unique_id:
            self.device_name = "USB摄像头"

@dataclass
class ZmqCfg:
    vision_pub_addr: str = 'tcp://*:5561'

@dataclass
class Cfg:
    camera: object = None
    zmq: object = None

cfg = Cfg(camera=CamCfg(), zmq=ZmqCfg())
svc = VisionService(config=cfg)
svc.start()
