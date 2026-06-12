"""
Vision Service 启动入口

Usage:
    python -m services.vision_service
    python -m services.vision_service --display
    python -m services.vision_service --addr "tcp://*:5561" --device-name "USB摄像头"
"""
import argparse
from services.vision_service import VisionService
from configs.config import get_config


def main():
    parser = argparse.ArgumentParser(description='HomeBot Vision Service')
    parser.add_argument('--display', action='store_true', help='Show video window')
    parser.add_argument('--addr', default=None, help='Publish address (default: tcp://*:5560)')
    parser.add_argument('--device', type=int, default=None, help='Camera device ID (default: from config)')
    parser.add_argument('--device-name', default=None, help='Camera device name substring (default: from config)')
    parser.add_argument('--unique-id', default=None, help='macOS AVFoundation uniqueID (default: from config)')
    parser.add_argument('--flip-horizontal', action='store_true', help='Flip image horizontally (mirror)')
    parser.add_argument('--list-cameras', action='store_true', help='List available cameras and exit')
    args = parser.parse_args()

    # 列出摄像头并退出
    if args.list_cameras:
        from hal.camera.enumeration import print_camera_list
        print_camera_list()
        return

    # 如果指定了 device / device_name / unique_id，临时修改 config
    config = None
    if args.device is not None or args.device_name is not None or args.unique_id is not None:
        import copy
        base_config = get_config()
        config = copy.deepcopy(base_config)
        if args.device is not None:
            config.camera.device_id = args.device
            print(f"[VisionService] Using camera device_id={args.device}")
        if args.device_name is not None:
            config.camera.device_name = args.device_name
            print(f"[VisionService] Using camera device_name='{args.device_name}'")
        if args.unique_id is not None:
            config.camera.unique_id = args.unique_id
            print(f"[VisionService] Using camera unique_id='{args.unique_id}'")

    flip = args.flip_horizontal
    if args.addr:
        service = VisionService(pub_addr=args.addr, config=config, flip_horizontal=flip)
    else:
        service = VisionService(config=config, flip_horizontal=flip)
    service.start(display=args.display)


if __name__ == '__main__':
    main()
