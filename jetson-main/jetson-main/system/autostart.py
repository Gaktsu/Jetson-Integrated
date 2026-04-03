"""
부팅 시 자동 실행 설정
"""
import os
import sys


def setup_autostart(script_path: str = None) -> bool:
    """
    시스템 부팅 시 자동 실행 설정
    
    Args:
        script_path: 실행할 스크립트 경로 (None이면 현재 main.py)
    
    Returns:
        성공 여부
    """
    if script_path is None:
        script_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    
    # Windows: 작업 스케줄러
    if sys.platform == "win32":
        print("Windows 자동 시작 설정은 작업 스케줄러를 사용하세요.")
        return False
    
    # Linux: systemd 서비스
    elif sys.platform == "linux":
        service_content = f"""[Unit]
Description=Person Detection System
After=network.target

[Service]
Type=simple
User={os.getenv('USER')}
ExecStart=/usr/bin/python3 {os.path.abspath(script_path)}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        service_path = "/etc/systemd/system/person-detection.service"
        print(f"다음 내용을 {service_path}에 저장하세요:")
        print(service_content)
        print("\n활성화 명령:")
        print("sudo systemctl enable person-detection.service")
        print("sudo systemctl start person-detection.service")
        return False
    
    return False


def disable_autostart() -> bool:
    """자동 실행 비활성화"""
    if sys.platform == "linux":
        print("자동 실행 비활성화:")
        print("sudo systemctl disable person-detection.service")
    return False
