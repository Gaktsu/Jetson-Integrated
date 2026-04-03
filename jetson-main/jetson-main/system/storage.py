"""
저장소 용량 관리
"""
import os
import shutil
from typing import List
from config.settings import SAVE_DIR


def get_disk_usage(path: str = SAVE_DIR) -> dict:
    """
    디스크 사용량 확인
    
    Args:
        path: 확인할 경로
    
    Returns:
        {'total': total_bytes, 'used': used_bytes, 'free': free_bytes, 'percent': percent}
    """
    usage = shutil.disk_usage(path)
    return {
        'total': usage.total,
        'used': usage.used,
        'free': usage.free,
        'percent': (usage.used / usage.total) * 100
    }


def get_directory_size(path: str = SAVE_DIR) -> int:
    """
    디렉토리 크기 계산 (바이트)
    
    Args:
        path: 디렉토리 경로
    
    Returns:
        총 크기 (bytes)
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def list_old_files(path: str = SAVE_DIR, limit: int = 10) -> List[tuple]:
    """
    오래된 파일 목록 반환
    
    Args:
        path: 디렉토리 경로
        limit: 반환할 파일 개수
    
    Returns:
        [(파일경로, 수정시간), ...] 리스트 (오래된 순)
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                mtime = os.path.getmtime(filepath)
                files.append((filepath, mtime))
    
    # 오래된 순 정렬
    files.sort(key=lambda x: x[1])
    return files[:limit]


def cleanup_old_files(path: str = SAVE_DIR, threshold_percent: float = 80.0) -> int:
    """
    디스크 용량이 임계값을 초과하면 오래된 파일 삭제
    
    Args:
        path: 디렉토리 경로
        threshold_percent: 임계값 (%)
    
    Returns:
        삭제된 파일 개수
    """
    usage = get_disk_usage(path)
    if usage['percent'] < threshold_percent:
        return 0
    
    deleted_count = 0
    old_files = list_old_files(path, limit=50)
    
    for filepath, mtime in old_files:
        try:
            os.remove(filepath)
            deleted_count += 1
            print(f"삭제됨: {filepath}")
            
            # 용량 재확인
            usage = get_disk_usage(path)
            if usage['percent'] < threshold_percent:
                break
        except Exception as e:
            print(f"삭제 실패: {filepath}, 오류: {e}")
    
    return deleted_count


def format_bytes(bytes_size: int) -> str:
    """바이트를 읽기 쉬운 형식으로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"
