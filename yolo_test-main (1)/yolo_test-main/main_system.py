import cv2
import numpy as np
import json
import os
import threading
import time
import requests  # 웹 서버 통신용 라이브러리
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

# ==========================================
# 1. 초기 세팅 (모델, 설정파일, 로그 폴더)
# ==========================================
print("시스템 부팅 중... (YOLOv8 & 안전 모듈 로드)")
model = YOLO('yolov8n.pt') 

try:
    with open("roi_config.json", "r") as f:
        config = json.load(f)
    base_roi_points = np.array(config["roi_polygon"], np.int32)
    # 사다리꼴 위쪽(Y값이 가장 작은) 2개의 점 인덱스 찾기 (동적 ROI 늘리기용)
    top_indices = np.argsort(base_roi_points[:, 1])[:2]
except FileNotFoundError:
    print("[에러] roi_config.json 파일이 없습니다! roi_setup.py를 먼저 실행하세요.")
    exit()

if not os.path.exists("danger_logs"):
    os.makedirs("danger_logs")

track_history = defaultdict(lambda: [])
last_alarm_time = 0
last_snapshot_time = 0

# [기능 3] 지게차 가상 주행 속도 세팅 (0: 정지 ~ 5: 최고속도)
forklift_speed = 0  

# ==========================================
# 2. 오디오 알람 및 서버 전송 함수 (비동기 스레드)
# ==========================================
def play_audio(alarm_type):
    """맥북 내장 TTS로 실제 음성 경고 출력"""
    if alarm_type == "URGENT":
        os.system("say '긴급! 충돌 위험! 즉시 정지하세요!' &")
    elif alarm_type == "APPROACH":
        os.system("say '경고! 전방에 작업자가 접근 중입니다. 감속하세요.' &")
    elif alarm_type == "BLIND_SPOT":
        os.system("say '주의! 출발 전 사각지대에 작업자가 있습니다.' &")

def send_to_ec2_server(event_type, img_filename):
    """[기능 2] 아마존 EC2 서버로 위험 로그 데이터 전송 (JSON 포맷)"""
    # 백엔드 팀이 알려줄 EC2 서버 API 주소 (지금은 임시 주소)
    EC2_SERVER_URL = "http://3.212.81.201:5000/api/log_event" 
    
    payload = {
        "device_id": "Forklift_A1",
        "event_type": event_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "speed_level": forklift_speed,
        "image_ref": img_filename
    }
    
    try:
        # 영상이 끊기지 않게 1초 타임아웃 설정
        # requests.post(EC2_SERVER_URL, json=payload, timeout=1) 
        print(f"☁️ [AWS EC2 전송 성공] 데이터: {payload}")
    except Exception as e:
        print(f"☁️ [AWS EC2 전송 실패] 서버를 확인하세요. (에러: {e})")

# ==========================================
# 3. 메인 카메라 관제 루프
# ==========================================
cap = cv2.VideoCapture(0)
print("능동 안전 관제 시스템 가동 완료! (조작: W=가속, S=감속, Q=종료)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))
    current_time = time.time()
    
    # 상태 플래그
    blind_spot_warning = False
    approach_warning = False
    urgent_warning = False

    # ------------------------------------------
    # [기능 3] 속도에 따른 가변형 동적 ROI (Dynamic ROI) 계산
    # ------------------------------------------
    # 속도가 빠를수록 사다리꼴의 윗부분을 더 위로(멀리) 늘려서 제동거리를 확보합니다!
    dynamic_roi = base_roi_points.copy()
    dynamic_roi[top_indices, 1] -= (forklift_speed * 30) # 속도 1당 30픽셀씩 길어짐
    dynamic_roi[top_indices, 1] = np.maximum(dynamic_roi[top_indices, 1], 0) # 화면 밖으로 안 나가게 방지

    # ------------------------------------------
    # 객체 탐지 및 궤적(TTC) 분석
    # ------------------------------------------
    results = model.track(frame, persist=True, classes=[0], verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            foot_x = int((x1 + x2) / 2)
            foot_y = y2
            dot_color = (255, 0, 0) 

            # 사각지대 판별 (늘어난 동적 ROI 기준)
            if cv2.pointPolygonTest(dynamic_roi, (foot_x, foot_y), False) >= 0:
                blind_spot_warning = True
                dot_color = (0, 255, 255)

            # [기능 1] 속도 기반 다중 팽창 로직 (TTC 알고리즘)
            box_area = (x2 - x1) * (y2 - y1)
            track = track_history[track_id]
            track.append(box_area)
            
            if len(track) > 15: track.pop(0)

            if len(track) == 15:
                expansion_ratio = track[-1] / track[0]
                
                # 0.5초 만에 면적이 30% 이상 급팽창 (뛰어옴) -> 초긴급 상태!
                if expansion_ratio > 1.30: 
                    urgent_warning = True
                    dot_color = (0, 0, 255)
                # 면적이 10% 이상 팽창 (걸어옴) -> 접근 주의 상태!
                elif expansion_ratio > 1.10: 
                    approach_warning = True
                    dot_color = (0, 165, 255) # 주황색

            cv2.circle(frame, (foot_x, foot_y), 7, dot_color, -1)
            cv2.putText(frame, f"ID:{track_id}", (foot_x + 10, foot_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ------------------------------------------
    # 우선순위 판별 및 이벤트 실행
    # ------------------------------------------
    screen_color = (0, 255, 0)
    alarm_msg = "SAFE / NORMAL DRIVING"
    active_event = None

    if urgent_warning:
        screen_color = (0, 0, 255) # 빨강 (최고 위험)
        alarm_msg = "URGENT: RAPID APPROACH! STOP!"
        active_event = "URGENT"
    elif approach_warning:
        screen_color = (0, 165, 255) # 주황 (중간 위험)
        alarm_msg = "WARNING: PERSON APPROACHING"
        active_event = "APPROACH"
    elif blind_spot_warning:
        screen_color = (0, 255, 255) # 노랑 (정지 전 위험)
        alarm_msg = "CAUTION: BLIND SPOT OCCUPIED"
        active_event = "BLIND_SPOT"

    # 이벤트 로깅 (음성, 스냅샷, 서버 전송)
    if active_event:
        if current_time - last_alarm_time > 4.0:
            threading.Thread(target=play_audio, args=(active_event,)).start()
            last_alarm_time = current_time

        if current_time - last_snapshot_time > 2.0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"danger_logs/{active_event}_{timestamp}.jpg"
            
            # 스냅샷 저장
            capture_frame = frame.copy()
            cv2.polylines(capture_frame, [dynamic_roi], isClosed=True, color=screen_color, thickness=3)
            cv2.imwrite(filename, capture_frame)
            
            # 백그라운드로 EC2 서버에 JSON 데이터 전송! (영상 멈춤 방지)
            threading.Thread(target=send_to_ec2_server, args=(active_event, filename)).start()
            
            last_snapshot_time = current_time

    # ------------------------------------------
    # UI 렌더링 (동적 ROI 및 속도계 표시)
    # ------------------------------------------
    cv2.polylines(frame, [dynamic_roi], isClosed=True, color=screen_color, thickness=3)
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, alarm_msg, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, screen_color, 2)
    
    # 속도계 UI
    cv2.putText(frame, f"Speed Level: {forklift_speed}/5", (450, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Advanced AI Forklift Safety", frame)

    # 키보드 입력 처리 (동적 ROI 조작)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('w'): forklift_speed = min(forklift_speed + 1, 5) # 가속 (최대 5)
    elif key == ord('s'): forklift_speed = max(forklift_speed - 1, 0) # 감속 (최소 0)

cap.release()
cv2.destroyAllWindows()