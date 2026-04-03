import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading
import time

ROI_POINTS = np.array([[100, 480], [540, 480], [450, 150], [190, 150]], np.int32)

ALARM_COOLDOWN = 2.0 
last_alarm_time = 0

def play_alarm_jetson():
    """
    젯슨(Linux)용 경고음 함수
    방법 1: 'espeak' 패키지를 설치해서 TTS 사용 (추천)
    방법 2: 'aplay'로 미리 준비된 mp3/wav 파일 재생
    """

    os.system('espeak "Danger" --stdout | aplay') 


print("모델 로딩 중...")
model = YOLO('best.pt') 

model.to('cuda') 
print("모델이 CUDA(GPU)에 로드되었습니다.")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. (USB 연결 확인)")
    exit()

print("실행 중... 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break


    results = model(frame, stream=True, verbose=False)

    roi_color = (0, 255, 0) 
    alert_triggered = False

    overlay = frame.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                foot_x = (x1 + x2) // 2
                foot_y = y2 

                is_inside = cv2.pointPolygonTest(ROI_POINTS, (foot_x, foot_y), False)

                if is_inside >= 0:
                    alert_triggered = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "DANGER!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if alert_triggered:
        roi_color = (0, 0, 255) 
        current_time = time.time()
        if current_time - last_alarm_time > ALARM_COOLDOWN:
            threading.Thread(target=play_alarm_jetson).start()
            last_alarm_time = current_time

    cv2.fillPoly(overlay, [ROI_POINTS], color=roi_color)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    cv2.imshow('Forklift Safety - Jetson', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()