import cv2
import numpy as np

# [1] 외부 라이브러리 불러오기 및 환경 테스트
# AI 객체 인식 모델인 YOLO 라이브러리를 불러옵니다.
try:
    from ultralytics import YOLO
    HAS_YOLO = True # 불러오기 성공
except Exception as e:
    HAS_YOLO = False # 윈도우 환경 등에서 에러가 나면 AI 기능을 끄고 진행
    print(f"AI 모듈을 건너뜁니다 (윈도우 테스트 모드). 사유: {e}")

# 젯슨 나노의 하드웨어 핀(부저 등)을 제어하는 라이브러리를 불러옴
try:
    import Jetson.GPIO as GPIO
    HAS_GPIO = True # 불러오기 성공
except Exception as e:
    HAS_GPIO = False # 일반 PC 환경에서는 하드웨어 제어 기능을 끄고 진행
    print(f"Jetson.GPIO 모듈을 건너뜁니다. 사유: {e}")


# [2] 위험 감지기 메인 클래스
class DangerDetector:
    def __init__(self, model_path='ai_module/best.pt', buzzer_pin=18):
        # 환경 설정 상태와 부저 핀 번호를 클래스 변수로 저장
        self.has_yolo = HAS_YOLO
        self.has_gpio = HAS_GPIO
        self.buzzer_pin = buzzer_pin
        
        # UI에서 AI 감지 기능을 켜고 끌 수 있도록 상태를 저장하는 변수
        self.detection_enabled = True

        # 4개의 카메라 각각의 기본 좌표(사다리꼴)를 독립적으로 생성
        # 화면 정중앙 쯤에 임시 박스를 그림
        default_roi = [[220, 340], [420, 340], [420, 140], [220, 140]]
        # 딕셔너리 형태로 {0: 좌표, 1: 좌표, 2: 좌표, 3: 좌표} 를 만듬
        self.rois = {i: np.array(default_roi, np.int32) for i in range(4)}

        # AI 모델이 사용 가능하다면 학습된 파일(best.pt)을 불러와 준비
        if self.has_yolo:
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"YOLO 모델 로드 실패: {e}")
                self.has_yolo = False

        # 하드웨어 제어가 가능하면 부저 핀을 출력 모드로 설정하고 소리를 끔
        if self.has_gpio:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.buzzer_pin, GPIO.OUT)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
            except Exception as e:
                print(f"GPIO 초기화 실패: {e}")
                self.has_gpio = False


    # [3] 설정값 연동 함수들
    # UI 설정창에서 사용자가 그린 새로운 4개의 점 좌표를 받아 업데이트
    def set_roi_points(self, cam_idx, points):
        if points is not None and len(points) == 4:
            self.rois[cam_idx] = np.array(points, np.int32)
            print(f"CAM {cam_idx+1} 위험 구역 좌표 업데이트 완료: {self.rois[cam_idx]}")

    # UI 설정창을 열 때 기존에 그려둔 좌표를 넘김
    def get_roi_points(self, cam_idx):
        return self.rois[cam_idx].tolist()

    # UI 설정창의 토글 버튼에 맞춰 AI 감지 기능을 켜거나 끔
    def set_detection_enabled(self, enabled):
        self.detection_enabled = enabled
        print(f"위험 구역 감지 기능: {'켜짐' if enabled else '꺼짐'}")
        
        # 기능을 끌 때 혹시라도 울리고 있던 부저 소리를 확실하게 끔
        if not enabled and self.has_gpio:
            GPIO.output(self.buzzer_pin, GPIO.LOW)


    # [4] 실시간 영상 분석 및 그리기 메인 함수
    # 실시간 카메라 프레임을 한 장씩 넘겨받아 사람을 찾고 선을 그림
    def process_frame(self, frame, cam_idx):
        # AI 모듈이 없다면 그냥 원본 사진과 위험하지 않음(False)을 그대로 돌려보냄
        if not self.has_yolo:
            return frame, False

        danger_detected = False
        current_roi = self.rois[cam_idx] # 현재 검사 중인 카메라의 구역 좌표를 가져옴
        
        # 영상 위에 하늘색 선으로 우리가 설정한 위험 구역을 그림
        try:
            cv2.polylines(frame, [current_roi], True, (255, 255, 0), 2)
        except Exception:
            # 윈도우 환경 등에서 그림 그리기에 실패하면 텍스트로 대체 표시
            if frame.shape[0] > 100:
                cv2.putText(frame, "[AI ROI SET]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # AI 감지 기능이 꺼져있다면 구역 선만 그린 채로 바로 종료
        if not self.detection_enabled:
            return frame, False

        # AI 감지 진행
        try:
            # YOLO 모델에 사진을 넣고 사람이나 물체를 찾음
            results = self.model(frame, stream=True, verbose=False)
            
            for r in results:
                for box in r.boxes: # 찾은 물체들의 네모 박스 정보 반복문
                    # 박스의 좌상단(x1, y1)과 우하단(x2, y2) 좌표 추출
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 물체의 하단 중앙 좌표(cx, cy)를 계산합니다. (사람의 발 위치 기준)
                    cx, cy = (x1 + x2) // 2, y2 

                    # 사람의 발 위치(cx, cy)가 위험 구역(current_roi) 다각형 안쪽에 있는지 수학적으로 판별
                    is_inside = cv2.pointPolygonTest(current_roi, (cx, cy), False) >= 0
                    
                    if is_inside:
                        # 선 안에 들어왔다면 위험 감지 플래그를 참(True)으로 바꿈
                        danger_detected = True
                        # 사람 주변에 빨간색 박스와 DANGER 경고 문구를 그림
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "DANGER!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # 선 밖에 있다면 안전하므로 초록색 박스를 그림
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception:
            # 윈도우 환경에서 연산 에러가 나면 조용히 무시하고 넘어감
            pass 

        # 하드웨어 제어
        if self.has_gpio:
            if danger_detected:
                GPIO.output(self.buzzer_pin, GPIO.HIGH) # 위험하면 소리 켬
            else:
                GPIO.output(self.buzzer_pin, GPIO.LOW)  # 안전하면 소리 끔

        # 선과 박스가 모두 그려진 그림(frame)과, 현재 위험한지 여부(danger_detected)를 돌려줌
        return frame, danger_detected

    # 프로그램 종료 시 하드웨어 부저 장치를 안전하게 해제
    def cleanup(self):
        if self.has_gpio:
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            GPIO.cleanup()