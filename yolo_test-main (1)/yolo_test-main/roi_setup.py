import cv2
import numpy as np
import json

points = []

def draw_roi(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"[{len(points)}/4] 좌표 저장됨: ({x}, {y})")

def main():
    global points
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("ROI Calibration Tool")
    cv2.setMouseCallback("ROI Calibration Tool", draw_roi)

    print("=== ROI 캘리브레이션 툴 ===")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # ★ 핵심 추가: 메인 시스템과 똑같이 화면을 640x480으로 줄인 상태에서 점을 찍습니다!
        frame = cv2.resize(frame, (640, 480))

        for p in points:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], (0, 255, 0), 2)
        
        if len(points) == 4:
            cv2.line(frame, points[3], points[0], (0, 255, 0), 2)
            cv2.putText(frame, "SAVED! Press 'Q' to quit.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("ROI Calibration Tool", frame)

        if len(points) == 4:
            with open("roi_config.json", "w") as f:
                json.dump({"roi_polygon": points}, f)
            cv2.waitKey(2000)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()