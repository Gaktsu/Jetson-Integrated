"""
AI model wrapper module.
"""
from __future__ import annotations

import cv2
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger, EventType
from ai.detector import Detection
from config.settings import CONFIDENCE_THRESHOLD, INFER_DEVICE, INFER_HALF, INFER_IMGSZ, MODEL_PATH, TARGET_CLASS_ID

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

logger = get_logger("ai.model")

def load_model() -> Optional["YOLOInference"]:
    """YOLO 모델 로드. 실패 시 None 반환.
    TensorRT로 전환 시 YOLOInference.run_inference()만 교체하면 됩니다.
    """
    try:
        logger.event_info(EventType.MODULE_INIT, "YOLO 모델 로드 중", {"model_path": MODEL_PATH})
        model = YOLOInference(MODEL_PATH, CONFIDENCE_THRESHOLD, INFER_IMGSZ)
        logger.event_info(EventType.MODULE_INIT, "YOLO 모델 로드 완료")
        return model
    except Exception as e:
        logger.event_error(EventType.ERROR_OCCURRED, "YOLO 모델 로드 실패",
                           {"error": str(e)}, exc_info=True)
        return None

class YOLOInference:
    """YOLO 추론 클래스"""
    
    def __init__(self, model_path: str, conf: float = 0.5, imgsz: int = 640):
        if YOLO is None:
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "ultralytics 패키지가 설치되지 않음"
            )
            raise ImportError("ultralytics 패키지가 설치되지 않았습니다.")
        
        logger.event_info(
            EventType.MODULE_INIT,
            "YOLO 모델 초기화 시작",
            {"model_path": model_path, "conf": conf, "imgsz": imgsz}
        )
        
        self.model = YOLO(model_path)

        # ── 디바이스 강제 지정 및 GPU 사용 검증 ──
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            if INFER_DEVICE.startswith("cuda") and not cuda_ok:
                logger.event_error(
                    EventType.ERROR_OCCURRED,
                    "CUDA 사용 불가 — CPU로 fallback 발생 (GPU 환경 확인 필요)",
                    {"requested_device": INFER_DEVICE, "cuda_available": False},
                )
                self.device = "cpu"
            else:
                self.device = INFER_DEVICE
            logger.event_info(
                EventType.MODULE_INIT,
                "추론 디바이스 확정",
                {"device": self.device, "cuda_available": cuda_ok},
            )
        except Exception as e:
            logger.event_error(EventType.ERROR_OCCURRED, "디바이스 설정 실패", {"error": str(e)})
            self.device = "cpu"

        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names if hasattr(self.model, "names") else []
        
        logger.event_info(
            EventType.MODULE_INIT,
            "YOLO 모델 초기화 완료",
            {"num_classes": len(self.names)}
        )
        
    def run_inference(self, frame: cv2.Mat):
        """
        YOLO 모델 추론 실행

        Args:
            frame: 입력 프레임 (OpenCV BGR numpy array)
                   전처리(resize, normalize, tensor 변환)는 Ultralytics 내부에서 자동 처리
        Returns:
            Ultralytics Results 객체 리스트
        """
        # TensorRT 전환 시 이 메서드만 교체하면 됩니다
        # (예: self.model = torch2trt 또는 TRTModule)
        return self.model(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            half=INFER_HALF,                  # settings.py의 INFER_HALF 사용
            device=self.device,               # GPU 강제 지정 (설정 기반)
            classes=[TARGET_CLASS_ID],        # NMS 전 person 클래스만 처리 → 후처리 부하 감소
            verbose=False
        )

    def postprocess_results(self, results) -> List[Detection]:
        """
        Ultralytics Results 객체를 Detection 리스트로 변환

        Ultralytics 내부 처리 항목:
            - confidence threshold 필터링 (conf 파라미터)
            - NMS
            - bbox 좌표 역스케일 (원본 이미지 기준 xyxy)
        직접 처리 항목:
            - person 클래스(cls_id=0)만 필터링
            - Detection TypedDict 변환

        Args:
            results: run_inference()의 반환값

        Returns:
            Detection 리스트
        """
        if not results:
            return []

        r = results[0]
        boxes = r.boxes
        detections: List[Detection] = []

        if boxes is None or len(boxes) == 0:
            return detections

        # boxes.data: (N, 6) — [x1, y1, x2, y2, conf, cls]
        # 단일 GPU→CPU 전송으로 xyxy/conf/cls를 한 번에 처리
        data = boxes.data.cpu().numpy()
        total = len(data)

        # numpy 벡터화: TARGET_CLASS_ID 마스크를 한 번에 적용 (Python 루프 최소화)
        mask = data[:, 5].astype(int) == TARGET_CLASS_ID
        person_rows = data[mask]
        class_name = (
            self.names[TARGET_CLASS_ID]
            if 0 <= TARGET_CLASS_ID < len(self.names)
            else str(TARGET_CLASS_ID)
        )

        for row in person_rows:
            detections.append(Detection(
                bbox=(int(row[0]), int(row[1]), int(row[2]), int(row[3])),
                confidence=float(row[4]),
                class_id=TARGET_CLASS_ID,
                class_name=class_name,
            ))

        logger.debug(
            "객체 탐지 완료",
            {"num_detections": len(detections), "total_boxes": total}
        )

        return detections

    def predict(self, frame: cv2.Mat) -> List[Detection]:
        """
        run_inference + postprocess_results 단일 호출 편의 메서드

        Args:
            frame: 입력 프레임

        Returns:
            Detection 리스트
        """
        results = self.run_inference(frame)
        return self.postprocess_results(results)
