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
        
    def run_inference(self, frame: cv2.Mat, tracking: bool = False):
        """
        YOLO 모델 추론 실행

        Args:
            frame:    입력 프레임 (OpenCV BGR numpy array)
                      전처리(resize, normalize, tensor 변환)는 Ultralytics 내부에서 자동 처리
            tracking: True 이면 model.track(persist=True)로 객체 추적 활성화.
                      False 이면 기존 model() 단순 탐지 사용.
        Returns:
            Ultralytics Results 객체 리스트
        """
        # TensorRT 전환 시 이 메서드만 교체하면 됩니다
        # (예: self.model = torch2trt 또는 TRTModule)
        common_kwargs = dict(
            conf=self.conf,
            imgsz=self.imgsz,
            half=INFER_HALF,           # settings.py의 INFER_HALF 사용
            device=self.device,        # GPU 강제 지정 (설정 기반)
            classes=[TARGET_CLASS_ID], # NMS 전 person 클래스만 처리 → 후처리 부하 감소
            verbose=False,
        )
        if tracking:
            # persist=True: 프레임 간 동일 객체에 일관된 track_id 유지
            return self.model.track(frame, persist=True, **common_kwargs)
        return self.model(frame, **common_kwargs)

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

        # track_ids: run_inference(tracking=True) 시 boxes.id에 track ID가 담김
        # 단순 탐지 모드에서는 boxes.id 가 None
        raw_ids = boxes.id  # Tensor or None
        track_ids: Optional[List[Optional[int]]]
        if raw_ids is not None:
            track_ids = raw_ids.int().cpu().tolist()
        else:
            track_ids = None

        # boxes.xyxy / conf / cls 를 개별 속성으로 가져옴
        # (tracking 시 boxes.data 열 순서가 달라질 수 있어 명시적 접근 선호)
        xyxy = boxes.xyxy.cpu().numpy()          # (N, 4)
        confs = boxes.conf.cpu().numpy()         # (N,)
        clss = boxes.cls.cpu().numpy().astype(int)  # (N,)
        total = len(xyxy)

        class_name = (
            self.names[TARGET_CLASS_ID]
            if 0 <= TARGET_CLASS_ID < len(self.names)
            else str(TARGET_CLASS_ID)
        )

        for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
            if cls_id != TARGET_CLASS_ID:
                continue
            tid: Optional[int] = track_ids[i] if track_ids is not None else None
            detections.append(Detection(
                bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                confidence=float(conf),
                class_id=TARGET_CLASS_ID,
                class_name=class_name,
                track_id=tid,
            ))

        logger.debug(
            "객체 탐지 완료",
            {"num_detections": len(detections), "total_boxes": total}
        )

        return detections

    def predict(self, frame: cv2.Mat, tracking: bool = False) -> List[Detection]:
        """
        run_inference + postprocess_results 단일 호출 편의 메서드

        Args:
            frame:    입력 프레임
            tracking: True 이면 model.track() 기반 추적 활성화

        Returns:
            Detection 리스트 (tracking=True 시 각 항목에 track_id 포함)
        """
        results = self.run_inference(frame, tracking=tracking)
        return self.postprocess_results(results)
