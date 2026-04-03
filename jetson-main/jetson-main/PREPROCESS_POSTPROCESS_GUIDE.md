# YOLO 추론 파이프라인: 전처리/후처리 상세 설명

이 문서는 현재 프로젝트의 `ai/model.py` 기준으로, 추론 시 수행되는 전처리와 후처리를 정리합니다.
특히 "코드에서 직접 수행하는 처리"와 "Ultralytics가 내부적으로 자동 수행하는 처리"를 구분해서 설명합니다.

## 1) 전체 흐름

1. OpenCV 카메라 프레임 입력 (`frame`)
2. `YOLOInference.run_inference()` 호출
3. Ultralytics 내부 전처리 수행
4. 모델 추론
5. Ultralytics 내부 후처리 수행 (confidence threshold, NMS, 좌표 복원)
6. `YOLOInference.postprocess_results()`에서 person 필터링 + `Detection` 변환

---

## 2) 전처리(Preprocessing)

## 2-1. 코드에서 직접 수행하는 전처리

현재 코드에서는 별도 수동 전처리를 하지 않습니다.
즉, 입력 프레임(OpenCV BGR ndarray)을 그대로 모델에 전달합니다.

```python
return self.model(
    frame,
    conf=self.conf,
    imgsz=self.imgsz,
    half=INFER_HALF,
    classes=[0],
    verbose=False
)
```

## 2-2. Ultralytics 내부 전처리(자동)

Ultralytics는 전달받은 프레임을 모델 입력 형식으로 자동 변환합니다.
일반적으로 다음 과정이 포함됩니다.

1. Resize/Letterbox
- 입력 크기를 `imgsz`에 맞게 조정합니다.
- 종횡비(aspect ratio)를 가능한 유지하기 위해 패딩(letterbox)을 사용합니다.

2. 채널/배치 형태 정리
- 원본 OpenCV 배열은 보통 `HWC`(높이, 너비, 채널) 구조입니다.
- 배치 차원을 추가하여 모델 입력 형태로 만듭니다.
- 내부적으로 모델 연산용 텐서 형태(일반적으로 `NCHW`)로 변환됩니다.

3. dtype 변환
- 정수형(`uint8`) 이미지를 부동소수점으로 변환합니다.
- `half=True`이면 `float16`(FP16), 아니면 일반적으로 `float32`(FP32)를 사용합니다.

4. 정규화(normalization)
- 보통 픽셀 범위를 `0~255`에서 `0~1`로 스케일링합니다.

5. 디바이스 이동
- CPU 텐서를 GPU(CUDA) 또는 지정 장치로 이동시켜 추론합니다.

---

## 3) 후처리(Postprocessing)

## 3-1. Ultralytics 내부 후처리(자동)

모델이 출력한 raw prediction에 대해 Ultralytics가 기본 후처리를 수행합니다.

1. Confidence threshold 적용
- `conf=self.conf` 기준으로 낮은 신뢰도 박스를 제거합니다.

2. NMS(Non-Maximum Suppression)
- 같은 객체를 중복으로 감지한 박스들을 IoU 기준으로 정리해 대표 박스만 남깁니다.

3. 좌표 복원(원본 기준)
- 전처리 단계의 리사이즈/패딩에 맞춰 계산된 좌표를 원본 이미지 좌표계(`xyxy`)로 복원합니다.

## 3-2. 코드에서 직접 수행하는 후처리

`postprocess_results()`에서 프로젝트 목적에 맞게 결과를 한 번 더 정리합니다.

1. person 클래스만 필터링
- `classes=[0]`로 이미 1차 제한을 걸었지만,
- `ci == 0` 조건으로 2차 필터링을 다시 수행합니다.

2. 결과 텐서 CPU 복사 및 numpy 변환
- `boxes.data.cpu().numpy()`로 한 번에 가져와 처리합니다.

3. Detection 포맷 변환
- `[x1, y1, x2, y2, conf, cls]`를 읽어
- `Detection` TypedDict 형태로 변환합니다.

---

## 4) 핵심 용어 정리

## NHWC
- 텐서 축 순서가 `(N, H, W, C)`인 형식입니다.
- `N`: 배치 크기, `H`: 높이, `W`: 너비, `C`: 채널 수
- 일부 프레임워크/입력 파이프라인에서 주로 사용됩니다.

## NCHW
- 텐서 축 순서가 `(N, C, H, W)`인 형식입니다.
- PyTorch 기반 모델에서 일반적으로 선호되는 메모리/연산 레이아웃입니다.

## dtype
- 데이터 타입(data type)을 의미합니다.
- 이미지 입력에서는 주로 아래 타입이 중요합니다.
  - `uint8`: 0~255 정수 픽셀
  - `float32`: 일반적인 추론 부동소수점
  - `float16`: 메모리/연산량 감소(하드웨어 지원 시 속도 이점)

## BGR
- OpenCV 기본 채널 순서입니다.
- 채널 순서가 Blue, Green, Red 입니다.
- 일부 모델/라이브러리는 RGB를 기대하므로 내부 변환 여부가 중요합니다.

## NMS (Non-Maximum Suppression)
- 겹치는 다수의 바운딩 박스 중 가장 신뢰도 높은 박스를 중심으로 중복을 제거하는 알고리즘입니다.
- 객체 검출에서 중복 감지를 줄이는 핵심 단계입니다.

## Confidence threshold
- 예측 박스의 신뢰도 점수 하한선입니다.
- threshold를 높이면 오탐(false positive)은 줄 수 있지만, 미탐(false negative)은 늘 수 있습니다.

---

## 5) 이 프로젝트에서의 실무 포인트

1. `classes=[0]`
- person만 검출 대상으로 제한해 불필요한 클래스 연산/후처리를 줄입니다.

2. `half=INFER_HALF`
- Jetson 환경에서 FP16 사용 시 메모리 및 추론 속도에 유리할 수 있습니다.
- 단, 모델/하드웨어 조합에 따라 정확도/안정성 확인이 필요합니다.

3. 이중 person 필터링
- Ultralytics 인자 제한 + 코드 재필터링을 함께 사용해 안정성을 높입니다.

4. 출력 포맷 표준화
- 후단 파이프라인이 사용하기 쉬운 `Detection` 형태로 통일합니다.

---

## 6) 현재 코드 기준 체크리스트

- 전처리 수동 코드가 필요한가?
  - 현재는 Ultralytics 자동 전처리로 충분
- confidence threshold 값이 환경에 적절한가?
  - 오탐/미탐 균형 기준으로 조정 필요
- NMS 파라미터(IoU 등) 튜닝이 필요한가?
  - 밀집 장면에서 성능 영향 큼
- FP16 사용 시 정확도 저하 없는가?
  - 샘플 영상으로 비교 검증 권장

이 문서를 기준으로, 추후 TensorRT 전환 시에도
`run_inference()`와 `postprocess_results()` 책임 분리를 유지하면 변경 영향 범위를 줄일 수 있습니다.
