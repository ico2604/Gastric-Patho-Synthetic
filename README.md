# 🔬 Gastric-Patho-Synthetic (위암 병리 멀티모달 프로젝트)
> **위암 병리 이미지(Segmentation) 및 합성 판독문(NLP) 통합 분석**

본 프로젝트는 AI_Hub의 '위암 병리 이미지 및 판독문 합성데이터'를 활용하여, 위암의 4가지 주요 병리 분류에 따른 세포 분할 및 진단 데이터 분석을 수행합니다.

---

## 📊 1. 데이터셋 구성 (Dataset Statistics)
총 10,000장의 고해상도(1024x1024) PNG 이미지와 JSON 어노테이션으로 구성되어 있습니다.

| 데이터 분류 | 데이터 형식 | 수량 (Patches) |
| :--- | :--- | :--- |
| **위염 (STNT)** | PNG : JSON | 2,500 |
| **형선암 (STIN)** | PNG : JSON | 2,500 |
| **미만형선암 (STDI)** | PNG : JSON | 2,500 |
| **혼합형선암 (STMX)** | PNG : JSON | 2,500 |

## 엉 2. 어노테이션 구조 (Annotation Format)
데이터 로더(`data_loader.py`) 구현 시 다음의 JSON 구조를 참조합니다.

### 2.1 Clinical Information (임상 정보)
- `tumor_code`: "STOP" (위암 병리 코드)
- `category`: 병리 분류 (STNT, STIN, STDI, STMX)
- `tumor_category`: "normal" (위염) / "abnormal" (선암 3종)
- `diagnosis`: 상세 진단명 및 판독문 텍스트

### 2.2 Image & Object Information (이미지 및 객체)
- **Size**: 1024 x 1024 (MPP 정보 포함)
- **Type**: 합성 데이터(S) 위주 구성
- **Segmentation Label**:
  - `Tumor`: 종양 세포
  - `Stroma`: 기질
  - `Normal`: 정상 조직
  - `Immune`: 면역 세포
- **Format**: `polygon` 좌표 기반 [X, Y] 리스트

## 👨‍💻 프로젝트 팀 및 역할
- **팀원 1**: 데이터 파이프라인 구축 및 JSON 파싱 (Polygon to Mask 변환)
- **팀원 2**: Vision 모델 구현 (U-Net++, DeepLabV3+ 등)
- **팀원 3**: NLP 모델 구현 (판독문 텍스트 기반 분류 및 생성 연구)
- **팀원 4**: 멀티모달 통합 및 성능 평가 (mIoU, Dice Score, F1-Score)

## 📁 디렉토리 구조
```text
├── data/
├── Training/
│   ├── 01.원천데이터/          # 학습용 PNG 이미지
│   │   ├── TS_미만형선암/
│   │   ├── TS_위염/
│   │   ├── TS_장형선암/
│   │   └── TS_혼합형선암/
│   └── 02.라벨링데이터/        # 학습용 JSON 어노테이션
│       ├── TS_미만형선암/
│       ├── TS_위염/
│       ├── TS_장형선암/
│       └── TS_혼합형선암/
└── Validation/
    ├── 01.원천데이터/          # 검증용 PNG 이미지
    │   ├── VS_미만형선암/
    │   ├── VS_위염/
    │   ├── VS_장형선암/
    │   └── VS_혼합형선암/
    └── 02.라벨링데이터/        # 검증용 JSON 어노테이션
        ├── VS_미만형선암/
        ├── VS_위염/
        ├── VS_장형선암/
        └── VS_혼합형선암/
├── src/
│   ├── utils/
│   │   └── mask_utils.py # Polygon 좌표를 binary mask로 변환하는 모듈
│   ├── train_vision.py
│   └── train_nlp.py
├── requirements.txt
└── README.md
