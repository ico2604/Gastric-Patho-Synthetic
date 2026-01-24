# 🔬 Gastric-Patho-Synthetic (위암 병리 멀티모달 프로젝트)
> **위암 병리 이미지(Segmentation) 및 합성 판독문(NLP) 통합 분석**

본 프로젝트는 AI_Hub의 '위암 병리 이미지 및 판독문 합성데이터'를 활용하여, 위암의 4가지 주요 병리 분류에 따른 세포 분할 및 진단 데이터 분석을 수행합니다.

---

## 📊 1. 데이터셋 구성 (Dataset Statistics)
- **총 수량**: 고해상도(1024x1024) 이미지 및 JSON 10,000 세트
- **병리 분류 (Clf)**: 위염(STNT), 장형선암(STIN), 미만형선암(STDI), 혼합형선암(STMX) 각 2,500장
- **세포 분할 (Seg)**: 종양(Tumor), 기질(Stroma), 정상(Normal), 면역(Immune) 4개 클래스

| 데이터 분류 | 데이터 형식 | 수량 (Patches) |
| :--- | :--- | :--- |
| **위염 (STNT)** | PNG : JSON | 2,500 |
| **형선암 (STIN)** | PNG : JSON | 2,500 |
| **미만형선암 (STDI)** | PNG : JSON | 2,500 |
| **혼합형선암 (STMX)** | PNG : JSON | 2,500 |

## 🏗️ 2. 모델 라인업 (Model Zoo)
| Task | Models | Framework |
| :--- | :--- | :--- |
| **Classification** | ResNet50, EfficientNet-V2, Swin-T, DenseNet121 | PyTorch 2.4.1 |
| **Segmentation** | U-Net, U-Net++, DeepLabV3+, SAM | PyTorch 2.4.1 |

## 📁 3. 디렉토리 구조 (Directory Structure)
```text
Gastric-Patho-Synthetic/
├── data/
│   ├── raw/                 # AI_Hub 원본 데이터 (Raw Data)
│   │   ├── Training/        # [원천데이터/라벨링데이터 하위 구조 생략]
│   │   └── Validation/
│   └── processed/           # 전처리 완료된 데이터 (학습에 직접 사용)
│       ├── images_512/      # 512x512 리사이징 이미지 (분류용)
│       └── masks/           # JSON을 변환한 Binary Mask PNG (세그멘테이션용)
├── checkpoints/             # 학습된 모델(.pth) 저장소
├── src/
│   ├── data_loader.py       # JSON 파싱 및 Mask 변환 핵심 모듈
│   ├── preprocess.py        # 리사이징 및 마스크 생성 자동화 스크립트
│   ├── classification/      # 분류 팀: models.py, train.py, val.py
│   └── segmentation/        # 세그멘테이션 팀: models.py, train.py, val.py
├── requirements.txt
└── README.md
