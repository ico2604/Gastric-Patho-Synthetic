import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from matplotlib.patheffects import withStroke

# 실행 시점 타임스탬프
CURRENT_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from data_loader import get_loader, get_transforms
import segmentation_models_pytorch as smp

# --- 1. 환경 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "unet_resnet50_best.pth") # 파일명 확인
RESULT_DIR = os.path.join(BASE_DIR, "logs", "seg", "test", CURRENT_TIME) 
os.makedirs(RESULT_DIR, exist_ok=True)

# 시각화용 색상 체계 (Train과 동일하게 설정)
CLASS_INFO = {
    1: {"name": "Tumor", "color": "#FF0000"},   # Red
    2: {"name": "Stroma", "color": "#0000FF"},  # Blue
    3: {"name": "Normal", "color": "#00FF00"},  # Green
    4: {"name": "Immune", "color": "#FF00FF"}   # Magenta (진분홍)
}

# --- 2. 모델 로드 ---
model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=NUM_CLASSES).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ 모델 로드 성공: {MODEL_PATH}")
else:
    print(f"❌ 모델을 찾을 수 없습니다: {MODEL_PATH}")
    sys.exit()

# --- 3. 데이터로더 ---
test_loader = get_loader(
    img_dir=os.path.join(BASE_DIR, "data", "processed", "images_512", "Validation"), # Test 폴더가 따로 없다면 Validation 활용
    mask_dir=os.path.join(BASE_DIR, "data", "processed", "masks", "Validation"),
    json_dir=None,
    batch_size=1, shuffle=False, transform=get_transforms(is_train=False), task='seg'
)

# --- 4. 시각화 보조 함수 ---
def plot_overlay(ax, img, mask, alpha=0.5):
    """이미지 위에 마스크를 RGBA 방식으로 얹는 함수"""
    ax.imshow(img)
    for cls_idx, info in CLASS_INFO.items():
        m = (mask == cls_idx)
        if np.any(m):
            colored_mask = np.zeros((*mask.shape, 4))
            col = plt.cm.colors.to_rgb(info['color'])
            colored_mask[m] = [*col, alpha]
            ax.imshow(colored_mask, interpolation='nearest')

# --- 5. 테스트 및 시각화 루프 ---
print(f"🔬 Test를 시작합니다. 대상: {len(test_loader)} 이미지")
total_dice = 0

with torch.no_grad():
    for i, (image, mask, _) in enumerate(tqdm(test_loader)):
        image, mask = image.to(DEVICE), mask.to(DEVICE).long()
        output = model(image)
        
        # 지표 계산
        pred_labels = torch.argmax(output, dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(pred_labels.unsqueeze(1), mask.unsqueeze(1), mode='multiclass', num_classes=NUM_CLASSES)
        dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        total_dice += dice

        # 시각화 리포트 생성 (상위 10개)
        if i < 10:
            img_np = image[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            gt_np = mask[0].cpu().numpy()
            pred_np = pred_labels[0].cpu().numpy()

            fig = plt.figure(figsize=(24, 12), facecolor='white')
            
            # [1행] 요약 섹션
            titles = ["Original H&E Slide", "Expert GT (Annotation)", f"AI Prediction (Dice: {dice:.4f})"]
            for idx in range(3):
                ax = plt.subplot(2, 4, idx+1)
                if idx == 0: ax.imshow(img_np)
                elif idx == 1: plot_overlay(ax, img_np, gt_np)
                elif idx == 2: plot_overlay(ax, img_np, pred_np)
                
                t = ax.set_title(titles[idx], fontsize=16, fontweight='bold', pad=15)
                t.set_path_effects([withStroke(linewidth=3, foreground='white')])
                ax.axis('off')

            # [2행] 클래스별 상세 섹션 (Focus)
            for idx, (cls_idx, info) in enumerate(CLASS_INFO.items()):
                ax = plt.subplot(2, 4, 5 + idx)
                ax.imshow(img_np)
                
                class_mask = (pred_np == cls_idx)
                if np.any(class_mask):
                    colored_mask = np.zeros((*pred_np.shape, 4))
                    col = plt.cm.colors.to_rgb(info['color'])
                    colored_mask[class_mask] = [*col, 0.7] # 개별 컷은 더 진하게
                    ax.imshow(colored_mask, interpolation='nearest')
                
                t = ax.set_title(f"Focus: {info['name']}", fontsize=15, fontweight='bold', color=info['color'])
                t.set_path_effects([withStroke(linewidth=4, foreground='white')])
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, f"test_result_{i}.png"), bbox_inches='tight', dpi=150)
            plt.close()

print(f"\n✅ 최종 Test Dice Score: {total_dice/len(test_loader):.4f}")
print(f"📂 시각화 결과 저장 완료: {RESULT_DIR}")