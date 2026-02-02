import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.colors import ListedColormap
from matplotlib.patheffects import withStroke

# --- 0. 경로 및 전역 설정 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

# 실행 시점 타임스탬프
CURRENT_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# 전역 파일/폴더명 설정
MODEL_NAME = "unet_resnet50_best.pth"
HISTORY_CSV = "train_history.csv"
CURVE_PLOT = "learning_curves.png"

# 디렉토리 설정
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs", "seg", "train", CURRENT_TIME)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 데이터 경로
IMG_TRAIN_DIR = os.path.join(BASE_DIR, "data", "processed", "images_512", "Training")
IMG_VAL_DIR = os.path.join(BASE_DIR, "data", "processed", "images_512", "Validation")
MASK_TRAIN_DIR = os.path.join(BASE_DIR, "data", "processed", "masks", "Training")
MASK_VAL_DIR = os.path.join(BASE_DIR, "data", "processed", "masks", "Validation")
JSON_TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "Training", "02.라벨링데이터")
JSON_VAL_DIR = os.path.join(BASE_DIR, "data", "raw", "Validation", "02.라벨링데이터")

from data_loader import get_loader, get_transforms
import segmentation_models_pytorch as smp

# --- 1. 하이퍼파라미터 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 30      # Early Stopping이 있으므로 넉넉히 설정
LR = 1e-4
NUM_CLASSES = 5  # 0:Background, 1:Tumor, 2:Stroma, 3:Normal, 4:Immune
PATIENCE = 7     # Early Stopping 기준

# --- 2. 모델 / 손실 함수 / 최적화 ---
model = smp.Unet(
    encoder_name="resnet50", 
    encoder_weights="imagenet", 
    in_channels=3, 
    classes=NUM_CLASSES 
).to(DEVICE)

criterion = smp.losses.DiceLoss(mode='multiclass')
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- 4. 데이터로더 ---
train_loader = get_loader(IMG_TRAIN_DIR, MASK_TRAIN_DIR, JSON_TRAIN_DIR, BATCH_SIZE, get_transforms(is_train=True), task='seg')
val_loader = get_loader(IMG_VAL_DIR, MASK_VAL_DIR, JSON_VAL_DIR, BATCH_SIZE, get_transforms(is_train=False), task='seg')

# --- 5. 지표 계산 함수 ---
def calculate_dice(preds, masks):
    pred_labels = torch.argmax(preds, dim=1).unsqueeze(1)
    mask_labels = masks.unsqueeze(1).long()
    
    tp, fp, fn, tn = smp.metrics.get_stats(pred_labels, mask_labels, mode='multiclass', num_classes=NUM_CLASSES)
    dice_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    return dice_score

# --- 6. 시각화 함수 ---
def save_sample_results(epoch, images, masks, preds):
    """
    GT와 AI 예측 결과의 가시성을 극대화하여 시각화
    """
    # 배경(H&E)이 레드/보라 계열이므로 대비가 강한 네온/진한 색상 사용
    class_info = {
        1: {"name": "Tumor", "color": "#FF0000"},   # 밝은 빨강
        2: {"name": "Stroma", "color": "#0000FF"},  # 진한 파랑
        3: {"name": "Normal", "color": "#00FF00"},  # 형광 초록
        4: {"name": "Immune", "color": "#FF00FF"}   # 진한 마젠타 (하늘색보다 훨씬 잘 보임)
    }
    
    img = images[0].permute(1, 2, 0).cpu().numpy()
    # 정규화 해제 및 이미지 클리핑 (이미지가 너무 밝거나 어둡지 않게)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    gt = masks[0].cpu().numpy()
    pred = torch.argmax(preds[0], dim=0).detach().cpu().numpy()

    fig = plt.figure(figsize=(24, 12), facecolor='white')
    
    titles = ["Original H&E Slide", "Expert Annotation (GT Overlay)", "AI Prediction (Total Overlay)"]
    
    for i in range(3):
        ax = plt.subplot(2, 4, i+1)
        # 1. 배경 이미지 먼저 출력
        ax.imshow(img)
        
        target_mask = gt if i == 1 else pred if i == 2 else None
        
        # 2. 그 위에 마스크를 루프 돌며 '얹음'
        if target_mask is not None:
            for cls_idx, info in class_info.items():
                m = (target_mask == cls_idx)
                if np.any(m):
                    # 마스크가 있는 부분만 색상을 칠함 (nan을 사용하여 배경 투과)
                    colored_mask = np.zeros((*target_mask.shape, 4)) # RGBA
                    col = plt.cm.colors.to_rgb(info['color'])
                    colored_mask[m] = [*col, 0.5] # 0.5는 투명도(Alpha)
                    ax.imshow(colored_mask, interpolation='nearest')
        
        t = ax.set_title(titles[i], fontsize=16, fontweight='bold', pad=15)
        t.set_path_effects([withStroke(linewidth=3, foreground='white')])
        ax.axis('off')

    # [2행 상세] 각 클래스별 Focus
    for i, (cls_idx, info) in enumerate(class_info.items()):
        ax = plt.subplot(2, 4, 5 + i)
        ax.imshow(img)
        
        class_mask = (pred == cls_idx)
        if np.any(class_mask):
            colored_mask = np.zeros((*pred.shape, 4))
            col = plt.cm.colors.to_rgb(info['color'])
            colored_mask[class_mask] = [*col, 0.7] # 개별 컷은 더 진하게(0.7)
            ax.imshow(colored_mask, interpolation='nearest')
            
        t = ax.set_title(f"{cls_idx}: {info['name']}", fontsize=15, fontweight='bold', color=info['color'])
        t.set_path_effects([withStroke(linewidth=4, foreground='white')])
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, f"epoch_{epoch}_detailed.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"📸 가시성 개선 리포트 저장 완료: {save_path}")

def plot_learning_curves(history_csv_path, save_path):
    df = pd.read_csv(history_csv_path)
    epochs = range(1, len(df) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, df['train_loss'], label='Train Loss')
    plt.plot(epochs, df['val_loss'], label='Val Loss')
    plt.title('Loss Curve'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, df['train_dice'], label='Train Dice')
    plt.plot(epochs, df['val_dice'], label='Val Dice')
    plt.title('Dice Curve'); plt.legend(); plt.grid(True)

    plt.savefig(save_path)
    plt.close()

# --- 7. 학습 루프 ---
print(f"🚀 학습 시작 | 장치: {DEVICE} | 폴더명: {CURRENT_TIME}")

history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum, train_dice_sum = 0, 0
    
    for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss_sum += loss.item()
        train_dice_sum += calculate_dice(outputs, masks)

    # 평균 계산
    avg_train_loss = train_loss_sum / len(train_loader)
    avg_train_dice = train_dice_sum / len(train_loader)

    # 검증
    model.eval()
    val_loss_sum, val_dice_sum = 0, 0
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE).long()
            outputs = model(images)
            val_loss_sum += criterion(outputs, masks).item()
            val_dice_sum += calculate_dice(outputs, masks)
            
    avg_val_loss = val_loss_sum / len(val_loader)
    avg_val_dice = val_dice_sum / len(val_loader)

    # 기록 누적
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_dice"].append(avg_train_dice)
    history["val_dice"].append(avg_val_dice)
    
    print(f"📊 [Ep {epoch}] Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
    
    # Best Model 저장 및 Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        save_sample_results(epoch, images, masks, outputs)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_NAME))
        print(f"⭐ Best Model Saved!")
    else:
        early_stop_counter += 1
        print(f"⚠️ 개선 없음 ({early_stop_counter}/{PATIENCE})")

    if early_stop_counter >= PATIENCE:
        print(f"🛑 조기 종료 (Epoch {epoch})")
        break

# 결과 저장
history_df = pd.DataFrame(history)
history_path = os.path.join(SAVE_DIR, HISTORY_CSV)
history_df.to_csv(history_path, index=False)

# 그래프 생성
plot_learning_curves(history_path, os.path.join(LOG_DIR, CURVE_PLOT))

print(f"🎉 모든 학습 완료! 결과 저장 위치: {SAVE_DIR}")