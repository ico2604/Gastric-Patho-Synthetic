import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib.colors import ListedColormap
from matplotlib.patheffects import withStroke

# 설정 로드
from model import GastricMTLModel
from loss import CombinedMTLLoss

import segmentation_models_pytorch as smp

# --- 1. 전역 설정 및 변수화 (Config) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from data_loader import get_loader, get_transforms

# 하이퍼파라미터 및 모델 정보 변수화
CONFIG = {
    "num_seg_classes": 5,
    "num_cls_classes": 4,
    "batch_size": 8,
    "epochs": 10,
    "lr": 1e-4,
    "lambda_seg": 1.0,
    "lambda_cls": 0.5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 경로 변수화
PATHS = {
    "train_img": os.path.join(BASE_DIR, "data/processed/images_512/Training"),
    "val_img": os.path.join(BASE_DIR, "data/processed/images_512/Validation"),
    "train_mask": os.path.join(BASE_DIR, "data/processed/masks/Training"),
    "val_mask": os.path.join(BASE_DIR, "data/processed/masks/Validation"),
    "train_json": os.path.join(BASE_DIR, "data/raw/Training/02.라벨링데이터"),
    "val_json": os.path.join(BASE_DIR, "data/raw/Validation/02.라벨링데이터"),
    "save_dir": os.path.join(BASE_DIR, "checkpoints/mt"),
    "model_path": os.path.join(BASE_DIR, "checkpoints/mt/mtl_best.pth"),
    "log_base_dir": os.path.join(BASE_DIR, "logs/mt")
}

# 실행 시점별 로그 폴더 생성
CURRENT_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(PATHS["log_base_dir"], CURRENT_TIME)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PATHS["save_dir"], exist_ok=True)


# --- 2. 모델 생성 함수 변수화 ---
def create_model(config, paths):
    """
    설정값(config)과 경로(paths)를 인자로 받아 모델을 생성하고 
    기존 가중치가 있다면 로드합니다.
    """
    print(f"📡 모델 생성 중... (Device: {config['device']})")
    model = GastricMTLModel(
        n_seg_classes=config["num_seg_classes"], 
        n_cls_classes=config["num_cls_classes"]
    ).to(config["device"])
    
    if os.path.exists(paths["model_path"]):
        print(f"🔄 기존 학습된 가중치 로드: {paths['model_path']}")
        # map_location을 설정해야 GPU/CPU 환경 변화에 대응 가능합니다.
        model.load_state_dict(torch.load(paths["model_path"], map_location=config["device"]))
    else:
        print("🆕 새로운 모델로 학습을 시작합니다.")
        
    return model


# --- 3. 데이터 로더 및 구성 요소 초기화 ---
train_loader = get_loader(
    PATHS["train_img"], PATHS["train_mask"], PATHS["train_json"], 
    CONFIG["batch_size"], get_transforms(is_train=True), task='seg'
)
val_loader = get_loader(
    PATHS["val_img"], PATHS["val_mask"], PATHS["val_json"], 
    CONFIG["batch_size"], get_transforms(is_train=False), task='seg'
)

model = create_model(CONFIG, PATHS)
criterion = CombinedMTLLoss(l_seg=CONFIG["lambda_seg"], l_cls=CONFIG["lambda_cls"])
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])


# --- 4. 시각화 함수 ---
def save_mt_report(epoch, images, masks, labels, seg_preds, cls_preds, save_path):
    class_info = {1:{"name":"Tumor","color":"#FF0000"}, 2:{"name":"Stroma","color":"#0000FF"}, 
                  3:{"name":"Normal","color":"#00FF00"}, 4:{"name":"Immune","color":"#FF00FF"}}
    cls_names = {0: "Gastritis", 1: "Tubular", 2: "Mixed", 3: "Diffuse"}

    img = images[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    gt_mask = masks[0].cpu().numpy()
    pred_mask = torch.argmax(seg_preds[0], dim=0).detach().cpu().numpy()
    
    true_label = cls_names[labels[0].item()]
    pred_label = cls_names[torch.argmax(cls_preds[0]).item()]

    fig = plt.figure(figsize=(24, 12), facecolor='white')
    
    # [1행] 원본 / GT / AI 예측 결과 비교
    titles = ["Original H&E", "GT Annotation", f"AI Pred (Cls: {pred_label})"]
    for i in range(3):
        ax = plt.subplot(2, 4, i+1)
        ax.imshow(img)
        target = gt_mask if i==1 else pred_mask if i==2 else None
        if target is not None:
            for c_idx, info in class_info.items():
                m = (target == c_idx)
                if np.any(m):
                    colored = np.zeros((*target.shape, 4))
                    colored[m] = [*plt.cm.colors.to_rgb(info['color']), 0.5]
                    ax.imshow(colored)
        t = ax.set_title(titles[i], fontsize=15, fontweight='bold')
        t.set_path_effects([withStroke(linewidth=3, foreground='white')])
        ax.axis('off')

    # 분류 결과 요약 박스
    ax_txt = plt.subplot(2, 4, 4)
    ax_txt.axis('off')
    box_col = 'green' if true_label == pred_label else 'red'
    ax_txt.text(0.5, 0.5, f"REAL: {true_label}\nPRED: {pred_label}", fontsize=18, 
                weight='bold', color=box_col, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor=box_col, boxstyle='round,pad=1'))

    # [2행] 클래스별 개별 결과 (Focus)
    for i, (c_idx, info) in enumerate(class_info.items()):
        ax = plt.subplot(2, 4, 5+i)
        ax.imshow(img)
        m = (pred_mask == c_idx)
        if np.any(m):
            colored = np.zeros((*pred_mask.shape, 4))
            colored[m] = [*plt.cm.colors.to_rgb(info['color']), 0.7]
            ax.imshow(colored)
        t = ax.set_title(f"Focus: {info['name']}", fontsize=13, color=info['color'], weight='bold')
        t.set_path_effects([withStroke(linewidth=3, foreground='white')])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"epoch_{epoch}_mtl.png"))
    plt.close()

# --- 지표 계산 함수 (지표를 알아야 점수를 매깁니다) ---
def get_metrics(seg_pred, seg_gt, cls_pred, cls_gt):
    # 1. Segmentation Dice Score (smp 활용)
    tp, fp, fn, tn = smp.metrics.get_stats(
        torch.argmax(seg_pred, dim=1).unsqueeze(1), 
        seg_gt.unsqueeze(1).long(), 
        mode='multiclass', num_classes=CONFIG["num_seg_classes"]
    )
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    
    # 2. Classification Accuracy
    cls_acc = (torch.argmax(cls_pred, dim=1) == cls_gt).float().mean()
    
    return dice, cls_acc

# --- 학습 루프 준비 ---

best_val_loss = float('inf')  # 가장 낮은 Loss 저장용
history = [] # 결과 기록용

print(f"🚀 학습 시작 (Total Epochs: {CONFIG['epochs']})")

for epoch in range(1, CONFIG["epochs"] + 1):
    # --- [TRAIN] ---
    model.train()
    train_loss, train_dice, train_acc = 0, 0, 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [Train]")
    for imgs, masks, labels in pbar:
        imgs, masks, labels = imgs.to(CONFIG["device"]), masks.to(CONFIG["device"]), labels.to(CONFIG["device"])
        
        optimizer.zero_grad()
        seg_out, cls_out = model(imgs)
        loss, _, _ = criterion(seg_out, masks, cls_out, labels)
        
        loss.backward()
        optimizer.step()
        
        # 점수 계산
        d_score, a_score = get_metrics(seg_out, masks, cls_out, labels)
        train_loss += loss.item()
        train_dice += d_score
        train_acc += a_score
        
        pbar.set_postfix(Loss=f"{loss.item():.4f}", Dice=f"{d_score:.4f}")

    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    # --- [VALIDATION] ---
    model.eval()
    val_loss, val_dice, val_acc = 0, 0, 0
    
    with torch.no_grad():
        for imgs, masks, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            imgs, masks, labels = imgs.to(CONFIG["device"]), masks.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            seg_out, cls_out = model(imgs)
            loss, _, _ = criterion(seg_out, masks, cls_out, labels)
            
            d_score, a_score = get_metrics(seg_out, masks, cls_out, labels)
            val_loss += loss.item()
            val_dice += d_score
            val_acc += a_score

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    # --- [결과 리포트 및 저장] ---
    print(f"\n📊 Epoch {epoch} Result:")
    print(f"   [Train] Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | Acc: {avg_train_acc:.4f}")
    print(f"   [Val]   Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | Acc: {avg_val_acc:.4f}")

    # 1. 시각화 결과 저장 (매 에폭)
    save_mt_report(epoch, imgs, masks, labels, seg_out, cls_out, LOG_DIR)
    
    # 2. 베스트 모델 저장 (Val Loss 기준 개선 시)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), PATHS["model_path"])
        print(f"⭐ Best Model Saved! (Loss: {best_val_loss:.4f})")
    
    print("-" * 50)

print("🎉 모든 학습 과정이 완료되었습니다.")