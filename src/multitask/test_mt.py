import torch
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patheffects import withStroke

# 설정 및 모델 로드
from model import GastricMTLModel

# --- 1. 설정 (Config & Paths) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from data_loader import get_loader, get_transforms



CONFIG = {
    "num_seg_classes": 5,
    "num_cls_classes": 4,
    "batch_size": 1, # 테스트는 1장씩 정밀하게 확인
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

PATHS = {
    "test_img": os.path.join(BASE_DIR, "data/processed/images_512/Test"), # 혹은 Test 폴더
    "test_mask": os.path.join(BASE_DIR, "data/processed/masks/Test"),
    "test_json": os.path.join(BASE_DIR, "data/raw/Test/02.라벨링데이터"),
    "model_path": os.path.join(BASE_DIR, "checkpoints/mt/mtl_best.pth"),
    "result_dir": os.path.join(BASE_DIR, "logs/mt/test_report")
}
os.makedirs(PATHS["result_dir"], exist_ok=True)

# --- 2. 시각화 비교 함수 (Expert vs AI) ---
def save_comparison_report(idx, img, gt_mask, pred_mask, gt_label, pred_label):
    class_info = {1:{"name":"Tumor","color":"#FF0000"}, 2:{"name":"Stroma","color":"#0000FF"}, 
                  3:{"name":"Normal","color":"#00FF00"}, 4:{"name":"Immune","color":"#FF00FF"}}
    cls_names = {0: "STDI", 1: "STNT", 2: "STIN", 3: "STMX"}

    # 이미지 정규화 해제
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='black')
    
    # [1] 원본 이미지
    axes[0].imshow(img)
    axes[0].set_title("Original H&E Slide", color='white', fontsize=15)
    
    # [2] Expert Annotation (GT)
    axes[1].imshow(img)
    for c_idx, info in class_info.items():
        m = (gt_mask == c_idx)
        if np.any(m):
            colored = np.zeros((*gt_mask.shape, 4))
            colored[m] = [*plt.cm.colors.to_rgb(info['color']), 0.5]
            axes[1].imshow(colored)
    axes[1].set_title(f"Expert: {cls_names[gt_label]}", color='cyan', fontsize=15, fontweight='bold')

    # [3] AI Prediction
    axes[2].imshow(img)
    for c_idx, info in class_info.items():
        m = (pred_mask == c_idx)
        if np.any(m):
            colored = np.zeros((*pred_mask.shape, 4))
            colored[m] = [*plt.cm.colors.to_rgb(info['color']), 0.5]
            axes[2].imshow(colored)
    
    # 분류 결과가 맞으면 초록색, 틀리면 빨간색 제목
    is_correct = (gt_label == pred_label)
    title_col = 'lime' if is_correct else 'tomato'
    axes[2].set_title(f"AI Pred: {cls_names[pred_label]}", color=title_col, fontsize=15, fontweight='bold')

    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS["result_dir"], f"test_{idx:03d}.png"), facecolor='black')
    plt.close()

# --- 3. 메인 테스트 루프 ---
def test():
    # 모델 로드
    model = GastricMTLModel(CONFIG["num_seg_classes"], CONFIG["num_cls_classes"]).to(CONFIG["device"])
    model.load_state_dict(torch.load(PATHS["model_path"], map_location=CONFIG["device"]))
    model.eval()

    test_loader = get_loader(PATHS["test_img"], PATHS["test_mask"], PATHS["test_json"], 
                             CONFIG["batch_size"], get_transforms(is_train=False), task='seg', shuffle=True)

    total_dice = 0
    total_acc = 0
    
    print("🔍 테스트를 시작합니다...")
    with torch.no_grad():
        for i, (imgs, masks, labels) in enumerate(tqdm(test_loader)):
            imgs, masks, labels = imgs.to(CONFIG["device"]), masks.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            seg_out, cls_out = model(imgs)
            
            # 예측값 가공
            pred_mask = torch.argmax(seg_out[0], dim=0).cpu().numpy()
            pred_label = torch.argmax(cls_out[0]).item()
            gt_mask = masks[0].cpu().numpy()
            gt_label = labels[0].item()

            # 시각화 저장 (샘플 20장만 저장하거나 전체 저장)
            if i < 50: 
                save_comparison_report(i, imgs[0], gt_mask, pred_mask, gt_label, pred_label)

            # 점수 누적 로직 (간단 구현)
            # (학습 때 썼던 get_metrics 함수를 활용하면 더 정확합니다)
            acc = 1 if pred_label == gt_label else 0
            total_acc += acc

    print(f"\n" + "="*30)
    print(f"🏆 TEST RESULT")
    print(f"   Classification Accuracy: {total_acc/len(test_loader)*100:.2f}%")
    print(f"   Report Saved at: {PATHS['result_dir']}")
    print("="*30)

if __name__ == "__main__":
    test()