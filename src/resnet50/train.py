import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import shutil
import timm
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
from torch.nn import functional as F

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from data_loader import get_loader, get_transforms

# --- 하이퍼파라미터 및 상세 설정 ---
IMG_TRAIN_DIR = os.path.join(BASE_DIR, "data", "processed", "images_512", "Training")
IMG_VAL_DIR = os.path.join(BASE_DIR, "data", "processed", "images_512", "Validation")
IMG_TEST_DIR = os.path.join(BASE_DIR, "data", "processed", "images_512", "Test")
JSON_TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "Training", "02.라벨링데이터")
JSON_VAL_DIR = os.path.join(BASE_DIR, "data", "raw", "Validation", "02.라벨링데이터")
JSON_TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "Test", "02.라벨링데이터")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = os.path.join(SAVE_DIR, "resnet50_best.pth")
LOG_FILE = os.path.join(LOG_DIR, f"train_log_{TIMESTAMP}.csv")
PLOT_FILE = os.path.join(LOG_DIR, f"result_plot_{TIMESTAMP}.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 10
PATIENCE = 5  # Early Stopping: 5에폭 동안 개선 없으면 종료
CLASS_NAMES = ['Gastritis', 'STIN', 'STDI', 'STMX']

# --- 모델 생성 함수 ---
def create_model():
    print(f"📡 ResNet-50 모델 로드 중... (Device: {DEVICE})")
    model = timm.create_model('resnet50', pretrained=True, num_classes=len(CLASS_NAMES))
    if os.path.exists(MODEL_PATH):
        print(f"🔄 기존 학습된 가중치 로드: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE)

# --- 학습 및 검증 함수 (Early Stopping 포함) ---
def train_and_validate(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = []
    best_val_loss = float('inf')
    early_stop_counter = 0

    print(f"🚀 학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(EPOCHS):
        epoch_start_time = datetime.now()
        
        # Training Step
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # Validation Step
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        t_loss, v_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        t_acc, v_acc = train_correct/len(train_loader.dataset), val_correct/len(val_loader.dataset)
        duration = (datetime.now() - epoch_start_time).total_seconds()

        log_entry = {
            "epoch": epoch + 1, "train_loss": t_loss, "train_acc": t_acc,
            "val_loss": v_loss, "val_acc": v_acc, "duration": duration
        }
        history.append(log_entry)

        print(f"✅ Epoch {epoch+1}: Val Loss={v_loss:.4f}, Val Acc={v_acc:.4f} ({duration:.1f}s)")

        # Early Stopping & Model Save
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"⭐ Best Model Saved (Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"🛑 조기 종료 (Early Stopping) 적용됨!")
                break

    pd.DataFrame(history).to_csv(LOG_FILE, index=False)
    print(f"📑 로그 저장 완료: {LOG_FILE}")
    plot_history(pd.DataFrame(history))

# --- 학습 결과 시각화 ---
def plot_history(df=None):
    if df is None:
        if not os.path.exists(LOG_FILE):
            log_files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith('.csv')]
            if not log_files: return print("❌ 로그가 없습니다.")
            df = pd.read_csv(max(log_files, key=os.path.getctime))
        else:
            df = pd.read_csv(LOG_FILE)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Loss Trend'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
    plt.title('Accuracy Trend'); plt.legend()
    
    plt.savefig(PLOT_FILE); plt.close()
    print(f"📈 그래프 저장 완료: {PLOT_FILE}")

# --- 최종 테스트 및 사례 분석 ---
def test_model(model, test_loader):
    print("\n🔍 Test Set 평가 시작...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    print("\n📊 [Test Report]")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
    plt.savefig(os.path.join(LOG_DIR, f"cm_{TIMESTAMP}.png")); plt.close()

# --- Grad-CAM (XAI) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients, self.activations = None, None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))

    def generate(self, input_image, class_idx):
        self.model.zero_grad()
        output = self.model(input_image)
        output[0, class_idx].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        return np.maximum(cam, 0), F.softmax(output, dim=1)[0, class_idx].item()

def run_explainable_ai(model, test_loader):
    print("\n" + "="*50)
    print("🔍 [Interactive XAI Analysis] 특정 클래스 심층 분석")
    print("="*50)
    
    # 1. 사용자 입력 받기
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i}: {name}")
    
    try:
        target_class_idx = int(input(f"\n분석할 클래스 번호를 선택하세요 (0~{len(CLASS_NAMES)-1}): "))
        num_samples = int(input("각 사례별(성공/실패) 분석할 샘플 개수를 입력하세요: "))
        if num_samples <= 0: return print("❗ 개수는 1개 이상이어야 합니다.")
    except ValueError:
        return print("❗ 올바른 숫자를 입력해주세요.")

    model.eval()
    cam_gen = GradCAM(model, model.layer4[-1])
    
    success_list = []
    failure_list = []

    # 2. 데이터 탐색 (선택한 클래스에 해당하는 데이터만 수집)
    print(f"\n📡 {CLASS_NAMES[target_class_idx]} 클래스의 샘플을 수집 중...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)

            for i in range(len(images)):
                # 우리가 선택한 클래스가 실제 정답(Ground Truth)인 데이터만 추출
                if labels[i].item() == target_class_idx:
                    is_correct = (preds[i] == labels[i])
                    sample_data = {
                        'img': images[i:i+1],
                        'label': labels[i].item(),
                        'pred': preds[i].item(),
                        'prob': probs[i][preds[i]].item()
                    }

                    if is_correct and len(success_list) < num_samples:
                        success_list.append(sample_data)
                    elif not is_correct and len(failure_list) < num_samples:
                        failure_list.append(sample_data)

            # 필요한 개수를 모두 채우면 중단
            if len(success_list) >= num_samples and len(failure_list) >= num_samples:
                break

    # 3. 결과 검증: 샘플이 부족한 경우 사용자에게 알림
    if not success_list and not failure_list:
        return print(f"❗ 해당 클래스({CLASS_NAMES[target_class_idx]})에 대한 데이터를 찾을 수 없습니다.")
    
    print(f"✅ 수집 완료: 성공 {len(success_list)}건, 실패 {len(failure_list)}건")

    # 4. 시각화 (동적으로 서브플롯 생성)
    total_samples = len(success_list) + len(failure_list)
    plt.figure(figsize=(12, 5 * total_samples))

    current_plot = 1
    # 성공/실패 데이터를 하나의 리스트로 합침
    all_targets = [("Success", success_list), ("Failure", failure_list)]

    for title_prefix, data_list in all_targets:
        for data in data_list:
            # Grad-CAM 생성
            cam, _ = cam_gen.generate(data['img'], data['pred'])
            
            # 원본 이미지 복원
            orig_img = data['img'][0].permute(1, 2, 0).cpu().numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())

            # [왼쪽] 원본 이미지 + 정보
            plt.subplot(total_samples, 2, current_plot)
            plt.imshow(orig_img)
            color = 'blue' if title_prefix == "Success" else 'red'
            plt.title(f"[{title_prefix}]\nTrue: {CLASS_NAMES[data['label']]}\nPred: {CLASS_NAMES[data['pred']]} ({data['prob']:.2f})", color=color)
            plt.axis('off')

            # [오른쪽] Grad-CAM Overlay
            plt.subplot(total_samples, 2, current_plot + 1)
            plt.imshow(orig_img)
            plt.imshow(cam, cmap='jet', alpha=0.4)
            plt.title(f"Focus: {CLASS_NAMES[data['pred']]}")
            plt.axis('off')
            
            current_plot += 2

    plt.tight_layout()
    save_filename = f"XAI_{CLASS_NAMES[target_class_idx]}_n{num_samples}_{TIMESTAMP}.png"
    save_path = os.path.join(LOG_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 상세 분석 리포트 저장 완료: {save_path}")
# --- 메인 함수 ---
def main():
    model = create_model()
    while True:
        print(f"\n=== 🏥 Healthcare AI Research System (V2) ===")
        print("1. [Train] 학습 및 조기종료 적용")
        print("2. [Verify] 학습 결과 시각화")
        print("3. [Test] 최종 성능 평가 & 사례 분석")
        print("4. [XAI] Grad-CAM 판단 근거 확인")
        print("5. [Export] 모델 내보내기")
        print("0. 종료")
        choice = input("선택: ")

        if choice == '1':
            train_loader = get_loader(IMG_TRAIN_DIR, None, JSON_TRAIN_DIR, BATCH_SIZE, get_transforms('clf', 512, True), 'clf')
            val_loader = get_loader(IMG_VAL_DIR, None, JSON_VAL_DIR, BATCH_SIZE, get_transforms('clf', 512, False), 'clf')
            train_and_validate(model, train_loader, val_loader)
        elif choice == '2':
            plot_history()
        elif choice == '3':
            test_loader = get_loader(IMG_TEST_DIR, None, JSON_TEST_DIR, BATCH_SIZE, get_transforms('clf', 512, False), 'clf', shuffle=False)
            test_model(model, test_loader)
        elif choice == '4':
            test_loader = get_loader(IMG_TEST_DIR, None, JSON_TEST_DIR, 1, get_transforms('clf', 512, False), 'clf')
            run_explainable_ai(model, test_loader)
        elif choice == '5':
            name = input("파일명: ")
            shutil.copy2(MODEL_PATH, os.path.join(BASE_DIR, name + ".pth"))
        elif choice == '0': break

if __name__ == "__main__":
    main()