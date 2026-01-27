import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import shutil
import timm
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns # 혼동 행렬 시각화용
import numpy as np

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))
from data_loader import get_loader, get_transforms

# --- 상세 설정 ---
IMG_TRAIN_DIR = os.path.join(BASE_DIR, "data", "processed", "images_512", "Training")
IMG_VAL_DIR = os.path.join(BASE_DIR, "data", "processed", "images_512", "Validation")
IMG_TEST_DIR = os.path.join(BASE_DIR, "data", "processed", "images_512", "Test") # Test 경로 추가
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = os.path.join(SAVE_DIR, "resnet50_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 클래스 이름 정의 (시각화용)
CLASS_NAMES = ['Gastritis', 'STIN', 'STDI', 'STMX']

def create_model():
    print(f"📡 ResNet-50 모델 로드 중... (Device: {DEVICE})")
    model = timm.create_model('resnet50', pretrained=True, num_classes=4)
    if os.path.exists(MODEL_PATH):
        print(f"🔄 기존 학습된 가중치 로드: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE)

def train_and_validate(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 상세 기록을 위한 리스트
    history = []
    best_acc = 0.0

    print(f"🚀 학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(EPOCHS):
        epoch_start_time = datetime.now()
        
        # --- Training Step ---
        model.train()
        train_loss, train_correct = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation Step ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        # 수치 계산
        t_loss = train_loss / len(train_loader)
        t_acc = (train_correct.double() / len(train_loader.dataset)).item()
        v_loss = val_loss / len(val_loader)
        v_acc = (val_correct.double() / len(val_loader.dataset)).item()
        duration = (datetime.now() - epoch_start_time).total_seconds()

        # 상세 기록 저장 (딕셔너리 형태)
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": round(t_loss, 4),
            "train_acc": round(t_acc, 4),
            "val_loss": round(v_loss, 4),
            "val_acc": round(v_acc, 4),
            "duration_sec": round(duration, 2),
            "timestamp": datetime.now().strftime('%H:%M:%S')
        }
        history.append(epoch_log)

        print(f"✅ Epoch {epoch+1} 종료 | Val Acc: {v_acc:.4f} | Time: {duration:.1f}s")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"⭐ Best Model Saved (Acc: {best_acc:.4f})")

    # CSV 파일로 최종 저장
    df = pd.DataFrame(history)
    df.to_csv(LOG_FILE, index=False)
    print(f"\n📑 상세 학습 로그 저장 완료: {LOG_FILE}")
    return df

def plot_history(df=None):
    """CSV 로그를 읽어와서 상세한 분석 그래프 생성"""
    if df is None:
        if not os.path.exists(LOG_FILE):
            # 가장 최근 로그 파일을 찾음
            log_files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith('.csv')]
            if not log_files: return print("❌ 로그 파일이 없습니다.")
            LOG_FILE_LATEST = max(log_files, key=os.path.getctime)
            df = pd.read_csv(LOG_FILE_LATEST)
        else:
            df = pd.read_csv(LOG_FILE)

    plt.figure(figsize=(15, 6))
    
    # 1. Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss', color='blue')
    plt.plot(df['epoch'], df['val_loss'], 'o-', label='Val Loss', color='red')
    plt.fill_between(df['epoch'], df['train_loss'], df['val_loss'], color='gray', alpha=0.1)
    plt.title('Training & Validation Loss', fontsize=12)
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

    # 2. Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_acc'], 'o-', label='Train Acc', color='green')
    plt.plot(df['epoch'], df['val_acc'], 'o-', label='Val Acc', color='orange')
    plt.axhline(y=max(df['val_acc']), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(df["val_acc"]):.4f}')
    plt.title('Training & Validation Accuracy', fontsize=12)
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()
    print(f"📈 상세 분석 그래프 저장 완료: {PLOT_FILE}")

# [추가] 최종 테스트 함수
def test_model(model, test_loader):
    print("\n🔍 Test Set 평가 및 이미지 분석을 시작합니다...")
    model.eval()
    all_preds = []
    all_labels = []
    
    # 1. 모든 예측값 수집
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 성능 보고서 출력
    print("\n📊 [Test Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    # 2. 맞춘/틀린 인덱스 추출
    correct_idx = np.where(all_preds == all_labels)[0]
    wrong_idx = np.where(all_preds != all_labels)[0]

    # 3. 분석용 이미지 추출 함수 (에러 방지를 위해 DataLoader에서 직접 추출)
    def save_analysis_images(indices, title, filename):
        if len(indices) == 0:
            print(f"ℹ️ {title} 사례가 없어 건너뜁니다.")
            return
        
        display_num = min(len(indices), 10)
        target_indices = indices[:display_num] # 앞에서부터 최대 10개만 선택
        
        plt.figure(figsize=(20, 10))
        
        # 선택된 인덱스의 이미지만 DataLoader에서 다시 가져오기
        count = 0
        for i, (img, lbl) in enumerate(test_loader.dataset):
            if i in target_indices:
                # tensor를 numpy 이미지로 변환
                img_display = img.permute(1, 2, 0).numpy()
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
                
                plt.subplot(2, 5, count + 1)
                plt.imshow(img_display)
                color = 'blue' if title == "Success" else 'red'
                
                # 예측값 찾기 (전체 리스트에서의 인덱스 i 사용)
                pred_name = CLASS_NAMES[all_preds[i]]
                true_name = CLASS_NAMES[all_labels[i]]
                
                plt.title(f"True: {true_name}\nPred: {pred_name}", color=color)
                plt.axis('off')
                count += 1
                if count >= display_num: break

        plt.suptitle(f"{title} Examples", fontsize=20)
        save_path = os.path.join(LOG_DIR, f"{filename}_{TIMESTAMP}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"📸 {title} 분석 이미지 저장 완료: {save_path}")

    # 성공/실패 사례 시각화 실행
    save_analysis_images(correct_idx, "Success", "analysis_success")
    save_analysis_images(wrong_idx, "Failure", "analysis_failure")

    # 4. 혼동 행렬 저장
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.savefig(os.path.join(LOG_DIR, f"confusion_matrix_{TIMESTAMP}.png"))
    plt.close()

def main():
    model = create_model()
    while True:
        print(f"\n=== 🚀 Gastric Cancer AI Analysis System ===")
        print("1. 모델 학습 및 상세 기록 시작")
        print("2. 학습 결과 시각화 (PNG)")
        print("3. 모델 파일 내보내기 (Export)")
        print("4. 최종 테스트 (Test Set 평가)") # 메뉴 추가
        print("5. 종료")
        choice = input("번호를 입력하세요: ")

        if choice == '1':
            train_loader = get_loader(IMG_TRAIN_DIR, None, BATCH_SIZE, get_transforms('clf', 512, True), 'clf')
            val_loader = get_loader(IMG_VAL_DIR, None, BATCH_SIZE, get_transforms('clf', 512, False), 'clf')
            train_and_validate(model, train_loader, val_loader)
        elif choice == '2':
            plot_history()
        elif choice == '3':
            if os.path.exists(MODEL_PATH):
                name = input("저장할 파일명: ")
                target = os.path.join(BASE_DIR, name + ".pth" if not name.endswith(".pth") else name)
                shutil.copy2(MODEL_PATH, target)
                print(f"✅ Export 완료: {target}")
            else: print("❌ 저장된 모델이 없습니다.")
        elif choice == '4':
            if not os.path.exists(IMG_TEST_DIR):
                print(f"❌ Test 데이터 경로가 없습니다: {IMG_TEST_DIR}")
                continue
            test_loader = get_loader(IMG_TEST_DIR, None, 16, get_transforms('clf', 512, False), 'clf', shuffle=False)
            test_model(model, test_loader)
        elif choice == '5': break

if __name__ == "__main__":
    main()