import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

# 1. 경로 설정 (반드시 본인의 폴더 구조에 맞게 수정하세요)
# 터미널에서 'ls ../data/processed/images_512/Validation/label.csv'가 실행되는지 확인!
TEST_PATH = '../data/processed/images_512/Validation' 
WEIGHT_PATH = './weights/best_model.pth'
IMG_SIZE = 256
BATCH_SIZE = 8 # MPS 메모리 부족 방지를 위해 낮게 설정

class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        label_path = os.path.join(self.path, 'label.csv')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"경로에 label.csv가 없습니다: {label_path}")
            
        self.anno = pd.read_csv(label_path)
        self.transform = transform
        self.classes = ['STDI', 'STNT', 'STIN', 'STMX']
        self.class_dict = {cat: i for i, cat in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.anno)
        
    def __getitem__(self, idx):
        row = self.anno.iloc[idx]
        file_path = row['file_name'].replace('\\', '/')
        img_full_path = os.path.join(self.path, '01.원천데이터', file_path)
        
        image = Image.open(img_full_path).convert('RGB')
        target = self.class_dict[row['category']]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)

def run():
    print("--- 테스트 프로그램을 시작합니다 ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] 사용 중인 디바이스: {device}")

    # 전처리
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 데이터 로딩
    print(f"[*] 데이터 경로 확인: {TEST_PATH}")
    try:
        dataset = CustomDataset(TEST_PATH, transform=test_transform)
        # num_workers=0 필구 (맥북 프리징 방지)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"[*] 데이터 개수: {len(dataset)}개")
    except Exception as e:
        print(f"[!] 에러 발생 (경로를 확인하세요): {e}")
        return

    # 모델 로드
    print("[*] 모델 가중치 로드 중...")
    model = torchvision.models.efficientnet_v2_s()
    model.classifier[1] = torch.nn.Linear(1280, 4)
    
    if os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("[*] 가중치 로드 완료!")
    else:
        print(f"[!] 가중치 파일이 없습니다: {WEIGHT_PATH}")
        return

    # 추론
    all_preds = []
    all_labels = []
    print("[*] 성능 지표 계산 시작...")
    
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="추론 진행 중"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())

    # 결과 출력
    print("\n" + "="*50)
    print("              성능 검증 결과")
    print("="*50)
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1-Score: {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print("-" * 50)
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))
    print("="*50)

if __name__ == "__main__":
    run()