import warnings
warnings.filterwarnings('ignore')
import os 
import time
import random
import argparse
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from glob import glob
from urllib.request import urlopen
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

#Fix Seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def time_log():
    return datetime.datetime.now().strftime("|%Y-%m-%d|%H:%M:%S.%f|")[:-4]+'|'

class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.anno = pd.read_csv(os.path.join(self.path, 'label.csv'))
        self.anno_dict = self.anno.to_dict('records')
        self.transform = transform
        
        self.classes = {'normal': 0, 'abnormal': 1}
        self.class_dict = dict(zip(self.classes, range(len(self.classes))))
        
        # [핵심 수정] 하위 폴더 어디에 있든 파일명으로 경로를 매핑
        self.img_dir = os.path.join(self.path, '01.원천데이터')
        print(f"🔍 이미지 위치 스캔 중: {self.img_dir}")
        self.all_file_paths = {}
        for root, _, files in os.walk(self.img_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.all_file_paths[f] = os.path.join(root, f)
        print(f"✅ 스캔 완료! 총 {len(self.all_file_paths)}개의 이미지를 찾았습니다.")
    
    def __len__(self):
        return len(self.anno)
        
    def __getitem__(self, idx):
        # CSV에 적힌 이름에서 순수 파일명만 추출
        raw_name = self.anno_dict[idx]['file_name'].replace('\\', '/')
        file_only = os.path.basename(raw_name)
        
        # 미리 스캔해둔 경로에서 파일 찾기
        img_path = self.all_file_paths.get(file_only)
        
        # [수정] 파일이 없으면 에러 대신 다음 인덱스를 불러옴 (Skip 로직)
        if img_path is None or not os.path.exists(img_path):
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)
            
        try:
            image = Image.open(img_path).convert('RGB')
            targets = self.class_dict[self.anno_dict[idx]['tumor_category']]
            
            if self.transform:
                image = self.transform(image)
            return image, targets
        except Exception:
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

def model_train(model, data_loader, loss_fn, optimizer, device, verbose = False):
    model.train()
    running_loss = 0
    corr = 0
    cnt = 0
    
    # verbose가 True일 때만 tqdm 표시
    pbar = tqdm(data_loader) if verbose else data_loader
    
    for img, lbl in pbar:
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, lbl)
        loss.backward()
        optimizer.step()

        _, pred = output.max(dim=1)
        corr += pred.eq(lbl).sum().item()
        running_loss += loss.item() * img.size(0)
        cnt += img.size(0)
        
        if verbose and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(loss=running_loss/cnt, acc=corr/cnt)
            
    acc = corr / cnt if cnt > 0 else 0
    return running_loss / cnt if cnt > 0 else 0, acc

def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        cnt = 0
        gts = []
        preds = []
        for img, lbl in data_loader:
            img, lbl = img.to(device), lbl.to(device)
            output = model(img)
            
            _, pred = output.max(dim=1)
            preds.append(pred.detach().cpu())
            gts.append(lbl.detach().cpu())

            corr += torch.sum(pred.eq(lbl)).item()
            running_loss += loss_fn(output, lbl).item() * img.size(0)
            cnt += img.size(0)
            
        acc = corr / cnt if cnt > 0 else 0
        f1 = f1_score(np.concatenate(gts), np.concatenate(preds), average='macro')
        return running_loss / cnt if cnt > 0 else 0, acc, f1

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',          type=int,   default=10)
    parser.add_argument('--batch_size',     type=int,   default=4)
    parser.add_argument('--LR',     type=float, default=1e-4)
    parser.add_argument('--WD',     type=float, default=0.0001)
    parser.add_argument('--img_size',     type=int, default=256)    
    parser.add_argument('--train_path', type=str, default='/Users/admin/CV_project/Gastric-Patho-Synthetic/data/raw/Training')
    parser.add_argument('--valid_path', type=str, default='/Users/admin/CV_project/Gastric-Patho-Synthetic/data/raw/Validation')
    parser.add_argument('--verbose', type=bool, default=True)
    
    args  = parser.parse_args()
    
    # 장치 설정 (MPS 최우선)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"{time_log()} [INFO] Using device: {device}")

    # ImageNet 데이터셋의 평균과 표준편차 (전이학습 시 필수 표준 수치)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    image_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
    # --- 데이터 증강(Augmentation) 강화 ---
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15), # 15도 내외로 무작위 회전
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # 미세한 밝기/대비 변화
    # -----------------------------------
        transforms.ToTensor(), 
        transforms.Normalize(mean=norm_mean, std=norm_std) # 모델이 보기 편하게 정규화
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=norm_mean, std=norm_std) # 검증/테스트 시에도 똑같이 정규화!
    ])

    print(f"{time_log()} [INFO] Load Dataset")
    train_data = CustomDataset(args.train_path, transform=image_transform)
    valid_data = CustomDataset(args.valid_path, transform=validation_transform)

    # 2,000개 데이터 제한 (Subset 활용)
    train_limit = min(len(train_data), 2000)
    train_subset = torch.utils.data.Subset(train_data, range(train_limit))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"{time_log()} [INFO] Load Model")
    model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=len(train_data.classes), bias=True)
    model.to(device) # 위에서 설정한 장치로 전송
    
    optimizer = optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.WD)
    loss_fn = nn.CrossEntropyLoss()
    min_loss = np.inf
    
    print(f"{time_log()} [INFO] Start Training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device, args.verbose)
        val_loss, val_acc, val_f1 = model_evaluate(model, valid_loader, loss_fn, device) 

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), f'best_model.pth')

        print(f'{time_log()} [INFO] Epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}, val_F1: {val_f1:.5f}')
    
    time_spent = time.time() - start_time
    print(f"{time_log()} [INFO] End Training  ---- {int(time_spent//60)}분 {time_spent%60:.2f}초 소요 ----")

# ==========================================
    # [최종] 화면에 그래프와 분석 결과 띄우기 (들여쓰기 수정됨)
    # ==========================================
    print(f"\n{time_log()} [INFO] 분석 결과 창을 화면에 띄웁니다...")
        
    # 최적의 모델 가중치 로드
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    sample_imgs, sample_preds, sample_lbls = [], [], []

    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(valid_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
                
            if i == 0: # 첫 번째 배치 샘플 저장
                sample_imgs, sample_preds, sample_lbls = imgs.cpu(), preds.cpu(), lbls.cpu()

    # --- 첫 번째 화면: Confusion Matrix (혼동 행렬) ---
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.title('How well did the AI distinguish?')
    plt.xlabel('Predicted (AI)')
    plt.ylabel('Actual (Doctor)')
    plt.tight_layout()
    plt.show(block=True) 

    # --- 두 번째 화면: 실제 이미지 판독 결과 (8장) ---
    plt.figure(figsize=(14, 7))
    for i in range(min(8, len(sample_imgs))):
        plt.subplot(2, 4, i + 1)
        img = sample_imgs[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1) # 이미지 시각화
            
        true_name = 'Abnormal' if sample_lbls[i]==1 else 'Normal'
        pred_name = 'Abnormal' if sample_preds[i]==1 else 'Normal'
        color = 'blue' if sample_lbls[i] == sample_preds[i] else 'red'
            
        plt.imshow(img)
        plt.title(f"Target: {true_name}\nAI: {pred_name}", color=color, fontsize=12)
        plt.axis('off')
        
    plt.suptitle("AI Medical Diagnosis Test Samples", fontsize=16)
    plt.tight_layout()
    plt.show(block=True) 

    print("\n📊 [ 최종 지표 리포트 ]")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Abnormal']))