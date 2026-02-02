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
from sklearn.metrics import f1_score

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

# Fix Seeds
def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

seed_everything(0)

def time_log():
    return datetime.datetime.now().strftime("|%Y-%m-%d|%H:%M:%S.%f|")[:-4]+'|'

class SegmentationDataset(Dataset):
    def __init__(self, path, transform=None, img_size=256):
        self.path = path
        label_path = os.path.join(self.path, 'label.csv')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"label.csv를 찾을 수 없습니다: {label_path}")
            
        self.anno = pd.read_csv(label_path)
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.anno)
        
    def __getitem__(self, idx):
        # 1. 원천 데이터 (이미지) 로드
        img_name = self.anno.iloc[idx]['file_name'].replace('\\', '/')
        img_path = os.path.join(self.path, '01.원천데이터', img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 2. 라벨 데이터 (마스크 이미지) 로드 
        # 'mask_name'은 CSV 내 마스크 파일 경로 컬럼명으로 수정 필요
        mask_name = self.anno.iloc[idx]['mask_name'].replace('\\', '/') 
        mask_path = os.path.join(self.path, '02.라벨링데이터', mask_name)
        mask = Image.open(mask_path).convert('L') # 흑백
        
        # Segmentation은 이미지와 마스크의 크기가 항상 동일해야 함
        image = image.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        
        if self.transform:
            image = self.transform(image)
        
        # 마스크는 0~1 사이의 텐서로 변환 (암 부위 1, 배경 0)
        mask = torch.as_tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)
        mask = mask / 255.0 if mask.max() > 1 else mask
        
        return image, mask

def model_train(model, data_loader, loss_fn, optimizer, device, verbose=False):
    model.train()
    running_loss = 0
    pbar = tqdm(data_loader, disable=not verbose)
    
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)
        
        optimizer.zero_grad()
        # DeepLabV3의 출력은 딕셔너리 형태 {'out': tensor}
        output = model(img)['out']
        loss = loss_fn(output, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)
        if verbose:
            pbar.set_postfix(loss=running_loss/((pbar.n+1)*img.size(0)))
            
    return running_loss / len(data_loader.dataset)

def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for img, mask in data_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)['out']
            loss = loss_fn(output, mask)
            running_loss += loss.item() * img.size(0)
            
    return running_loss / len(data_loader.dataset)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--LR',           type=float, default=1e-4)
    parser.add_argument('--img_size',     type=int,   default=256)    
    parser.add_argument('--train_path',   type=str,   default='../data/Training')
    parser.add_argument('--valid_path',   type=str,   default='../data/Validation')
    parser.add_argument('--verbose',      type=bool,  default=True)
    args = parser.parse_args()
    
    # Device 설정
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"{time_log()} [INFO] Using {device}")

    # Transform 설정 (마스크와 정렬을 위해 Resize는 Dataset 내부에서 수행)
    image_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = SegmentationDataset(args.train_path, transform=image_transform, img_size=args.img_size)
    valid_data = SegmentationDataset(args.valid_path, transform=image_transform, img_size=args.img_size)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 모델 로드 (DeepLabV3 사용)
    print(f"{time_log()} [INFO] Load Model (DeepLabV3 ResNet50)")
    model = deeplabv3_resnet50(weights='DEFAULT')
    # 이진 분류를 위해 최종 출력 채널을 1로 변경
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.LR)
    # 세그멘테이션용 손실함수
    loss_fn = nn.BCEWithLogitsLoss()
    min_loss = np.inf

    print(f"{time_log()} [INFO] Start Training")
    for epoch in range(args.epochs):
        train_loss = model_train(model, train_loader, loss_fn, optimizer, device, args.verbose)
        val_loss = model_evaluate(model, valid_loader, loss_fn, device) 

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), './weights/best_segmentation_model.pth')
            print(f"      >> Saved Best Model ({min_loss:.5f})")

        print(f'{time_log()} Epoch {epoch+1:02d}, Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')