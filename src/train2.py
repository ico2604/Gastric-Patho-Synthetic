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
from sklearn.metrics import f1_score

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms

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
    # MPS는 현재 deterministic 설정이 제한적일 수 있으나 기본 시드는 고정합니다.

seed_everything(0)

def time_log():
    return datetime.datetime.now().strftime("|%Y-%m-%d|%H:%M:%S.%f|")[:-4]+'|'

class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        # 경로 구분자 호환성을 위해 os.path.join 사용
        label_path = os.path.join(self.path, 'label.csv')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"label.csv를 찾을 수 없습니다: {label_path}")
            
        self.anno = pd.read_csv(label_path)
        self.anno_dict = self.anno.to_dict('records')
        self.transform = transform
        
        self.classes = ['STDI', 'STNT', 'STIN', 'STMX']
        self.class_dict = {cat: i for i, cat in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.anno)
        
    def __getitem__(self, idx):
        # 윈도우 경로(\\)를 맥/리눅스 경로(/)에 맞게 처리
        file_path = self.anno_dict[idx]['file_name'].replace('\\', '/')
        img_full_path = os.path.join(self.path, '01.원천데이터', file_path)
        
        image = Image.open(img_full_path).convert('RGB')
        
        cat_name = self.anno_dict[idx]['category'] 
        target = self.class_dict[cat_name]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)
    
def model_train(model, data_loader, loss_fn, optimizer, device, verbose=False):
    model.train()
    running_loss = 0
    corr = 0
    cnt = 0
    
    pbar = tqdm(data_loader, disable=not verbose)
    
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
        
        if verbose:
            pbar.set_postfix(loss=running_loss/cnt, acc=corr/cnt)
            
    acc = corr / len(data_loader.dataset)
    return running_loss / len(data_loader.dataset), acc

def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        gts = []
        preds = []
        for img, lbl in data_loader:
            img, lbl = img.to(device), lbl.to(device)
            output = model(img)
            
            _, pred = output.max(dim=1)
            
            # MPS/GPU 데이터를 CPU로 옮겨서 metric 계산
            gts.append(lbl.detach().cpu())
            preds.append(pred.detach().cpu())

            corr += torch.sum(pred.eq(lbl)).item()
            running_loss += loss_fn(output, lbl).item() * img.size(0)
            
        acc = corr / len(data_loader.dataset)
        f1 = f1_score(torch.cat(gts), torch.cat(preds), average='macro')
        return running_loss / len(data_loader.dataset), acc, f1

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--LR',           type=float, default=4e-06)
    parser.add_argument('--WD',           type=float, default=1e-4)
    parser.add_argument('--img_size',     type=int,   default=256)    
    # 맥 환경에 맞는 경로로 수정 필요 (상대 경로 확인)
    parser.add_argument('--train_path',   type=str,   default='../data/processed/images_512/Training')
    parser.add_argument('--valid_path',   type=str,   default='../data/processed/images_512/Validation')
    parser.add_argument('--verbose',      type=bool,  default=True)
    
    args = parser.parse_args()
    
    # Device 설정 (MPS 우선)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"{time_log()} [INFO] Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"{time_log()} [INFO] Using CUDA")
    else:
        device = torch.device("cpu")
        print(f"{time_log()} [INFO] Using CPU")

    save_weight_dir = './weights'
    os.makedirs(save_weight_dir, exist_ok=True)

    # Transform 설정
    image_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)), # 튜플 형태로 입력 권장
        transforms.RandomHorizontalFlip(0.5), 
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(), 
        # Normalize를 추가하면 학습 성능이 더 좋아질 수 있습니다.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"{time_log()} [INFO] Load Dataset")
    train_data = CustomDataset(args.train_path, transform=image_transform)
    valid_data = CustomDataset(args.valid_path, transform=validation_transform)

    # num_workers=4 설정
    num_workers = 4 

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size,
                              shuffle=True, 
                              num_workers=num_workers,
                              pin_memory=True if torch.cuda.is_available() else False)

    valid_loader = DataLoader(valid_data, 
                             batch_size=args.batch_size,
                             shuffle=False, 
                             num_workers=num_workers,
                             pin_memory=True if torch.cuda.is_available() else False)
    
    print(f"{time_log()} [INFO] Load Model (EfficientNet V2 S)")
    model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=len(train_data.classes), bias=True)
    model.to(device)
    
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
            torch.save(model.state_dict(), os.path.join(save_weight_dir, 'best_model.pth'))
            print(f"      >> Model Saved (Best Loss: {min_loss:.5f})")

        print(f'{time_log()} [INFO] Epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}, val_F1: {val_f1:.5f}')

    time_spent = time.time() - start_time
    print(f"{time_log()} [INFO] End Training ---- {int(time_spent//60)}m {time_spent%60:.2f}s spent ----")