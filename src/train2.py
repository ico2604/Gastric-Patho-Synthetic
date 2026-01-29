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
        
        # [수정] 4개 카테고리에 맞게 클래스 정의
        self.classes = ['STDI', 'STNT', 'STIN', 'STMX']
        self.class_dict = {cat: i for i, cat in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.anno)
        
    def __getitem__(self, idx):
        # [수정] 파일 경로 로직 확인 (01.원천데이터 하위 폴더 구조 포함)
        img_full_path = os.path.join(self.path, '01.원천데이터', self.anno_dict[idx]['file_name'])
        image = Image.open(img_full_path).convert('RGB') # 에러 방지를 위해 .convert('RGB') 추가 권장
        
        # [수정] CSV의 컬럼명이 'category'인지 'tumor_category'인지 확인 후 매칭
        # 이전 단계에서 'category'로 저장하셨으므로 이를 참조합니다.
        cat_name = self.anno_dict[idx]['category'] 
        target = self.class_dict[cat_name]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)
    
def model_train(model, data_loader, loss_fn, optimizer, device, verbose = False):
    
    model.train()
    
    running_loss = 0
    corr = 0
    if verbose==1:
        progress_bar = tqdm(data_loader)
    
    cnt = 0
    for img, lbl in data_loader:
        
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, lbl)
        loss.backward()
        optimizer.step()

        _, pred = output.max(dim=1)
        corr += pred.eq(lbl).sum().item()
        running_loss += loss.item() * img.size(0)
        cnt+=img.size(0)
        if verbose==1:
            progress_bar.set_postfix(loss=running_loss/cnt, acc = corr/cnt)
            progress_bar.update()
            
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
            gts.append(lbl)
            img, lbl = img.to(device), lbl.to(device)
            output = model(img)
            
            _, pred = output.max(dim=1)
            preds.append(pred.detach().cpu())

            corr += torch.sum(pred.eq(lbl)).item()
            running_loss += loss_fn(output, lbl).item() * img.size(0)
            
        acc = corr / len(data_loader.dataset)
        f1 = f1_score(np.concatenate(gts), np.concatenate(preds), average='macro')
        return running_loss / len(data_loader.dataset), acc, f1

    
save_weight_dir = './weights'
os.makedirs(save_weight_dir, exist_ok=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',          type=int,   default=20)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--LR',     type=float, default=4e-06)
    parser.add_argument('--WD',     type=float, default=1e-4)
    parser.add_argument('--img_size',     type=int, default=256)    
    parser.add_argument('--train_path', type=str, default='..\\data\\processed\\images_512\\Training')
    parser.add_argument('--valid_path', type=str, default='..\\data\\processed\\images_512\\Validation')
    parser.add_argument('--verbose', type=bool, default=True)
    print(f"{time_log()} [INFO] START ....")
    
    args  = parser.parse_args()
    
    epochs = args.epochs
    batch_size=args.batch_size
    LR=args.LR
    WD=args.WD
    img_size=args.img_size
    train_path=args.train_path
    valid_path = args.valid_path
    verbose = args.verbose
    
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Resize(img_size),          
            transforms.RandomHorizontalFlip(0.5), 
            transforms.RandomVerticalFlip(0.5),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Resize(img_size)
        ]
    )
    print(f"{time_log()} [INFO] Load Dataset")
    train_data = CustomDataset(train_path, transform=image_transform)
    valid_data = CustomDataset(valid_path, transform=validation_transform)

    num_workers = 0

    train_loader = DataLoader(train_data, 
                              batch_size=batch_size,
                              shuffle=True, 
                              num_workers=num_workers
                             )
    valid_loader = DataLoader(valid_data, 
                             batch_size=batch_size,
                             shuffle=False, 
                             num_workers=num_workers
                            )
    
    print(f"{time_log()} [INFO] Load Model")
    
    model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=len(train_data.classes), bias=True)
    model.to(device)    
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    min_loss = np.inf

        
# Epoch 별 훈련 및 검증을 수행합니다.
    print(f"{time_log()} [INFO] Start Training")
    start_time = time.time()
    for epoch in range(epochs):
        # Model Training
        # 훈련 손실과 정확도를 반환 받습니다.
        train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device, verbose)

        # 검증 손실과 검증 정확도를 반환 받습니다.
        val_loss, val_acc, val_f1 = model_evaluate(model, valid_loader, loss_fn, device) 

        # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_weight_dir, 'best_model.pth'))

        # Epoch 별 결과를 출력합니다.
        print(f'{time_log()} [INFO] Epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}, val_F1: {val_f1:.5f}')
    time_spent = time.time()-start_time
    print(f"{time_log()} [INFO] End Training  ----{ int(time_spent//60)}:{time_spent%60:.2f} spent ----")
            
