import warnings
warnings.filterwarnings('ignore')
import os 
import time
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score, classification_report

import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torchvision
from torchvision import transforms

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

class_dict = {'STDI': 0, 'STNT': 1, 'STIN': 2, 'STMX': 3}


def time_log():
    return datetime.datetime.now().strftime("|%Y-%m-%d|%H:%M:%S.%f|")[:-4]+'|'


def single_infer(file_path, model, device, img_size=256):
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size)
    ])
    
    x = np.array(Image.open(file_path))
    x = transform(x)
    x = x.unsqueeze(0)
    with torch.no_grad():
        x = x.to(device)
        output = model(x)
        _, pred = output.max(dim=1)
        pred = pred.detach().cpu()
    return int(pred)

def load_model(device, weight_path ='./weights/best_model.pth'):
    model = torchvision.models.efficientnet_v2_s()
    model.classifier[1]=nn.Linear(in_features=1280, out_features=4, bias=True)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model

def visualize_predictions(test_path, df, class_dict, num_samples=5):
    # 정답과 오답 데이터 분리
    correct_df = df[df['prediction'] == df['target']].sample(n=min(num_samples, len(df[df['prediction'] == df['target']])))
    incorrect_df = df[df['prediction'] != df['target']].sample(n=min(num_samples, len(df[df['prediction'] != df['target']])))
    
    # 클래스 인덱스를 이름으로 변환 (0 -> normal)
    inv_class_dict = {v: k for k, v in class_dict.items()}
    
    samples = pd.concat([correct_df, incorrect_df])
    
    plt.figure(figsize=(15, 6))
    for i, row in enumerate(samples.itertuples()):
        plt.subplot(2, num_samples, i + 1)
        
        # 이미지 로드 (경로 주의: 데이터 구조에 맞게 수정 필요)
        img_path = os.path.join(test_path, '01.원천데이터', row.file_name)
        img = Image.open(img_path)
        
        plt.imshow(img)
        color = 'green' if row.prediction == row.target else 'red'
        plt.title(f"True: {inv_class_dict[row.target]}\nPred: {inv_class_dict[row.prediction]}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    vis_path = os.path.join(test_path, 'sample_results.png')
    plt.savefig(vis_path)
    print(f"\n{time_log()} [INFO] Sample visualization saved to {vis_path}")
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size',     type=int, default=256)    
    parser.add_argument('--test_path', type=str, default='../data/processed/images_512/Test')
    
    args  = parser.parse_args()
    img_size = args.img_size
    test_path = args.test_path
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    print(f"{time_log()} [INFO] Start Test ...")
    start_time = time.time()
    
    target_df = pd.read_csv(os.path.join(test_path,'label.csv'))
    
    predict_ = []
    
    print(f"{time_log()} [INFO] Start Prediction ...")
    for i in tqdm(target_df['file_name']):
        # 1. i 내부에 섞여 있을지 모를 역슬래시(\)를 슬래시(/)로 통합 (맥/리눅스 대응)
        # 윈도우에서도 슬래시(/)는 경로 구분자로 잘 작동합니다.
        i_normalized = i.replace('\\', '/')
        
        # 2. os.path.join을 사용하여 경로 결합
        full_path = os.path.join(test_path, '01.원천데이터', i_normalized)
        
        # 3. 모델 추론 실행
        output = single_infer(full_path, model, device)
        
        # 4. 경로 분해 및 저장용 경로 생성
        path_parts = i_normalized.split('/') 
        
        # 파일명만 있거나 경로가 짧을 경우를 대비한 안전장치
        label = path_parts[-2] if len(path_parts) > 1 else ""
        fname = path_parts[-1]
        
        predict_.append(
            {
                # os.path.join은 실행 환경(OS)에 맞춰서 다시 경로를 합쳐줍니다.
                'file_name': os.path.join(label, fname),
                'prediction': output
            }
        )
    time_spent = time.time()-start_time
    print(f"{time_log()} [INFO] End Prediction  ----{ int(time_spent//60)}:{time_spent%60:.2f} spent ----")
    
    pred_df = pd.DataFrame(predict_)
    
    save_path = test_path+'/prediction.csv'
    pred_df.to_csv(test_path+'/prediction.csv', index=None)
    print(f"{time_log()} [INFO] Result Saved ... to {save_path}")

    
    # gt = pd.read_csv(os.path.join(test_path,'label.csv'))
    # gt['category_idx'] = gt['category'].apply(lambda x: class_dict[x])
    
    # pred = pd.read_csv(test_path+'/prediction.csv')
    # cm = confusion_matrix(np.array(gt['category_idx']), np.array(pred['prediction']))
    # print(f"{time_log()} [INFO] Calculate Confusion Matrix : \n{cm}")
    # f1 = f1_score(gt['category_idx'], pred['prediction'], average='macro')
    # print(f"{time_log()} [INFO] Calculate F1 Score : {f1}")
    # print(f"{time_log()} [INFO] Time Spent : {time.time()-start_time} Seconds")

    gt = pd.read_csv(os.path.join(test_path, 'label.csv'))
    gt['category_idx'] = gt['category'].apply(lambda x: class_dict[x])
    pred = pd.read_csv(test_path + '/prediction.csv')

    y_true = gt['category_idx']
    y_pred = pred['prediction']

    # 주요 지표 계산
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\n{time_log()} [RESULT] Performance Metrics:")
    print(f" - Accuracy  : {acc:.4f}")
    print(f" - Recall    : {rec:.4f} (심각한 케이스를 얼마나 잘 찾아냈는가)")
    print(f" - Precision : {prec:.4f} (예측한 결과가 얼마나 정확한가)")
    print(f" - F1 Score  : {f1:.4f} (종합 성적)")

    # 상세 리포트 출력
    print(f"\n{time_log()} [INFO] Detailed Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=list(class_dict.keys())))

    # --- 시각화: Confusion Matrix Heatmap ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_dict.keys(), 
                yticklabels=class_dict.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 결과 이미지 저장
    vis_save_path = os.path.join(test_path, 'confusion_matrix.png')
    plt.savefig(vis_save_path)
    print(f"\n{time_log()} [INFO] Visualization saved to {vis_save_path}")
    plt.show()

    full_results = gt.copy()
    full_results['prediction'] = pred['prediction']
    full_results['target'] = full_results['category_idx']

    visualize_predictions(test_path, full_results, class_dict)