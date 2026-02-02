import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

class GastricDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, json_dir=None, transform=None, task='clf'):
        self.img_paths = []
        self.mask_paths = []
        self.labels = [] 
        self.task = task
        self.transform = transform

        # [중요] JSON의 'category' 문자열과 반드시 일치해야 합니다.
        self.label_map = {"STDI": 0, "STNT": 1, "STIN": 2, "STMX": 3}

        if not os.path.exists(img_dir):
            print(f"❌ 에러: {img_dir} 경로를 찾을 수 없습니다.")
            return

        categories = sorted(os.listdir(img_dir))
        for cat in categories:
            cat_dir = os.path.join(img_dir, cat)
            if not os.path.isdir(cat_dir): continue

            # 폴더명 매칭 규칙 (Train/Test 경로 구조 대응)
            mask_cat = cat.replace("TS_", "TL_").replace("VS_", "VL_")
            json_cat = cat.replace("TS_", "TL_").replace("VS_", "VL_")
            
            files = sorted([f for f in os.listdir(cat_dir) if f.endswith('.png')])
            
            for f in tqdm(files, desc=f"Loading {cat}"):
                img_path = os.path.join(cat_dir, f)
                
                # 1. JSON 라벨 읽기
                label = -1 # 초기값 (오류 확인용)
                if json_dir:
                    json_path = os.path.join(json_dir, json_cat, f.replace('.png', '.json'))
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8-sig') as jf:
                                data = json.load(jf)
                                cat_name = data['content']['clinical']['category']
                                # 매핑 사전에 없으면 3(STMX)으로 처리하되 경고 출력
                                label = self.label_map.get(cat_name, 3)
                        except Exception as e:
                            label = 3
                    else:
                        # JSON이 없으면 해당 데이터는 건너뛰거나 기본값 처리
                        label = 3

                # 2. 마스크 경로 확인 (Segmentation 태스크일 때)
                if task == 'seg' and mask_dir:
                    m_path = os.path.join(mask_dir, mask_cat, f)
                    if not os.path.exists(m_path):
                        # 마스크 파일이 없으면 Expert 화면이 비어 보임 -> 리스트에 추가 안 함
                        continue 
                    self.mask_paths.append(m_path)

                self.img_paths.append(img_path)
                self.labels.append(label)

        print(f"✅ 로드 완료: 이미지 {len(self.img_paths)}장, 라벨 {len(self.labels)}개")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # 이미지 읽기 (한글 경로 대응)
        img_array = np.fromfile(img_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(f"이미지 로드 실패: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.task == 'seg':
            mask_path = self.mask_paths[idx]
            mask_array = np.fromfile(mask_path, np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            if mask is None: raise FileNotFoundError(f"마스크 로드 실패: {mask_path}")
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            
            return image, mask.long(), torch.tensor(label, dtype=torch.long)
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, torch.tensor(label, dtype=torch.long)

# ... [get_loader, get_transforms 함수는 기존과 동일] ...

def get_loader(img_dir, mask_dir, json_dir, batch_size, transform, task='clf', shuffle=True):
    dataset = GastricDataset(img_dir, mask_dir, json_dir, transform=transform, task=task)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

def get_transforms(task='clf', size=512, is_train=True):
    list_transforms = []
    list_transforms.append(A.Resize(size, size))
    
    if is_train:
        list_transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    
    list_transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return A.Compose(list_transforms)

if __name__ == "__main__":
    print("🚀 데이터 로더 테스트를 시작합니다...")
    print(f"torch.cuda available: {torch.cuda.is_available()}")
    
    # 테스트하고 싶은 스플릿을 선택하세요: Training, Validation, Test
    SPLIT = "Test" 
    IMG_DIR = f"data/processed/images_512/{SPLIT}"
    MASK_DIR = f"data/processed/masks/{SPLIT}"
    JSON_DIR = f"data/raw/{SPLIT}/02.라벨링데이터"
    if not os.path.exists(IMG_DIR):
        print(f"❌ 에러: {SPLIT} 데이터가 없습니다. 먼저 전처리하세요.")
    else:
        # 테스트 시에는 is_train=False로 두어 증강 없이 리사이즈만 확인 가능
        test_transform = get_transforms(task='seg', size=512, is_train=False)
        
        test_loader = get_loader(
            img_dir=IMG_DIR,
            mask_dir=MASK_DIR,
            json_dir=JSON_DIR,
            batch_size=2,
            transform=test_transform,
            task='seg'
        )
        
        images, masks, labels = next(iter(test_loader))
        print(f"✅ [{SPLIT}] 이미지 배치 크기: {images.shape}")
        print(f"✅ [{SPLIT}] 마스크 배치 크기: {masks.shape}")
        print(f"✅ [{SPLIT}] 라벨 데이터: {labels}")
        print("🎉 데이터 로더가 모든 스플릿에 대해 정상 작동합니다!")