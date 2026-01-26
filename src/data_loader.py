import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GastricDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, task='clf'):
        self.img_paths = []
        self.mask_paths = []
        self.task = task
        self.transform = transform

        if not os.path.exists(img_dir):
            print(f"❌ 경고: {img_dir} 경로를 찾을 수 없습니다.")
            return

        categories = sorted(os.listdir(img_dir))
        for cat in categories:
            cat_dir = os.path.join(img_dir, cat)
            if not os.path.isdir(cat_dir): continue
            
            # [핵심 수정] Test 규칙(SS -> SL) 추가 반영
            mask_cat = cat.replace("TS_", "TL_").replace("VS_", "VL_").replace("SS_", "SL_")
            
            files = sorted([f for f in os.listdir(cat_dir) if f.endswith('.png')])
            
            for f in files:
                self.img_paths.append(os.path.join(cat_dir, f))
                if task == 'seg' and mask_dir:
                    self.mask_paths.append(os.path.join(mask_dir, mask_cat, f))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # 1. 이미지 읽기 (한글 경로 대응)
        img_array = np.fromfile(img_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.task == 'seg':
            mask_path = self.mask_paths[idx]
            # 2. 마스크 읽기 (한글 경로 대응)
            mask_array = np.fromfile(mask_path, np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                raise FileNotFoundError(f"마스크를 읽을 수 없습니다: {mask_path}")
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            return image, mask.long()
        else:
            label = self._get_label_from_path(self.img_paths[idx])
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, torch.tensor(label, dtype=torch.long)

    def _get_label_from_path(self, path):
        path_upper = path.upper()
        if "위염" in path or "STNT" in path_upper: return 0
        if "장형" in path or "STIN" in path_upper: return 1
        if "미만" in path or "STDI" in path_upper: return 2
        return 3 # STMX

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

def get_loader(img_dir, mask_dir, batch_size, transform, task='clf', shuffle=True):
    dataset = GastricDataset(img_dir, mask_dir, transform=transform, task=task)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=0, 
                      pin_memory=True)

if __name__ == "__main__":
    print("🚀 데이터 로더 테스트를 시작합니다...")
    print(f"torch.cuda available: {torch.cuda.is_available()}")
    
    # 테스트하고 싶은 스플릿을 선택하세요: Training, Validation, Test
    SPLIT = "Test" 
    IMG_DIR = f"data/processed/images_512/{SPLIT}"
    MASK_DIR = f"data/processed/masks/{SPLIT}"
    
    if not os.path.exists(IMG_DIR):
        print(f"❌ 에러: {SPLIT} 데이터가 없습니다. 먼저 전처리하세요.")
    else:
        # 테스트 시에는 is_train=False로 두어 증강 없이 리사이즈만 확인 가능
        test_transform = get_transforms(task='seg', size=512, is_train=False)
        
        test_loader = get_loader(
            img_dir=IMG_DIR,
            mask_dir=MASK_DIR,
            batch_size=2,
            transform=test_transform,
            task='seg'
        )
        
        images, masks = next(iter(test_loader))
        print(f"✅ [{SPLIT}] 이미지 배치 크기: {images.shape}")
        print(f"✅ [{SPLIT}] 마스크 배치 크기: {masks.shape}")
        print("🎉 데이터 로더가 모든 스플릿에 대해 정상 작동합니다!")