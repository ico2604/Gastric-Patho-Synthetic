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

        # 카테고리별 순회
        categories = sorted(os.listdir(img_dir))
        for cat in categories:
            cat_dir = os.path.join(img_dir, cat)
            if not os.path.isdir(cat_dir): continue
            
            # 정렬을 해서 이미지와 마스크의 순서가 꼬이지 않게 보장합니다.
            files = sorted([f for f in os.listdir(cat_dir) if f.endswith('.png')])
            
            for f in files:
                self.img_paths.append(os.path.join(cat_dir, f))
                if task == 'seg' and mask_dir:
                    self.mask_paths.append(os.path.join(mask_dir, cat, f))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        if image is None: # 파일 손상 대비
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {self.img_paths[idx]}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.task == 'seg':
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"마스크를 읽을 수 없습니다: {self.mask_paths[idx]}")
            
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
        # 대소문자 및 폴더명 변동에 강한 구조
        path_upper = path.upper()
        if "위염" in path or "STNT" in path_upper: return 0
        if "장형" in path or "STIN" in path_upper: return 1
        if "미만" in path or "STDI" in path_upper: return 2
        return 3 # STMX

def get_transforms(task='clf', size=512, is_train=True):
    """
    [심화] is_train 인자를 추가해 학습 시에는 '데이터 증강'을 넣고
    검증/테스트 시에는 '크기 조절'만 하도록 설정합니다.
    """
    list_transforms = []
    
    # 1. 크기 조절 (공통)
    list_transforms.append(A.Resize(size, size))
    
    # 2. 데이터 증강 (학습 때만 적용)
    if is_train:
        list_transforms.extend([
            A.HorizontalFlip(p=0.5), # 좌우 반전
            A.VerticalFlip(p=0.5),   # 상하 반전 (병리 이미지는 방향이 무관하므로 필수)
            A.RandomRotate90(p=0.5), # 90도 회전
        ])
    
    # 3. 정규화 및 텐서 변환 (공통)
    list_transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return A.Compose(list_transforms)

def get_loader(img_dir, mask_dir, batch_size, transform, task='clf', shuffle=True):
    # [수정] transform을 외부에서 주입받도록 변경
    dataset = GastricDataset(img_dir, mask_dir, transform=transform, task=task)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=4, pin_memory=True)

if __name__ == "__main__":
    # 이 블록은 'python data_loader.py'라고 직접 실행할 때만 작동합니다.
    print("🚀 데이터 로더 테스트를 시작합니다...")
    
    # 1. 전처리된 데이터가 있는지 확인용 경로 (실제 경로에 맞춰 수정)
    IMG_DIR = "data/processed/images_512/Training"
    MASK_DIR = "data/processed/masks/Training"
    
    if not os.path.exists(IMG_DIR):
        print("❌ 에러: 전처리된 데이터가 없습니다. preprocess.py를 먼저 실행하세요.")
    else:
        # 2. 트랜스폼 생성
        test_transform = get_transforms(task='seg', size=512, is_train=True)
        
        # 3. 로더 생성 (배치 사이즈 2로 테스트)
        test_loader = get_loader(
            img_dir=IMG_DIR,
            mask_dir=MASK_DIR,
            batch_size=2,
            transform=test_transform,
            task='seg'
        )
        
        # 4. 첫 번째 배치만 가져와보기
        images, masks = next(iter(test_loader))
        print(f"✅ 이미지 배치 크기: {images.shape}") # [2, 3, 512, 512] 예상
        print(f"✅ 마스크 배치 크기: {masks.shape}") # [2, 512, 512] 예상
        print("🎉 데이터 로더가 정상적으로 작동합니다!")