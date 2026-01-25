import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

def generate_mask(json_path, shape=(1024, 1024)):
    """JSON의 Polygon 좌표를 Binary Mask로 변환"""
    mask = np.zeros(shape, dtype=np.uint8)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', [])
        label_map = {"Tumor": 1, "Stroma": 2, "Normal": 3, "Immune": 4}
        
        for obj in annotations:
            label_name = obj.get('label')
            points = obj.get('points') 
            
            if label_name in label_map and points:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], label_map[label_name])
    except Exception as e:
        print(f"\n[Error] JSON 파싱 실패: {json_path} | {e}")
            
    return mask

def run_preprocessing(target_size=(512, 512)):
    # 1. 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 처리할 데이터 스플릿 (학습용, 검증용)
    splits = ["Training", "Validation"]
    
    for split in splits:
        raw_base = os.path.join(base_dir, f"data/raw/{split}/01.원천데이터")
        label_base = os.path.join(base_dir, f"data/raw/{split}/02.라벨링데이터")
        
        # 저장될 위치: data/processed/images_512/Training/...
        processed_img_base = os.path.join(base_dir, f"data/processed/images_512/{split}")
        processed_mask_base = os.path.join(base_dir, f"data/processed/masks/{split}")

        # Training 폴더면 TL로 시작하는 폴더들을 찾고, Validation이면 VL로 시작하는 폴더를 찾음
        prefix = "TL" if split == "Training" else "VL"
        categories = [f"{prefix}_미만형선암", f"{prefix}_위염", f"{prefix}_장형선암", f"{prefix}_혼합형선암"]

        for cat in categories:
            img_dir = os.path.join(raw_base, cat)
            if not os.path.exists(img_dir):
                print(f"Skipping {img_dir} (폴더 없음)")
                continue
                
            img_paths = glob(os.path.join(img_dir, "*.png"))
            
            # 저장 경로 생성
            save_img_dir = os.path.join(processed_img_base, cat)
            save_mask_dir = os.path.join(processed_mask_base, cat)
            os.makedirs(save_img_dir, exist_ok=True)
            os.makedirs(save_mask_dir, exist_ok=True)
            
            print(f"\n🚀 [{split}] {cat} 전처리 시작 ({len(img_paths)}개)")
            
            for img_path in tqdm(img_paths, desc=f"Processing {cat}"):
                file_name = os.path.basename(img_path)
                json_path = os.path.join(label_base, cat, file_name.replace(".png", ".json"))
                
                if not os.path.exists(json_path):
                    continue
                
                # 이미지 처리
                img = cv2.imread(img_path)
                if img is None: continue
                
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(save_img_dir, file_name), img_resized)
                
                # 마스크 처리 (이미지와 동일한 리사이징 크기 적용)
                mask = generate_mask(json_path, shape=(img.shape[0], img.shape[1]))
                mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(save_mask_dir, file_name), mask_resized)

if __name__ == "__main__":
    run_preprocessing()