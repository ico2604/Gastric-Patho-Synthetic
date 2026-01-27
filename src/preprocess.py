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
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        content = data.get('content', {})
        file_info = content.get('file', {})
        objects = file_info.get('object', []) 
        
        label_map = {"Tumor": 1, "Stroma": 2, "Normal": 3, "Immune": 4}
        
        for obj in objects:
            label_name = obj.get('label')
            points = obj.get('coordinate') 
            
            if label_name in label_map and points:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], label_map[label_name])
                
    except Exception as e:
        print(f"\n[Error] JSON 파싱 실패: {json_path} | {e}")
            
    return mask

def run_preprocessing(mode="all", target_size=(512, 512)):
    # 1. 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if mode == "all":
        splits = ["Training", "Validation", "Test"]
    else:
        splits = [mode]
        
    for split in splits:
        if split == "Training":
            raw_prefix, label_prefix = "TS", "TL"
        elif split == "Validation":
            raw_prefix, label_prefix = "VS", "VL"
        else: # Test
            raw_prefix, label_prefix = "TS", "TL"
        
        # 원본 데이터 경로
        raw_base = os.path.join(base_dir, "data", "raw", split, "01.원천데이터")
        label_base = os.path.join(base_dir, "data", "raw", split, "02.라벨링데이터")
        
        # [수정 부분] 저장 경로 설정: split 뒤에 01.원천데이터 / 02.라벨링데이터 추가
        processed_img_base = os.path.join(base_dir, "data", "processed", "images_512", split, "01.원천데이터")
        processed_mask_base = os.path.join(base_dir, "data", "processed", "masks", split, "02.라벨링데이터")

        disease_names = ["미만형선암", "위염", "장형선암", "혼합형선암"]

        for disease in disease_names:
            raw_cat_dir = f"{raw_prefix}_{disease}"
            label_cat_dir = f"{label_prefix}_{disease}"
            
            img_dir = os.path.join(raw_base, raw_cat_dir)
            json_dir = os.path.join(label_base, label_cat_dir)
            
            if not os.path.exists(img_dir):
                continue
                
            img_paths = glob(os.path.join(img_dir, "*.png"))
            
            # 하위 카테고리 폴더(예: TS_미만형선암)까지 포함하여 생성
            save_img_dir = os.path.join(processed_img_base, raw_cat_dir)
            save_mask_dir = os.path.join(processed_mask_base, label_cat_dir)
            os.makedirs(save_img_dir, exist_ok=True)
            os.makedirs(save_mask_dir, exist_ok=True)
            
            print(f"\n🚀 [{split}] {disease} 전처리 시작 ({len(img_paths)}개)")
            
            for img_path in tqdm(img_paths, desc=f"Processing {split}/{disease}"):
                file_name = os.path.basename(img_path)
                json_path = os.path.join(json_dir, file_name.replace(".png", ".json"))
                
                if not os.path.exists(json_path):
                    continue
                
                # 이미지 처리 및 리사이징
                img_array = np.fromfile(img_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None: continue
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # 마스크 처리 및 리사이징
                mask = generate_mask(json_path, shape=(img.shape[0], img.shape[1]))
                mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

                # 이미지 저장 (01.원천데이터 하위)
                img_save_path = os.path.join(save_img_dir, file_name)
                _, encoded_img = cv2.imencode(".png", img_resized)
                with open(img_save_path, mode='w+b') as f:
                    encoded_img.tofile(f)
                
                # 마스크 저장 (02.라벨링데이터 하위)
                mask_save_path = os.path.join(save_mask_dir, file_name)
                _, encoded_mask = cv2.imencode(".png", mask_resized)
                with open(mask_save_path, mode='w+b') as f:
                    encoded_mask.tofile(f)

if __name__ == "__main__":
    run_preprocessing(mode="all")