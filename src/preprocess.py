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
        # utf-8 대신 utf-8-sig로 변경 (BOM 제거 대응)
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # 데이터 구조가 복잡하므로 안전하게 접근
        content = data.get('content', {})
        file_info = content.get('file', {})
        objects = file_info.get('object', []) # 제공해주신 JSON 예시 구조에 맞춤
        
        label_map = {"Tumor": 1, "Stroma": 2, "Normal": 3, "Immune": 4}
        
        for obj in objects:
            label_name = obj.get('label')
            points = obj.get('coordinate') # 제공해주신 예시에서는 points가 아니라 coordinate
            
            if label_name in label_map and points:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], label_map[label_name])
                
    except Exception as e:
        print(f"\n[Error] JSON 파싱 실패: {json_path} | {e}")
            
    return mask

def run_preprocessing(mode="Training", target_size=(512, 512)):
    # 1. 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 처리할 데이터 스플릿
    if mode == "all":
        splits = ["Training", "Validation"]
    else:
        splits = [mode]
        
    for split in splits:
        # 데이터셋 규칙 적용 (원천=S, 라벨=L)
        raw_prefix = "TS" if split == "Training" else "VS"
        label_prefix = "TL" if split == "Training" else "VL"
        
        raw_base = os.path.join(base_dir, "data", "raw", split, "01.원천데이터")
        label_base = os.path.join(base_dir, "data", "raw", split, "02.라벨링데이터")
        
        processed_img_base = os.path.join(base_dir, "data", "processed", "images_512", split)
        processed_mask_base = os.path.join(base_dir, "data", "processed", "masks", split)

        # 질환 카테고리
        disease_names = ["미만형선암", "위염", "장형선암", "혼합형선암"]

        for disease in disease_names:
            raw_cat_dir = f"{raw_prefix}_{disease}"
            label_cat_dir = f"{label_prefix}_{disease}"
            
            img_dir = os.path.join(raw_base, raw_cat_dir)
            json_dir = os.path.join(label_base, label_cat_dir)
            
            if not os.path.exists(img_dir):
                print(f"Skipping {img_dir} (폴더 없음)")
                continue
                
            img_paths = glob(os.path.join(img_dir, "*.png"))
            
            # 저장 경로 생성
            save_img_dir = os.path.join(processed_img_base, raw_cat_dir)
            save_mask_dir = os.path.join(processed_mask_base, label_cat_dir)
            os.makedirs(save_img_dir, exist_ok=True)
            os.makedirs(save_mask_dir, exist_ok=True)
            
            print(f"\n🚀 [{split}] {disease} 전처리 시작 ({len(img_paths)}개)")
            
            for img_path in tqdm(img_paths, desc=f"Processing {disease}"):
                file_name = os.path.basename(img_path)
                json_path = os.path.join(json_dir, file_name.replace(".png", ".json"))
                
                if not os.path.exists(json_path):
                    continue
                
                # --- 1. 이미지 읽기 (한글 경로 대응) ---
                img_array = np.fromfile(img_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None: 
                    continue

                # --- 2. 리사이징 ---
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # --- 3. 마스크 생성 및 리사이징 ---
                mask = generate_mask(json_path, shape=(img.shape[0], img.shape[1]))
                mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

                # --- 4. 이미지 저장 (한글 경로 대응) ---
                img_save_path = os.path.join(save_img_dir, file_name)
                extension = os.path.splitext(file_name)[1]
                _, encoded_img = cv2.imencode(extension, img_resized)
                with open(img_save_path, mode='w+b') as f:
                    encoded_img.tofile(f)
                
                # --- 5. 마스크 저장 (한글 경로 대응) ---
                mask_save_path = os.path.join(save_mask_dir, file_name)
                _, encoded_mask = cv2.imencode(".png", mask_resized)
                with open(mask_save_path, mode='w+b') as f:
                    encoded_mask.tofile(f)

if __name__ == "__main__":
    # 전체를 다 돌리려면 "all", 학습용만 하려면 "Training"
    run_preprocessing(mode="Training")