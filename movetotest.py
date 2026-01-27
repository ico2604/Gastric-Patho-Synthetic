import pandas as pd
import os
import shutil
from tqdm import tqdm

base_path = './data/raw'
train_path = os.path.join(base_path, 'Training')
test_path = os.path.join(base_path, 'Test')

# 1. CSV 로드
df = pd.read_csv(os.path.join(train_path, 'label.csv'))

# 2. 각 항목당 무조건 250개씩 추출
unique_cats = df['category'].unique()
test_rows = []

for cat in unique_cats:
    cat_df = df[df['category'] == cat]
    sample_n = min(len(cat_df), 250)
    test_rows.append(cat_df.sample(n=sample_n, random_state=42))

test_df = pd.concat(test_rows).reset_index(drop=True)

# 3. 데이터 이동 (규칙: 01=TS, 02=TL)
success_count = 0
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="데이터 이동 중"):
    # 파일명만 추출 (예: NIA6_S_STDI_00002.png)
    img_name = os.path.basename(row['file_name'])
    json_name = img_name.replace('.png', '.json')
    
    # 폴더명 추출 (예: TS_미만형선암)
    ts_folder = os.path.dirname(row['file_name']) 
    tl_folder = ts_folder.replace('TS_', 'TL_') # 폴더명의 TS를 TL로 변경
    
    # --- [이미지 이동 경로] ---
    # Training\01.원천데이터\TS_폴더\이미지.png
    src_img = os.path.join(train_path, '01.원천데이터', ts_folder, img_name)
    dst_img = os.path.join(test_path, '01.원천데이터', ts_folder, img_name)
    
    # --- [JSON 이동 경로] ---
    # Training\02.라벨링데이터\TL_폴더\라벨.json
    src_json = os.path.join(train_path, '02.라벨링데이터', tl_folder, json_name)
    dst_json = os.path.join(test_path, '02.라벨링데이터', tl_folder, json_name)
    
    # 목적지 폴더 생성
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    
    try:
        # 파일 존재 확인 후 이동
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_json):
            shutil.move(src_json, dst_json)
        success_count += 1
    except Exception as e:
        continue

# 4. 테스트 결과 CSV 저장
test_df.to_csv(os.path.join(test_path, 'label.csv'), index=False)

print(f"\n✅ 이동 완료! (총 {success_count} 세트)")
print(f"이미지 위치: Test/01.원천데이터/TS_...")
print(f"라벨 위치: Test/02.라벨링데이터/TL_...")