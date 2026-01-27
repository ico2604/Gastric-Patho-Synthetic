import pandas as pd
import os
import shutil
from tqdm import tqdm

# 1. 경로 설정
base_path = './data/raw'
train_path = os.path.join(base_path, 'Training')
# 삭제 대상 파일을 임시로 보낼 폴더 (진짜 삭제 대신 안전하게 이동)
deleted_path = os.path.join(base_path, 'Deleted_Train')

# 학습용 label.csv 로드
df = pd.read_csv(os.path.join(train_path, 'label.csv'))

# 2. 클래스별로 남길 개수 설정 (예: 클래스당 1000개만 남기기)
keep_n = 750 

# 남길 데이터와 삭제할 데이터 분리
keep_df = df.groupby('category', group_keys=False).apply(lambda x: x.sample(n=min(len(x), keep_n), random_state=42)).reset_index(drop=True)
delete_df = df[~df['file_name'].isin(keep_df['file_name'])].reset_index(drop=True)

print(f"남길 데이터 개수: {len(keep_df)}개")
print(f"삭제(이동)할 데이터 개수: {len(delete_df)}개")

# 3. 파일 이동(삭제) 실행
for idx, row in tqdm(delete_df.iterrows(), total=len(delete_df), desc="데이터 정리 중"):
    # 이미지 및 JSON 경로 설정
    img_rel_path = row['file_name']
    json_rel_path = img_rel_path.replace('TS_', 'TL_').replace('.png', '.json')
    
    # 원본 경로
    src_img = os.path.join(train_path, '01.원천데이터', img_rel_path)
    src_json = os.path.join(train_path, '02.라벨링데이터', json_rel_path)
    
    # 이동 대상 경로 (Deleted_Train 폴더)
    dst_img = os.path.join(deleted_path, '01.원천데이터', img_rel_path)
    dst_json = os.path.join(deleted_path, '02.라벨링데이터', json_rel_path)
    
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    
    # 파일 이동
    try:
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_json):
            shutil.move(src_json, dst_json)
    except Exception as e:
        continue

# 4. label.csv 업데이트 (남은 데이터로만 갱신)
keep_df.to_csv(os.path.join(train_path, 'label.csv'), index=False)

print(f"\n✅ 완료! {len(keep_df)}개만 남기고 나머지는 {deleted_path}로 옮겼습니다.")
print(f"이제 Training/label.csv 파일도 업데이트되어 학습 준비가 끝났습니다.")