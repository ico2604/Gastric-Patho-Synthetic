import json
from tqdm import tqdm
import pandas as pd
from glob import glob
import os

# 경로 설정
dataset_path = './data/raw' # r을 붙여서 백슬래시 문제를 방지합니다.
d_cats = ['Training', 'Validation']

for d in d_cats:
    print(f"\n[{d}] 데이터 처리 시작...")
    
    # 1. 이미지 파일 리스트 확보
    img_list = glob(os.path.join(dataset_path, d, '01.원천데이터', '**', '*.png'), recursive=True)
    
    # 2. 라벨 JSON 파일 리스트 확보 (02.라벨링데이터 폴더 전체를 뒤짐)
    # 폴더 구조가 다르더라도 파일명 매칭을 위해 미리 딕셔너리를 만듭니다.
    json_list = glob(os.path.join(dataset_path, d, '02.라벨링데이터', '**', '*.json'), recursive=True)
    
    # { '파일명': '전체경로' } 딕셔너리 생성 (예: {'NIA6_S_STDI_00002': 'D:\...\00002.json'})
    json_lookup = {os.path.splitext(os.path.basename(f))[0]: f for f in json_list}
    
    print(f"- 이미지 개수: {len(img_list)}개")
    print(f"- 찾은 JSON 개수: {len(json_list)}개")

    labels = []
    for img_path in tqdm(img_list, desc="매칭 중"):
        fname_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        
        # 3. 파일명(확장자 제외)이 같은 JSON이 있는지 확인
        if fname_no_ext in json_lookup:
            target_json = json_lookup[fname_no_ext]
            try:
                with open(target_json, 'r', encoding='utf-8-sig') as f:
                    json_data = json.load(f)
                    
                    # 경로에서 라벨 폴더명과 파일명 추출
                    parts = img_path.split(os.sep)
                    label_dir = parts[-2]
                    fname = parts[-1]

                    labels.append({
                        'tumor_category': json_data['content']['clinical']['tumor_category'],
                        'file_name': os.path.join(label_dir, fname)
                    })
            except Exception as e:
                continue
    
    # 4. 결과 저장
    if labels:
        df = pd.DataFrame(labels)
        save_csv = os.path.join(dataset_path, d, 'label.csv')
        df.to_csv(save_csv, index=None)
        print(f"✅ {d} 저장 완료! (총 {len(df)}개 매칭됨)")
        print(f"파일 저장 위치: {save_csv}")
    else:
        print(f"❌ {d} 매칭 실패: 이미지 파일명과 일치하는 JSON 파일을 찾지 못했습니다.")