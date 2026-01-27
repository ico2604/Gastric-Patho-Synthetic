import json
from tqdm import tqdm
import pandas as pd
from glob import glob
import os

# 경로 설정
dataset_path = './data/raw'
# d_cats = ['Training', 'Validation']
d_cats = ['Training', 'Validation', 'Test']


for d in d_cats:
    print(f"\n[{d}] 데이터 처리 시작...")
    
    # 1. 이미지 및 JSON 리스트 확보
    img_list = glob(os.path.join(dataset_path, d, '01.원천데이터', '**', '*.png'), recursive=True)
    json_list = glob(os.path.join(dataset_path, d, '02.라벨링데이터', '**', '*.json'), recursive=True)
    
    # 파일명 매칭을 위한 딕셔너리
    json_lookup = {os.path.splitext(os.path.basename(f))[0]: f for f in json_list}
    
    print(f"- 이미지 개수: {len(img_list)}개")
    print(f"- 찾은 JSON 개수: {len(json_list)}개")

    labels = []
    for img_path in tqdm(img_list, desc="매칭 및 추출 중"):
        fname_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        
        if fname_no_ext in json_lookup:
            target_json = json_lookup[fname_no_ext]
            try:
                with open(target_json, 'r', encoding='utf-8-sig') as f:
                    json_data = json.load(f)
                    
                    # 요청하신 대로 'category' 필드에서 값을 가져옵니다.
                    # 값은 STDI, STNT, STIN, STMX 중 하나가 들어갑니다.
                    category_val = json_data['content']['clinical']['category']
                    
                    parts = img_path.split(os.sep)
                    label_dir = parts[-2]
                    fname = parts[-1]

                    labels.append({
                        'category': category_val, # 컬럼명을 category로 변경
                        'file_name': os.path.join(label_dir, fname)
                    })
            except Exception as e:
                # 파일 읽기 오류나 키(Key)가 없을 경우 스킵
                continue
    
    # 4. 결과 저장 (하나의 label.csv로 통합 저장)
    if labels:
        df = pd.DataFrame(labels)
        save_csv = os.path.join(dataset_path, d, 'label.csv')
        df.to_csv(save_csv, index=None)
        print(f"✅ {d} 저장 완료! (총 {len(df)}개 매칭됨)")
        
        # 데이터가 잘 들어갔는지 카테고리별 개수 확인 (선택 사항)
        print(df['category'].value_counts()) 
    else:
        print(f"❌ {d} 매칭 실패: 데이터를 찾지 못했습니다.")

print("\n🚀 모든 카테고리가 통합된 label.csv 생성이 완료되었습니다!")