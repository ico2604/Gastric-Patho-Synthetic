import os

def get_filenames(root_path):
    # 하위 폴더를 포함하여 모든 파일의 '이름'만 추출합니다.
    filenames = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            filenames.append(file)
    return set(filenames)

# 각 경로 설정
test_path = 'data/processed/images_512/Test'
train_path = 'data/processed/images_512/Training'
val_path = 'data/processed/images_512/Validation'

# 파일명 집합 생성
test_files = get_filenames(test_path)
train_files = get_filenames(train_path)
val_files = get_filenames(val_path)

# 중복 확인
train_test_dup = train_files & test_files
train_val_dup = train_files & val_files
test_val_dup = test_files & val_files

# 결과 출력
print(f"Train ↔ Test 중복: {len(train_test_dup)}개")
print(f"Train ↔ Val 중복: {len(train_val_dup)}개")
print(f"Test ↔ Val 중복: {len(test_val_dup)}개")

if train_test_dup or train_val_dup or test_val_dup:
    print("\n[주의] 중복된 파일명이 발견되었습니다!")
    # 예시로 몇 개만 출력해보고 싶다면:
    # print(list(train_test_dup)[:5]) 
else:
    print("\n[안심] 모든 파일명이 고유합니다.")