#!/bin/bash

echo ""
echo "Step 2: 모든 .zip 파일을 전용 폴더에 압축 해제합니다..."
echo "--------------------------------------------------"

# 모든 하위 폴더의 .zip 파일을 찾음
find . -type f -name "*.zip" | while read -r zip_file; do
    dir_path=$(dirname "$zip_file")
    zip_name=$(basename "$zip_file")
    
    # 1. 압축 파일 이름에서 .zip을 떼고 폴더명 생성 (예: TS_위염)
    folder_name="${zip_name%.zip}"
    target_dir="$dir_path/$folder_name"
    
    # 2. 해당 폴더가 없으면 생성
    mkdir -p "$target_dir"
    
    echo "처리 중: $zip_name -> 폴더: $folder_name"
    
    # 3. 압축 해제
    # -o: 덮어쓰기 허용 (중복 워닝 방지)
    # -q: 메시지 출력 최소화
    # -d: 대상 디렉토리 지정
    unzip -oq "$zip_file" -d "$target_dir"
    
    # 만약 'absolute path' 워닝이 계속 거슬린다면 
    # 아래처럼 표준 에러(2)를 무시(> /dev/null)할 수 있습니다.
    # unzip -oq "$zip_file" -d "$target_dir" 2> /dev/null
done

echo "--------------------------------------------------"
echo "모든 작업이 완료되었습니다! 각 폴더 안을 확인해보세요."