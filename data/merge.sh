#!/bin/bash

# 1. 하위 디렉토리를 돌며 병합 작업 수행
echo "Step 1: 병합 작업을 시작합니다..."
find . -type f -name "*.zip.part0" | while read -r first_part; do
    
    dir_path=$(dirname "$first_part")
    filename=$(basename "$first_part")
    base_name="${filename%.part0}"
    
    echo "--------------------------------------------------"
    echo "위치: $dir_path"
    echo "병합 중: $base_name"
    
    # 해당 폴더로 들어가서 병합 후 다시 나오기
    # (절대 경로 문제 방지를 위해 pushd/popd 사용)
    pushd "$dir_path" > /dev/null
    find . -name "$base_name.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "$base_name"
    popd > /dev/null
    
    echo "병합 완료: $dir_path/$base_name"
done