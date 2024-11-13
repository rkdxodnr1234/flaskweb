import os
import zipfile

# 라벨링 파일이 포함된 폴더 경로
folder_path = "D:/JOLJAK/TOMATO_Train"  # 여기에는 라벨링 파일들이 있는 폴더 경로를 입력하세요
output_zip = "D:/JOLJAK/TOMATO_Train.zip"  # 압축 파일 이름

# .txt 파일만 선택하여 압축하기
with zipfile.ZipFile(output_zip, 'w') as zipf:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):  # 라벨링 파일 형식에 맞추어 확장자를 설정하세요
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, folder_path))

print("라벨링 파일만 따로 압축이 완료되었습니다.")
