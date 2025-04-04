import os

# 이미지가 있는 폴더
image_dir = 'JPEGImages_new'

# 출력할 텍스트 파일 이름
output_file = 'train_new.txt'

def extract_numbers():
    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith('.jpg'))

    with open(output_file, 'w') as f:
        for file in image_files:
            filename = os.path.splitext(file)[0]  # '000001.jpg' -> '000001'
            f.write(filename + '\n')

    print(f"✅ 저장 완료: {output_file}")

if __name__ == "__main__":
    extract_numbers()
