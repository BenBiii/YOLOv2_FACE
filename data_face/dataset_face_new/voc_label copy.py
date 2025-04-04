import xml.etree.ElementTree as ET
import os
from os import getcwd

# 클래스 이름
classes = ["face"]

# 디렉토리 경로 설정
anno_dir = 'Annotations_new'
image_dir = 'JPEGImages_new'
label_dir = 'labels_new'

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(image_id):
    in_file = open(f'{anno_dir}/{image_id}.xml', encoding='utf-8')
    out_file = open(f'{label_dir}/{image_id}.txt', 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (
            float(xmlbox.find('xmin').text),
            float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymin').text),
            float(xmlbox.find('ymax').text)
        )
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} " + " ".join([str(round(a, 6)) for a in bb]) + '\n')

if __name__ == '__main__':
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_ids = [os.path.splitext(f)[0] for f in image_files]

    list_file = open('train.txt', 'w')
    # ⚠️ f-string 안에서 백슬래시 쓰지 말고, 미리 경로 문자열 가공
    cwd_path = getcwd().replace("\\", "/")

    for image_id in image_ids:
        img_path = f"{cwd_path}/{image_dir}/{image_id}.jpg"
        list_file.write(img_path + "\n")
        convert_annotation(image_id)

    list_file.close()

    print("✅ YOLO 라벨 변환 완료.")
