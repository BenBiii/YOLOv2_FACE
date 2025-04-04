# -*- coding: utf-8 -*-

import os
import shutil
from xml.dom.minidom import Document
from PIL import Image

# 경로 설정
src_folder = './train'  # 원본 데이터셋 경로
image_out = './JPEGImages_new'
anno_out = './Annotations_new'

# 초기 파일 인덱스 설정
start_index = 16103

# 디렉토리 초기화
def clear_dirs():
    for d in [image_out, anno_out]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

# XML 생성 함수
def create_voc_xml(img_path, bbox_list, save_path, filename_id):
    img = Image.open(img_path)
    width, height = img.size
    depth = len(img.getbands())  # RGB -> 3

    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode('face_dataset'))
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(f"{filename_id:06}.jpg"))
    annotation.appendChild(filename)

    size = doc.createElement('size')
    for tag, value in zip(['width', 'height', 'depth'], [width, height, depth]):
        elem = doc.createElement(tag)
        elem.appendChild(doc.createTextNode(str(value)))
        size.appendChild(elem)
    annotation.appendChild(size)

    for bbox in bbox_list:
        x, y, w, h = bbox
        obj = doc.createElement('object')

        name = doc.createElement('name')
        name.appendChild(doc.createTextNode('face'))
        obj.appendChild(name)

        bndbox = doc.createElement('bndbox')
        for tag, val in zip(['xmin', 'ymin', 'xmax', 'ymax'], [x, y, x + w - 1, y + h - 1]):
            coord = doc.createElement(tag)
            coord.appendChild(doc.createTextNode(str(val)))
            bndbox.appendChild(coord)

        obj.appendChild(bndbox)
        annotation.appendChild(obj)

    with open(save_path, 'w') as f:
        f.write(doc.toprettyxml(indent='    '))

# 메인 실행 함수
def convert_dataset():
    clear_dirs()
    idx = start_index

    files = os.listdir(src_folder)
    files = [f for f in files if f.endswith('.jpg')]
    files.sort()  # 정렬은 필요에 따라

    for img_file in files:
        base_name = img_file.split('.jpg')[0]
        xml_file = base_name + '.xml'
        img_path = os.path.join(src_folder, img_file)
        xml_path = os.path.join(src_folder, xml_file)

        # xml 존재 확인
        if not os.path.exists(xml_path):
            continue

        # 바운딩 박스 추출
        from xml.etree import ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bbox_list = []

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            x1 = int(bndbox.find('xmin').text)
            y1 = int(bndbox.find('ymin').text)
            x2 = int(bndbox.find('xmax').text)
            y2 = int(bndbox.find('ymax').text)
            bbox_list.append([x1, y1, x2 - x1, y2 - y1])

        # 이미지 복사
        new_img_path = os.path.join(image_out, f"{idx:06}.jpg")
        shutil.copy(img_path, new_img_path)

        # 어노테이션 XML 저장
        new_xml_path = os.path.join(anno_out, f"{idx:06}.xml")
        create_voc_xml(img_path, bbox_list, new_xml_path, idx)

        idx += 1

    print(f"총 변환된 데이터 수: {idx - start_index}")

if __name__ == "__main__":
    convert_dataset()
