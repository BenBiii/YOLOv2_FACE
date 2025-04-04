import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans
import argparse

# YOLOv2 설정
INPUT_SIZE = (416, 416)   # 입력 이미지 크기 (width, height)
STRIDE = 32               # YOLOv2 stride (보통 32)
GRID_SIZE = INPUT_SIZE[0] // STRIDE  # feature map grid 크기 (416/32 = 13)
NUM_CLUSTERS = 5          # 생성할 anchor 개수

def parse_annotations(ann_dir):
    """
    주어진 VOC 형식 XML 파일에서 각 객체의 폭(width)과 높이(height)를 픽셀 단위로 추출.
    'difficult' 태그가 1인 객체는 무시합니다.
    """
    boxes = []
    for filename in os.listdir(ann_dir):
        if not filename.endswith('.xml'):
            continue
        xml_path = os.path.join(ann_dir, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue
            bb = obj.find('bndbox')
            x1 = float(bb.find('xmin').text)
            y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text)
            y2 = float(bb.find('ymax').text)
            w = x2 - x1
            h = y2 - y1
            boxes.append([w, h])
    return np.array(boxes, dtype=np.float32)

def compute_anchors(boxes, num_clusters, stride, grid_size):
    """
    1. (w, h)를 픽셀 단위에서 [0,1] 범위로 정규화 (입력 크기로 나눔)
    2. 정규화된 값을 대상으로 k-means clustering 수행
    3. 클러스터 중심을 grid cell 단위로 변환 (정규화 값 * grid_size)
    4. 만약 계산된 anchor의 너비 혹은 높이가 grid_size(13)를 초과하면,
       scale factor를 적용하여 최대 값이 grid_size가 되도록 조정.
    """
    iw, ih = INPUT_SIZE
    boxes_norm = boxes.copy()
    boxes_norm[:, 0] /= iw
    boxes_norm[:, 1] /= ih

    # k-means clustering (정규화된 값에 대해)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(boxes_norm)
    centers = kmeans.cluster_centers_

    # grid cell 단위로 변환: normalized 값 * grid_size
    anchors = centers * grid_size

    # 각 anchor에 대해, 너비 혹은 높이가 grid_size를 초과하면 scale factor 적용
    adjusted_anchors = []
    for anchor in anchors:
        w, h = anchor
        max_val = max(w, h)
        if max_val > grid_size:
            scale = grid_size / max_val
            w, h = w * scale, h * scale
        adjusted_anchors.append([w, h])
    anchors = np.array(adjusted_anchors)
    # Sort anchors by width (ascending)
    anchors = anchors[np.argsort(anchors[:,0])]
    return anchors

def main():
    parser = argparse.ArgumentParser(description="Compute YOLOv2 anchors from VOC XML annotations")
    parser.add_argument('--ann_dir', type=str, 
                        default=r'C:\python_work\tftrain\workspace\YOLOv2_FACE\data_face\Annotations',
                        help="VOC XML 어노테이션 파일들이 위치한 폴더 경로")
    parser.add_argument('--num_clusters', type=int, default=NUM_CLUSTERS,
                        help="생성할 anchor 개수 (기본 5)")
    parser.add_argument('--stride', type=int, default=STRIDE,
                        help="YOLOv2 stride (default 32)")
    args = parser.parse_args()

    # 어노테이션 파싱
    boxes = parse_annotations(args.ann_dir)
    print(f"Loaded {len(boxes)} bounding boxes from {args.ann_dir}")

    # k-means clustering을 통해 anchor 계산 (grid cell 단위)
    anchors = compute_anchors(boxes, args.num_clusters, args.stride, GRID_SIZE)
    print("Computed anchors (grid cell units, ≤ grid_size=13):")
    for i, (w, h) in enumerate(anchors):
        print(f"Anchor {i+1}: [{w:.4f}, {h:.4f}]")
        
    # 픽셀 단위로 변환하여 출력 (grid cell unit * stride)
    anchors_pixels = anchors * args.stride
    print("Computed anchors (pixel units):")
    for i, (w, h) in enumerate(anchors_pixels):
        print(f"Anchor {i+1}: [{w:.1f}, {h:.1f}]")

if __name__ == '__main__':
    main()
