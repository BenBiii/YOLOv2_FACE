import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans
import argparse
from config import config as cfg

ANNOTATIONS_PATHS = [
    r'C:\python_work\tftrain\workspace\YOLOv2_FACE\data_face\Annotations',
    r'C:\python_work\tftrain\workspace\YOLOv2_FACE\data_face\dataset_face_new\Annotations'
]
NUM_CLUSTERS = 5

def parse_annotation(paths):
    """
    여러 annotation 폴더에서 bounding box (width, height) 정보를 추출
    """
    boxes = []
    for path in paths:
        for fn in os.listdir(path):
            if not fn.endswith('.xml'):
                continue
            tree = ET.parse(os.path.join(path, fn))
            file_boxes = []
            for obj in tree.findall('object'):
                # difficult 무시
                if obj.find('difficult') is not None and int(obj.find('difficult').text) == 1:
                    continue
                bb = obj.find('bndbox')
                x1, y1, x2, y2 = map(float, (
                    bb.find('xmin').text,
                    bb.find('ymin').text,
                    bb.find('xmax').text,
                    bb.find('ymax').text
                ))
                width = x2 - x1
                height = y2 - y1
                file_boxes.append([width, height])
            if len(file_boxes) > 0:
                avg_box = np.mean(file_boxes, axis=0)
                boxes.append(avg_box)
    return np.array(boxes, dtype=np.float32)

def kmeans_anchor(boxes, k):
    iw, ih = cfg.input_size
    boxes[:, 0] /= iw
    boxes[:, 1] /= ih

    centers = KMeans(n_clusters=k, random_state=0).fit(boxes).cluster_centers_

    grid = cfg.test_input_size[0] // cfg.strides
    anchors = centers * grid

    # ✅ h > 13 이면 비율 유지하며 줄이기
    for i in range(len(anchors)):
        w, h = anchors[i]
        if h > 13:
            ratio = 13.0 / h
            anchors[i] = [w * ratio, 13.0]

    anchors = anchors[np.argsort(anchors[:, 0])]
    return anchors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=NUM_CLUSTERS)
    args = parser.parse_args()

    boxes = parse_annotation(ANNOTATIONS_PATHS)
    print(f"Loaded {len(boxes)} representative boxes from both datasets.")

    anchors = kmeans_anchor(boxes, args.k)

    # ✅ 리스트 형식 문자열로 출력
    formatted = ["[{:.4f}, {:.4f}]".format(w, h) for w, h in anchors]
    print("anchors = [{}]".format(", ".join(formatted)))
