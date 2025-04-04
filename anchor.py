import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans
import argparse
from config import config as cfg

ANNOTATIONS_PATH = r'C:\python_work\tftrain\workspace\YOLOv2_FACE\data_face\Annotations'
NUM_CLUSTERS = 5

def parse_annotation(path):
    boxes = []
    for fn in os.listdir(path):
        if not fn.endswith('.xml'): continue
        tree = ET.parse(os.path.join(path, fn))
        for obj in tree.findall('object'):
            if obj.find('difficult') is not None and int(obj.find('difficult').text)==1:
                continue
            bb = obj.find('bndbox')
            x1,y1,x2,y2 = map(float, (bb.find('xmin').text, bb.find('ymin').text,
                                      bb.find('xmax').text, bb.find('ymax').text))
            boxes.append([x2-x1, y2-y1])
    return np.array(boxes, dtype=np.float32)

def kmeans_anchor(boxes, k):
    # 1) Normalize by full-image dims → [0,1]
    iw, ih = cfg.input_size
    boxes[:,0] /= iw
    boxes[:,1] /= ih

    # 2) Cluster
    centers = KMeans(n_clusters=k, random_state=0).fit(boxes).cluster_centers_

    # 3) Scale to grid‑cell units
    grid = cfg.test_input_size[0] // cfg.strides  # 416/32 = 13
    anchors = centers * grid

    # Sort ascending by width
    anchors = anchors[np.argsort(anchors[:,0])]
    return anchors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_dir', default=ANNOTATIONS_PATH)
    parser.add_argument('--k', type=int, default=NUM_CLUSTERS)
    args = parser.parse_args()

    boxes = parse_annotation(args.annotation_dir)
    print(f"Loaded {len(boxes)} boxes.")

    anchors = kmeans_anchor(boxes, args.k)
    print("Anchors (grid‑cell units ≤13):")
    for i, (w, h) in enumerate(anchors):
        print(f"Anchor {i+1}: [{w:.4f}, {h:.4f}]")
