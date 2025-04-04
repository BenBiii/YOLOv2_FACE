import os
import argparse
import time
import torch
from torch.autograd import Variable
from PIL import Image
from test import prepare_im_data
from yolov2 import Yolov2
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from pathlib import Path
from config.config import BASE_DIR  # YOLOv2_FACE 기준 절대경로 자동 인식

base_path = Path(BASE_DIR)

def parse_args():
    parser = argparse.ArgumentParser('Yolo v2 Demo')
    parser.add_argument('--model_path', dest='model_path',
                        default=str(base_path / 'output' / 'yolov2_epoch_54.pt'), type=str)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument('--save-dir', dest='save_dir',
                        default=str(base_path / 'detect'), type=str,
                        help='Directory to save detection images')
    parser.add_argument('--save', dest='save', default=False, type=bool,
                        help='Whether to save detection images')
    args = parser.parse_args()
    return args

def get_unique_save_path(save_dir, image_name):
    base_name, ext = os.path.splitext(image_name)
    counter = 1
    unique_name = image_name
    while (save_dir / unique_name).exists():
        unique_name = f"{base_name}_{counter}{ext}"
        counter += 1
    return save_dir / unique_name

def demo():
    args = parse_args()
    print('Call with args: {}'.format(args))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    images_dir = base_path / 'images'
    images_names = ['image_B.jpg', 'many.jpg', 'image_many.jpg']
    classes = ('face',)

    model = Yolov2()
    print('Loading model checkpoint from {}'.format(args.model_path))
    checkpoint = torch.load(args.model_path, map_location='cuda' if args.use_cuda else 'cpu')
    model.load_state_dict(checkpoint['model'])

    if args.use_cuda:
        model.cuda()
    model.eval()
    print('Model loaded.')

    for image_name in images_names:
        image_path = images_dir / image_name
        img = Image.open(image_path)
        im_data, im_info = prepare_im_data(img)

        if args.use_cuda:
            im_data = im_data.cuda()
        im_data_variable = Variable(im_data)

        tic = time.time()
        yolo_output = model(im_data_variable)
        yolo_output = [item[0].data for item in yolo_output]
        detections = yolo_eval(yolo_output, im_info, conf_threshold=0.4, nms_threshold=0.4)
        toc = time.time()

        print('Detection time: {:.4f}s, FPS: {}'.format(toc - tic, int(1 / (toc - tic)) if toc - tic > 0 else 0))

        if detections.size(0) > 0:
            det_boxes = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
            im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=classes)
        else:
            im2show = img

        if args.save:
            save_path = get_unique_save_path(save_dir, image_name)
            im2show.save(str(save_path))
            print("Saved detection result to {}".format(save_path))

        plt.figure()
        plt.imshow(im2show)
        plt.title(image_name)
        plt.show()

if __name__ == '__main__':
    demo()
