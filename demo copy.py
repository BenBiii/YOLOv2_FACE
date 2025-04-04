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

def parse_args():
    parser = argparse.ArgumentParser('Yolo v2 Demo')
    parser.add_argument('--model_path', dest='model_path',
                        default=r"C:\python_work\tftrain\workspace\YOLOv2_FACE\output\yolov2_epoch_1.pth",
                        type=str, help='Path to the checkpoint file')
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument('--save-dir', dest='save_dir',
                        default='runs/detect', type=str,
                        help='Directory to save detection images')
    args = parser.parse_args()
    return args

def demo():
    args = parse_args()
    print('Call with args: {}'.format(args))

    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Input images directory and file names
    images_dir = 'images'
    images_names = ['image_B.jpg', 'image8.jpg']
    classes = ('face',)

    # Initialize and load model checkpoint
    model = Yolov2()
    print('Loading model checkpoint from {}'.format(args.model_path))
    if torch.cuda.is_available():
        checkpoint = torch.load(args.model_path)
    else:
        checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if args.use_cuda:
        model.cuda()
    model.eval()
    print('Model loaded.')

    for image_name in images_names:
        image_path = os.path.join(images_dir, image_name)
        img = Image.open(image_path)
        im_data, im_info = prepare_im_data(img)

        if args.use_cuda:
            im_data = im_data.cuda()
        im_data_variable = Variable(im_data)

        tic = time.time()
        yolo_output = model(im_data_variable)
        # Extract tensor data from tuple outputs
        yolo_output = [item[0].data for item in yolo_output]
        detections = yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4)
        toc = time.time()
        cost_time = toc - tic
        print('Detection time: {:.4f}s, FPS: {}'.format(cost_time, int(1/cost_time) if cost_time > 0 else 0))

        if detections.size(0) > 0:
            det_boxes = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
            im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=classes)
        else:
            im2show = img  # No detections, use original image

        # Save the result image to the specified folder
        save_path = save_dir / image_name
        im2show.save(str(save_path))
        print("Saved detection result to {}".format(save_path))

        # Optionally, display the image
        plt.figure()
        plt.imshow(im2show)
        plt.title(image_name)
        plt.show()

if __name__ == '__main__':
    demo()
