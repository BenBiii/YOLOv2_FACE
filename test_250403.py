import os
import argparse
import time
import numpy as np
import pickle
import torch
import warnings
from torch.autograd import Variable
from PIL import Image
from yolov2 import Yolov2
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from util.network import WeightLoader
from torch.utils.data import DataLoader
from config import config as cfg
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser('Yolo v2 Test')
    # 기본 dataset을 'face'로 설정하여 widerface_val을 사용하도록 함
    parser.add_argument('--dataset', dest='dataset', default='face', type=str, help='Dataset type: "face" for widerface, etc.')
    parser.add_argument('--output_dir', dest='output_dir', default='output', type=str)
    #parser.add_argument('--model_name', dest='model_name',
    #                    default='yolov2_epoch_30', type=str)
    parser.add_argument('--model_path', dest='model_path',  #default=r"C:\python_work\tftrain\workspace\YOLOv2_FACE\output_13fit_250_215stop\yolov2_epoch_190.pt", type=str)
    default=r"C:\python_work\tftrain\workspace\YOLOv2_FACE\output\yolov2_epoch_54.pt", type=str)
    parser.add_argument('--nw', dest='num_workers', help='number of workers to load training data', default=1, type=int)
    parser.add_argument('--bs', dest='batch_size', default=2, type=int)
    parser.add_argument('--cuda', dest='use_cuda', default=False, type=bool)
    parser.add_argument('--vis', dest='vis', default=False, type=bool)
    #parser.add_argument('--conf-thresh', dest='conf_thresh', default=0.3, type=float)
    #parser.add_argument('--nms-thresh', dest='nms_thresh', default=0.45, type=float)
    #####
    parser.add_argument('--conf-thresh', dest='conf_thresh', default=0.5, type=float)
    parser.add_argument('--nms-thresh', dest='nms_thresh', default=0.4, type=float)
    #####
    args = parser.parse_args()
    return args

def prepare_im_data(img):
    """
    Prepare image data that will be fed to the network.
    Returns a tensor of shape (3, H, W) and im_info dictionary.
    """
    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    H, W = cfg.input_size
    im_data = img.resize((W, H))
    im_data = torch.from_numpy(np.array(im_data)).float() / 255.0
    im_data = im_data.permute(2, 0, 1).unsqueeze(0)
    return im_data, im_info

def test():
    args = parse_args()
    if args.vis:
        args.conf_thresh = 0.5
        #args.conf_thresh = 0.4
    print('Called with args:')
    print(args)

    # 데이터셋 선택: dataset이 'face'이면 widerface_val 사용
    if args.dataset.lower() == 'face':
        args.imdbval_name = 'widerface_val'
        #args.imdbval_name = 'widerface_test'
    elif args.dataset.lower() == 'voc07trainval':
        args.imdbval_name = 'voc_2007_trainval'
    elif args.dataset.lower() == 'voc07test':
        args.imdbval_name = 'voc_2007_test'
    else:
        raise NotImplementedError('Dataset {} not implemented.'.format(args.dataset))

    val_imdb = get_imdb(args.imdbval_name)
    val_dataset = RoiDataset(val_imdb, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # pt model root.
    model = Yolov2()
    #model_path = os.path.join(args.output_dir, args.model_name + '.pth')
    print('-------')
    print('Loading model from {}'.format(args.model_path))
    print('-------')
    if torch.cuda.is_available():
        checkpoint = torch.load(args.model_path)
    else:
        checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    if args.use_cuda:
        model.cuda()
    model.eval()
    print('Model loaded.')

    dataset_size = len(val_imdb.image_index)
    all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]
    det_file = os.path.join(args.output_dir, 'detections.pkl')

    img_id = -1
    with torch.no_grad():
        """
        for batch, (im_data, im_infos) in enumerate(val_dataloader):
            if args.use_cuda:
                im_data = im_data.cuda()
            im_data_variable = Variable(im_data)
            yolo_outputs = model(im_data_variable)
            for i in range(im_data.size(0)):
                img_id += 1
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0].item(), 'height': im_infos[i][1].item()}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                        nms_threshold=args.nms_thresh)
                print('Image detect [{}/{}]'.format(img_id+1, len(val_dataset)))"""
        total_images = len(val_dataset)
        pbar = tqdm(total=total_images, desc='Detecting', unit='img')
        for batch, (im_data, im_infos) in enumerate(val_dataloader):
            if args.use_cuda:
                im_data = im_data.cuda()
            im_data_variable = Variable(im_data)
            yolo_outputs = model(im_data_variable)
            for i in range(im_data.size(0)):
                img_id += 1
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0].item(), 'height': im_infos[i][1].item()}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh, nms_threshold=args.nms_thresh)
                pbar.update(1)
                if len(detections) > 0:
                    for cls in range(val_imdb.num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()
                if args.vis:
                    img = Image.open(val_imdb.image_path_at(img_id))
                    if len(detections) > 0:
                        det_boxes = detections[:, :5].cpu().numpy()
                        det_classes = detections[:, -1].long().cpu().numpy()
                        im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
                        plt.figure()
                        plt.imshow(im2show)
                        plt.title("Image {}".format(img_id+1))
                        plt.show()
                        

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print("Evaluating detections...")
    val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)

if __name__ == '__main__':
    test()
