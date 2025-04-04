from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, detection_collate
from yolov2 import Yolov2
from torch import optim
from util.network import adjust_learning_rate, WeightLoader
from tensorboardX import SummaryWriter
from config import config as cfg
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--max_epochs', dest='max_epochs', help='number of epochs to train', default=150, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch', default=1, type=int)
    parser.add_argument('--dataset', dest='dataset', default='face', type=str)
    parser.add_argument('--nw', dest='num_workers', help='number of workers to load training data', default=8, type=int)
    parser.add_argument('--output_dir', dest='output_dir', default='output', type=str)
    parser.add_argument('--use_tfboard', dest='use_tfboard', default=False, type=bool)
    parser.add_argument('--display_interval', dest='display_interval', default=10, type=int)
    parser.add_argument('--mGPUs', dest='mGPUs', default=False, type=bool)
    parser.add_argument('--save_interval', dest='save_interval', default=20, type=int)
    parser.add_argument('--cuda', dest='use_cuda', default=False, type=bool)
    parser.add_argument('--resume', dest='resume', default=False, type=bool)
    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch', default=100, type=int)
    parser.add_argument('--exp_name', dest='exp_name', default='default', type=str)
    parser.add_argument('--nan', dest='nan', default=False, type=bool, help='Enable anomaly detection (torch.autograd.set_detect_anomaly(True))')
    args = parser.parse_args()
    return args

def get_dataset(datasetnames):
    names = datasetnames.split('+')
    dataset = RoiDataset(get_imdb(names[0]))
    print('load dataset {}'.format(names[0]))
    for name in names[1:]:
        tmp = RoiDataset(get_imdb(name))
        dataset += tmp
        print('load and add dataset {}'.format(name))
    return dataset

def patched_load(self, model, weights_file):
    self.start = 0
    fp = open(weights_file, 'rb')
    header = np.fromfile(fp, count=4, dtype=np.int32)
    self.buf = np.fromfile(fp, dtype=np.float32)
    fp.close()
    size = self.buf.size
    self.dfs(model)
    if size != self.start:
        print("Warning: {} weights remain unused.".format(size - self.start))
WeightLoader.load = patched_load

def train():
    args = parse_args()
    if args.nan:
        torch.autograd.set_detect_anomaly(True)
    args.lr = cfg.lr
    args.decay_lrs = cfg.decay_lrs
    args.weight_decay = cfg.weight_decay
    args.momentum = cfg.momentum
    args.batch_size = cfg.batch_size
    args.pretrained_model = r"C:\python_work\tftrain\workspace\YOLOv2_FACE\output\yolo-voc.weights"
    print('Called with args:')
    print(args)
    lr = args.lr
    if args.use_tfboard:
        if args.exp_name == 'default':
            writer = SummaryWriter()
        else:
            writer = SummaryWriter('runs/' + args.exp_name)
    if args.dataset == 'face':
        args.imdb_name = 'widerface_train'
        args.imdbval_name = 'widerface_val'
    elif args.dataset == 'voc07train':
        args.imdb_name = 'voc_2007_train'
        args.imdbval_name = 'voc_2007_train'
    elif args.dataset == 'voc07trainval':
        args.imdb_name = 'voc_2007_trainval'
        args.imdbval_name = 'voc_2007_trainval'
    elif args.dataset == 'voc0712trainval':
        args.imdb_name = 'voc_2007_trainval+voc_2012_trainval'
        args.imdbval_name = 'voc_2007_test'
    else:
        raise NotImplementedError
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('loading dataset....')
    train_dataset = get_dataset(args.imdb_name)
    print('dataset loaded.')
    print('training rois number: {}'.format(len(train_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate, drop_last=True)
    print('initialize the model')
    tic = time.time()
    model = Yolov2(weights_file=args.pretrained_model)
    toc = time.time()
    print('model loaded: cost time {:.2f}s'.format(toc - tic))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.resume:
        print('resume training enable')
        resume_checkpoint_name = 'yolov2_epoch_{}.pt'.format(args.checkpoint_epoch)
        resume_checkpoint_path = os.path.join(output_dir, resume_checkpoint_name)
        print('resume from {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        print('learning rate is {}'.format(lr))
        adjust_learning_rate(optimizer, lr)
    if args.use_cuda:
        model.cuda()
    if args.mGPUs:
        model = nn.DataParallel(model)
    model.train()
    iters_per_epoch = int(len(train_dataset) / args.batch_size)
    current_epoch = args.start_epoch
    try:
        for epoch in range(args.start_epoch, args.max_epochs + 1):
            current_epoch = epoch
            loss_temp = 0
            tic = time.time()
            train_data_iter = iter(train_dataloader)
            if epoch in args.decay_lrs:
                lr = args.decay_lrs[epoch]
                adjust_learning_rate(optimizer, lr)
                print('adjust learning rate to {}'.format(lr))
            if cfg.multi_scale and epoch in cfg.epoch_scale:
                cfg.scale_range = cfg.epoch_scale[epoch]
                print('change scale range to {}'.format(cfg.scale_range))
            for step in tqdm(range(iters_per_epoch), desc=f"Epoch {epoch}/{args.max_epochs}", leave=False):
                if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
                    scale_index = np.random.randint(*cfg.scale_range)
                    cfg.input_size = cfg.input_sizes[scale_index]
                    print('change input size {}'.format(cfg.input_size))
                im_data, boxes, gt_classes, num_obj = next(train_data_iter)
                if args.use_cuda:
                    im_data = im_data.cuda()
                    boxes = boxes.cuda()
                    gt_classes = gt_classes.cuda()
                    num_obj = num_obj.cuda()
                im_data_variable = Variable(im_data)
                while True:
                    box_loss, iou_loss, class_loss = model(im_data_variable, boxes, gt_classes, num_obj, training=True)
                    loss = box_loss.mean() + iou_loss.mean() + class_loss.mean()
                    if torch.isnan(loss):
                        print("[epoch %2d][step %4d] Loss is nan. Repeating this step..." % (epoch, step + 1))
                        continue
                    else:
                        break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_temp += loss.item()
                if (step + 1) % args.display_interval == 0:
                    toc = time.time()
                    loss_temp /= args.display_interval
                    iou_loss_v = iou_loss.mean().item()
                    box_loss_v = box_loss.mean().item()
                    class_loss_v = class_loss.mean().item()
                    print("[epoch %2d][step %4d/%4d] loss: %.4f, lr: %.2e, time cost %.1fs" %
                          (epoch, step + 1, iters_per_epoch, loss_temp, lr, toc - tic))
                    print("\t\t\tiou_loss: %.4f, box_loss: %.4f, cls_loss: %.4f" %
                          (iou_loss_v, box_loss_v, class_loss_v))
                    if args.use_tfboard:
                        n_iter = (epoch - 1) * iters_per_epoch + step + 1
                        writer.add_scalar('losses/loss', loss_temp, n_iter)
                        writer.add_scalar('losses/iou_loss', iou_loss_v, n_iter)
                        writer.add_scalar('losses/box_loss', box_loss_v, n_iter)
                        writer.add_scalar('losses/cls_loss', class_loss_v, n_iter)
                    loss_temp = 0
                    tic = time.time()
            if epoch % 2 == 0:
                save_name = os.path.join(output_dir, 'yolov2_epoch_{}.pt'.format(epoch))
                torch.save({
                    'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                    'epoch': epoch,
                    'lr': lr
                }, save_name)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Saving checkpoint at epoch {}...".format(current_epoch))
        save_name = os.path.join(output_dir, 'yolov2_epoch_{}.pt'.format(current_epoch))
        torch.save({
            'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
            'epoch': current_epoch,
            'lr': lr
        }, save_name)
        print("Checkpoint saved as {}.".format(save_name))
        print("To resume training from this epoch, run the following command in cmd:")
        print("python train.py --resume True --checkpoint_epoch {} --dataset {} --max_epochs {}".format(current_epoch, args.dataset, args.max_epochs))
        return

if __name__ == '__main__':
    train()
