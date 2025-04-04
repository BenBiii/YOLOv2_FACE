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
from util.network import adjust_learning_rate
from tensorboardX import SummaryWriter
from config import config as cfg
from tqdm import tqdm  # progress bar 추가


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--max_epochs', dest='max_epochs', help='number of epochs to train', default=250, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch', default=1, type=int)
    parser.add_argument('--dataset', dest='dataset', default='face', type=str)
    parser.add_argument('--nw', dest='num_workers', help='number of workers to load training data', default=8, type=int)
    parser.add_argument('--output_dir', dest='output_dir', default='output', type=str)
    parser.add_argument('--use_tfboard', dest='use_tfboard', default=False, type=bool)
    parser.add_argument('--display_interval', dest='display_interval', default=20, type=int)
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


def train():
    args = parse_args()

    # Enable anomaly detection to track operations that cause NaN loss
    if args.nan:
        torch.autograd.set_detect_anomaly(True)

    # define the hyper parameters first
    args.lr = cfg.lr
    args.decay_lrs = cfg.decay_lrs
    args.weight_decay = cfg.weight_decay
    args.momentum = cfg.momentum
    args.batch_size = cfg.batch_size
    args.pretrained_model = os.path.join('data', 'pretrained', 'darknet19_448.weights')

    print('Called with args:')
    print(args)

    lr = args.lr

    # initial tensorboardX writer
    if args.use_tfboard:
        if args.exp_name == 'default':
            writer = SummaryWriter()
        else:
            writer = SummaryWriter('runs/' + args.exp_name)

    if '+' in args.dataset:
        # 여러 dataset 병합: 예) widerface_train+face_new_train
        args.imdb_name = args.dataset

        # val dataset은 기본 widerface_val로 고정하거나 조합해도 가능
        args.imdbval_name = 'widerface_val'  # 또는 'face_new_val' 혹은 둘 다 병합도 가능
    elif args.dataset == 'face':
        args.imdb_name = 'widerface_train'
        args.imdbval_name = 'widerface_val'
    elif args.dataset == 'face_new':
        args.imdb_name = 'face_new_train'
        args.imdbval_name = 'face_new_val'
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
        raise NotImplementedError("Unknown dataset: {}".format(args.dataset))

    #if args.dataset == 'face':
    #    # factory.py
    #    args.imdb_name = 'widerface_train'
    #    args.imdbval_name = 'widerface_val'
    #elif args.dataset == 'voc07train':
    #    args.imdb_name = 'voc_2007_train'
    #    args.imdbval_name = 'voc_2007_train'
    #elif args.dataset == 'voc07trainval':
    #    args.imdb_name = 'voc_2007_trainval'
    #    args.imdbval_name = 'voc_2007_trainval'
    #elif args.dataset == 'voc0712trainval':
    #    args.imdb_name = 'voc_2007_trainval+voc_2012_trainval'
    #    args.imdbval_name = 'voc_2007_test'
    #else:
    #    raise NotImplementedError

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load dataset
    print('loading dataset....')
    train_dataset = get_dataset(args.imdb_name)
    print('dataset loaded.')
    print('training rois number: {}'.format(len(train_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=detection_collate, drop_last=True)

    # initialize the model
    print('initialize the model')
    tic = time.time()
    model = Yolov2(weights_file=args.pretrained_model)
    toc = time.time()
    print('model loaded: cost time {:.2f}s'.format(toc - tic))

    # initialize the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

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

    # set the model mode to train because some layers behave differently during training.
    model.train()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)
    current_epoch = args.start_epoch  # Modified: 현재 에포크 번호 저장

    # Modified: 전체 학습 루프를 try/except 블록으로 감쌌습니다.
    try:
        # start training
        for epoch in range(args.start_epoch, args.max_epochs + 1):
            current_epoch = epoch  # 현재 에포크 번호 업데이트
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

            if epoch % 3 == 0:
            #if epoch % 10 == 0:
                save_name = os.path.join(output_dir, 'yolov2_epoch_{}.pt'.format(epoch))
                torch.save({
                    'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                    'epoch': epoch,
                    'lr': lr
                }, save_name)
    except KeyboardInterrupt:
        # Added: KeyboardInterrupt 발생 시 현재 에포크 상태 저장 및 재시작 명령어 출력
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
