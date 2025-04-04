# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from config import config as cfg
from util.bbox import generate_all_anchors, xywh2xxyy, box_transform_inv, box_ious, xxyy2xywh, box_transform
import torch.nn.functional as F


def build_target(output, gt_data, H, W):
    """
    Build the training target for output tensor

    Arguments:
      output_data -- tuple (delta_pred_batch, conf_pred_batch, class_pred_batch)
                      output data of the yolo network.
      gt_data     -- tuple (gt_boxes_batch, gt_classes_batch, num_boxes_batch)
                      ground truth data.
      H, W        -- grid height and width.

    Returns:
      iou_target  -- tensor of shape (B, H * W * num_anchors, 1)
      iou_mask    -- tensor of shape (B, H * W * num_anchors, 1)
      box_target  -- tensor of shape (B, H * W * num_anchors, 4)
      box_mask    -- tensor of shape (B, H * W * num_anchors, 1)
      class_target-- tensor of shape (B, H * W * num_anchors, 1)
      class_mask  -- tensor of shape (B, H * W * num_anchors, 1)
    """
    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    gt_boxes_batch = gt_data[0]
    gt_classes_batch = gt_data[1]
    num_boxes_batch = gt_data[2]

    bsize = delta_pred_batch.size(0)
    num_anchors = 5  # hard code for now

    iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * cfg.noobject_scale

    box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
    box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    anchors = torch.FloatTensor(cfg.anchors)

    # Generate grid anchors (format: x, y, w, h) normalized by grid dimensions.
    all_grid_xywh = generate_all_anchors(anchors, H, W)
    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
    all_anchors_xywh = all_grid_xywh.clone()
    all_anchors_xywh[:, 0:2] += 0.5
    if cfg.debug:
        print('all grid: ', all_grid_xywh[:12, :])
        print('all anchor: ', all_anchors_xywh[:12, :])
    all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)

    # Process each batch independently.
    for b in range(bsize):
        num_obj = num_boxes_batch[b].item()
        delta_pred = delta_pred_batch[b]
        gt_boxes = gt_boxes_batch[b][:num_obj, :]
        gt_classes = gt_classes_batch[b][:num_obj]

        # rescale ground truth boxes to grid scale
        gt_boxes[:, 0::2] *= W
        gt_boxes[:, 1::2] *= H

        all_anchors_xywh = all_anchors_xywh.view(-1, 4)
        box_pred = box_transform_inv(all_grid_xywh, delta_pred)
        box_pred = xywh2xxyy(box_pred)

        # Compute IoU between predicted boxes and ground truth boxes.
        ious = box_ious(box_pred, gt_boxes)
        ious = ious.view(-1, num_anchors, num_obj)
        max_iou, _ = torch.max(ious, dim=-1, keepdim=True)
        if cfg.debug:
            print('ious', ious)

        iou_thresh_filter = max_iou.view(-1) > cfg.thresh
        n_pos = torch.nonzero(iou_thresh_filter).numel()

        if n_pos > 0:
            iou_mask[b][max_iou >= cfg.thresh] = 0

        overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, num_anchors, num_obj)
        gt_boxes_xywh = xxyy2xywh(gt_boxes)

        for t in range(gt_boxes.size(0)):
            # Compute center of gt box to determine the grid cell index.
            gt_box_xywh = gt_boxes_xywh[t]
            gt_class = gt_classes[t]
            cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
            cell_idx = cell_idx_y * W + cell_idx_x
            # Clamp cell_idx to valid range [0, H*W - 1]
            cell_idx = torch.clamp(cell_idx.long(), 0, H * W - 1)

            overlaps_in_cell = overlaps[cell_idx, :, t]
            argmax_anchor_idx = torch.argmax(overlaps_in_cell)

            assigned_grid = all_grid_xywh.view(-1, num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0)
            gt_box = gt_box_xywh.unsqueeze(0)
            target_t = box_transform(assigned_grid, gt_box)
            if cfg.debug:
                print('assigned_grid, ', assigned_grid)
                print('gt: ', gt_box)
                print('target_t, ', target_t)
            box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0)
            box_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class
            class_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
            if cfg.debug:
                print(max_iou[cell_idx, argmax_anchor_idx, :])
            iou_mask[b, cell_idx, argmax_anchor_idx, :] = cfg.object_scale

    return iou_target.view(bsize, -1, 1), \
           iou_mask.view(bsize, -1, 1), \
           box_target.view(bsize, -1, 4), \
           box_mask.view(bsize, -1, 1), \
           class_target.view(bsize, -1, 1).long(), \
           class_mask.view(bsize, -1, 1)


def yolo_loss(output, target):
    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    iou_target = target[0]
    iou_mask = target[1]
    box_target = target[2]
    box_mask = target[3]
    class_target = target[4]
    class_mask = target[5]

    b, _, num_classes = class_score_batch.size()
    class_score_batch = class_score_batch.view(-1, num_classes)
    class_target = class_target.view(-1)
    class_mask = class_mask.view(-1)

    # ignore the gradient for background
    class_keep = class_mask.nonzero().squeeze(1)
    class_score_batch_keep = class_score_batch[class_keep, :]
    class_target_keep = class_target[class_keep]

    box_loss = 1 / b * cfg.coord_scale * F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask, reduction='sum') / 2.0
    iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
    class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

    return box_loss, iou_loss, class_loss
