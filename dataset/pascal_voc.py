#pascal_voc.py

from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
import numpy as np
import scipy.sparse
import uuid
import xml.etree.ElementTree as ET
from .voc_eval import voc_eval

#from .imdb import imdb, data_dir
from .imdb import imdb
from config import config as cfg


class pascal_voc(imdb):
    def __init__(self, image_set, year):
        super(pascal_voc, self).__init__('voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set

        self._devkit_path = self._get_default_path()
        #self._
        if self._year.lower() == 'face':
            self._data_path = self._devkit_path
        elif self._year.lower() == 'face_new':
            self._data_path = self._devkit_path
        else:
            self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        #self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        #self._classes = ('aeroplane', 'bicycle', 'bird', 'boat',
        #                 'bottle', 'bus', 'car', 'cat', 'chair',
        #                 'cow', 'diningtable', 'dog', 'horse',
        #                 'motorbike', 'person', 'pottedplant',
        #                 'sheep', 'sofa', 'train', 'tvmonitor')
        self._classes = ('face',)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': False,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        #return os.path.join(data_dir, 'VOCdevkit' + self._year)
        #return r"C:\python_work\tftrain\workspace\YOLOv2_FACE\data_face"
        if self._year == 'face':
            return r"C:\python_work\tftrain\workspace\YOLOv2_FACE\data_face"
        elif self._year == 'face_new':
            return r"C:\python_work\tftrain\workspace\YOLOv2_FACE\data_face\dataset_face_new"

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            #non_diff_objs = [
            #    obj for obj in objs if int(obj.find('difficult').text) == 0]
            
            non_diff_objs = []
            for obj in objs:
                difficult_tag = obj.find('difficult')
                difficult = int(difficult_tag.text) if difficult_tag is not None else 0
                if difficult == 0:
                    non_diff_objs.append(obj)
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                }

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        #filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        filedir = os.path.join(self._devkit_path, 'results', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
                    # 검출된 좌표를 0-based 좌표로 기록 (기존 +1 연산 제거)
                    #for k in range(dets.shape[0]):
                    #    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    #            format(index, dets[k, -1],
                    #                   dets[k, 0], dets[k, 1],
                    #                   dets[k, 2], dets[k, 3]))

    def _do_python_eval(self, output_dir='output'):
        """
        annopath = os.path.join(
            self._devkit_path,
            #'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            #'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        try:
            year_val = int(self._year)
            use_07_metric = True if year_val < 2010 else False
        except ValueError:
            use_07_metric = False  # 기본적으로 False로 설정
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)"""
        
        #annopath = r"C:\python_work\tftrain\workspace\YOLOv2_FACE\data_face\Annotations\{:s}.xml"
        annopath = os.path.join(
            self._data_path,
            'Annotations',
            '{:s}.xml'
        )
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        try:
            year_val = int(self._year)
            use_07_metric = True if year_val < 2010 else False
        except ValueError:
            use_07_metric = False  # 기본적으로 False로 설정
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        mean_ap = np.mean(aps)*100
        """
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)"
                """
        print('Mean AP = {:.2f}%'.format(mean_ap))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}%'.format(ap*100))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')

    def evaluate_detections(self, all_boxes, output_dir=None):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)
