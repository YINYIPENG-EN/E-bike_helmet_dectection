import colorsys
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pysnooper
import torchsnooper
from IPython import embed
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets_student import ssd_student
from utils.augmentations import letterbox

from utils.box_utils import letterbox_image, ssd_correct_boxes
from nets_student.ssd_student_layers import Detect
from utils.common import DetectMultiBackend
from utils.config import Config
from second_det import Second_Detec
from tools.Net_Vision import draw_cam
warnings.filterwarnings("ignore")

MEANS = (104, 117, 123)


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的config里面的num_classes、
#   model_path和classes_path参数的修改
# --------------------------------------------#
# ssd1 = SSD1()
class SSD(object):
    _defaults = {
        "classes_path": 'model_data/new_classes.txt',
        "input_shape": (512, 512, 3),
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化SSD
    # ---------------------------------------------------#
    def __init__(self, opt, **kwargs):
        self.opt = opt
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()



    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):

        # -------------------------------#
        #   计算总的类的数量
        # -------------------------------#

        self.num_classes = len(self.class_names) + 1

        device = torch.device('cuda' if self.opt.cuda else 'cpu')

        # ------------剪枝模型加载-------------------------
        # trt加载
        if self.opt.trt:
            self.net = DetectMultiBackend(self.opt.target_weights, device, False)
        else:
            model = torch.load(self.opt.target_weights, map_location=device)
            model.phase = 'detection'
            self.net = model.eval()
            if self.opt.cuda:
                self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = True
                self.net = self.net.cuda()

        self.detect = Detect(self.num_classes, 0, 200, self.opt.conf_thres, self.opt.iou_thres)
        # -----------------------------------------------



        #print('{} model, anchors, and classes loaded.'.format(self.opt.target_weights))
        print('model, anchors, and classes loaded.')
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    #@torchsnooper.snoop()
    def detect_image(self, image):
        # PIL读取的数据在trt下效果会差，具体原因不明
        with torch.no_grad():
            image0 = image  # copy image
            image_shape = image.shape[:2]  # get original image shape
            # image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)  # BGR2RGB
            # image0 = Image.fromarray(image0)  # image0 cv2Image
            image = letterbox(image, self.opt.input_shape)  # 这里的letterbox采用的yolov5的

            image = image[0] - MEANS
            image = image.transpose((2, 0, 1))[::-1]  # HWC->CHW
            image = np.ascontiguousarray(image)

            images = torch.from_numpy(image)  # numpy to tensor
            if self.cuda:
                images = images.cuda()
            if self.opt.trt:
                images = images.half() if self.net.fp16 else images.float()
            else:
                images = images.float()
            if len(image.shape) == 3:
                photo = images[None]  # expand for batch dim

            # ---------------------------------------------------#
            #   传入网络进行预测 torch output:(1,num_classes, 200, 5)
            # ---------------------------------------------------#
            preds = self.net(photo)
            if self.opt.trt:  # get onnx or engine outputs
                loc = preds[0]
                conf = preds[1]
                priors = preds[2]
                preds = self.detect.forward(loc, nn.Softmax(dim=-1).forward(conf), priors)
            else:  # torch
                loc = preds[0]
                conf = preds[1]
                priors = preds[2]
                preds = self.detect.forward(loc[0], nn.Softmax(dim=-1).forward(conf[0]), priors)
            top_conf = []
            top_label = []
            top_bboxes = []
            # ---------------------------------------------------#
            #   preds的shape为 1, num_classes, top_k, 5
            # ---------------------------------------------------#
            for i in range(preds.size(1)):
                j = 0
                while preds[0, i, j, 0] >= self.opt.conf_thres:  # preds[...,0] 第0个维度是conf
                    # ---------------------------------------------------#
                    #   score为当前预测框的得分
                    #   label_name为预测框的种类
                    # ---------------------------------------------------#
                    score = preds[0, i, j, 0]
                    label_name = self.class_names[i - 1]
                    # ---------------------------------------------------#
                    #   pt的shape为4, 当前预测框的左上角右下角
                    # ---------------------------------------------------#
                    pt = (preds[0, i, j, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]  # 这里的坐标是相对于网络输入大小图像的而不是原图
                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    j = j + 1

        # 如果不存在满足门限的预测框，直接返回原图
        if len(top_conf) <= 0:
            return image0

        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), \
                                                 np.expand_dims(top_bboxes[:, 1], -1), \
                                                 np.expand_dims(top_bboxes[:, 2], -1), \
                                                 np.expand_dims(top_bboxes[:, 3], -1)

        # -----------------------------------------------------------#
        #   去掉灰条部分
        # -----------------------------------------------------------#
        boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                  np.array([self.opt.input_shape, self.opt.input_shape]), image_shape)

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.opt.input_shape, 1)

        for i, c in enumerate(top_label):
            predicted_class = c
            score = top_conf[i]

            top, left, bottom, right = boxes[i]  # y1,x1,y2,x2
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))  # y1
            left = max(0, np.floor(left + 0.5).astype('int32'))  # x1
            bottom = min(image0.shape[0], np.floor(bottom).astype('int32'))  # 原图的H和boxes右下角y比较防止越界
            right = min(image0.shape[1], np.floor(right).astype('int32'))
            axis = [top, left, bottom, right]

            label = '{} {:.2f}'.format(predicted_class, score)
            cv2.putText(image0, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            print(label, top, left, bottom, right)
            for i in range(thickness):
                cv2.rectangle(image0,  (left, top), (right, bottom), (255, 0, 0), 1)
        return image0, axis


