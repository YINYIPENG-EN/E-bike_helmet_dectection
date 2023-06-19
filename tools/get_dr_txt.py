import colorsys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tools.ssd import SSD
from utils.box_utils import letterbox_image, ssd_correct_boxes
from nets_student.ssd_student_layers import Detect
MEANS = (104, 117, 123)

class mAP_SSD(SSD):

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        self.conf_thres = self.opt.conf_thres
        #self.opt.conf_thres = 0.5

        # 计算总的种类
        self.num_classes = len(self.class_names) + 1

        # 载入模型
        # model = get_ssd_student("detection", self.num_classes, self.confidence, self.nms_iou)

        # 载入权重
        print('Loading weights into state dict...')
        device = torch.device('cuda' if self.opt.cuda else 'cpu')
        # model.load_state_dict(torch.load(self.model_path, map_location=device))
        model = torch.load(self.opt.target_weights, map_location=device)
        model.phase = 'detection'
        self.net = model.eval()

        if self.opt.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.opt.target_weights))
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
    def detect_image(self, image_id, image):
        self.detect = Detect(self.num_classes, 0, 200, self.conf_thres, self.opt.iou_thres)
        f = open("./input/detection-results/" + image_id + ".txt", "w")
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.opt.input_shape, self.opt.input_shape)))
        photo = np.array(crop_img, dtype=np.float64)
        # 图片预处理，归一化
        with torch.no_grad():
            photo = Variable(
                torch.from_numpy(np.expand_dims(np.transpose(crop_img - MEANS, (2, 0, 1)), 0)).type(torch.FloatTensor))
            if self.opt.cuda:
                photo = photo.cuda()
            preds, _ = self.net(photo)
            # ---------------剪枝预测--------------------------
            loc = preds[0]  # loc
            conf = preds[1]  # conf
            priors = preds[2]  # priors
            preds = self.detect.forward(loc, nn.Softmax(dim=-1).forward(conf), priors)
            # -----------------------------------------------'''
            top_conf = []
            top_label = []
            top_bboxes = []
            for i in range(preds.size(1)):
                j = 0
                while preds[0, i, j, 0] >= self.opt.conf_thres:
                    score = preds[0, i, j, 0]
                    label_name = self.class_names[i - 1]
                    pt = (preds[0, i, j, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]
                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    j = j + 1

        # 将预测结果进行解码
        if len(top_conf) <= 0:
            return image

        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1],
                                                                                                      -1), np.expand_dims(
            top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条
        boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                  np.array([self.opt.input_shape, self.opt.input_shape]), image_shape)

        for i, c in enumerate(top_label):
            predicted_class = c
            score = str(float(top_conf[i]))

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return
