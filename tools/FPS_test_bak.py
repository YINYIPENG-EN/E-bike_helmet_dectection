import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from tools.ssd import SSD
import numpy as np
from utils.box_utils import letterbox_image,ssd_correct_boxes


MEANS = (104, 117, 123)
class FPS_SSD(SSD):
    def get_FPS(self, image, test_interval):
        # 调整图片使其符合输入要求
        image_shape = np.array(np.shape(image)[0:2])
        crop_img = np.array(letterbox_image(image, (self.opt.input_shape,self.opt.input_shape)))
        photo = np.array(crop_img,dtype = np.float64)
        # 图片预处理，归一化
        with torch.no_grad():
            photo = Variable(torch.from_numpy(np.expand_dims(np.transpose(crop_img-MEANS,(2,0,1)),0)).type(torch.FloatTensor))
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
                    label_name = self.class_names[i-1]
                    pt = (preds[0, i, j, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]
                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    j = j + 1
            # 将预测结果进行解码
            #if len(top_conf)>0:
                #top_conf = np.array(top_conf)
                #top_label = np.array(top_label)
                #top_bboxes = np.array(top_bboxes)
                #top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
                # 去掉灰条
                #boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.opt.input_shape,self.opt.input_shape]),image_shape)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
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
                        label_name = self.class_names[i-1]
                        pt = (preds[0, i, j, 1:]).detach().numpy()
                        coords = [pt[0], pt[1], pt[2], pt[3]]
                        top_conf.append(score)
                        top_label.append(label_name)
                        top_bboxes.append(coords)
                        j = j + 1
                # 将预测结果进行解码
                if len(top_conf)>0:
                    top_conf = np.array(top_conf)
                    top_label = np.array(top_label)
                    top_bboxes = np.array(top_bboxes)
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
                    # 去掉灰条
                    boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time