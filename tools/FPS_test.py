import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from tools.ssd import SSD
import numpy as np
from utils.box_utils import letterbox_image


MEANS = (104, 117, 123)
class FPS_SSD(SSD):
    def get_FPS(self, image, test_interval):
        # 调整图片使其符合输入要求

        crop_img = np.array(letterbox_image(image, (self.opt.input_shape,self.opt.input_shape)))

        # 图片预处理，归一化
        with torch.no_grad():
            photo = Variable(torch.from_numpy(np.expand_dims(np.transpose(crop_img-MEANS,(2,0,1)),0)).type(torch.FloatTensor))
            if self.opt.cuda:
                photo = photo.cuda()

            preds = self.net(photo)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(test_interval):
            with torch.no_grad():
                starter.record()
                preds = self.net(photo)
                # ---------------剪枝预测--------------------------
                # loc = preds[0]  # loc
                # conf = preds[1]  # conf
                # priors = preds[2]  # priors
                # preds = self.detect.forward(loc, nn.Softmax(dim=-1).forward(conf), priors)
                ender.record()
                torch.cuda.synchronize()
        tact_time = starter.elapsed_time(ender)
        return tact_time