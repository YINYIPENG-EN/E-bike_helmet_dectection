import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from PIL import Image

import torchvision.transforms as transforms

# 卷积层的可视化
def Cnn_View(cnn_output, Or_img):
    '''
    cnn_output.size(1)是获得上一层的通道数
    如果你用的是CPU推理的，那么在cam处你应该将张量放在cpu上【我这里默认用的cuda】
    因为我的网络输入大小为224*224大小，所以需要对resize成224*224，以保证叠加图像大小一致！！
    最后将热力图和原始图进行一个叠加
    '''
    cnn_output = cnn_output.detach()
    Or_img = cv2.imread(Or_img)
    #cam = nn.Conv2d(cnn_output.size(1), 1, 1, 1, 1, bias=False).cuda()  # 512是上一层卷积输出后的通道，可以根据自己的网络修改
    cnn_output1 = cnn_output

    cnn_output = torch.flatten(cnn_output, 1)  # 平铺  (batch_size,512*7*7)
    preds = torch.softmax(cnn_output[0], dim=-1).cpu().numpy()

    #cam_output = cam(cnn_output1)
    weights = torch.ones_like(cnn_output1)
    print(weights.shape)
    cam_output = F.conv2d(cnn_output1, nn.Parameter(weights), None, stride=1, padding=1)
    cam_output[0, :, :, 1] = cam_output[0, :, :, 1 if preds.all() > 0.5 else 0]

    cam_output /= 10
    cam_output[cam_output < 0] = 0
    cam_output[cam_output > 1] = 1
    cam_output = cam_output.cpu().numpy()
    cam_output = cam_output.squeeze(0)
    img = cam_output.transpose(1, 2, 0)
    img = cv2.resize(img, (300, 300))
    img = np.uint8(255 * img)
    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    Or_img = cv2.resize(Or_img, (300, 300))
    out = cv2.addWeighted(Or_img, 0.8, heatmap, 0.4, 0)
    plt.axis('off')
    plt.imshow(out[:, :, ::-1])
    plt.show()


def draw_cam(x):
   y = x.mean([2, 3])
   y = y.cpu().numpy()
   index = np.argmax(y)  # 可以计算得出哪个通道均值最大
   feature = x.cpu().numpy()
   plt.imshow(feature[0, index-1, :, :], cmap='viridis')
   plt.axis('off')
   plt.show()
