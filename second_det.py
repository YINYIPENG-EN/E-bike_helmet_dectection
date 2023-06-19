import numpy as np
from PIL import Image
import cv2

def Second_Detec(image_copy,top,left,bottom,right):
    image_copy = np.array(image_copy)  # Image转array
    algo_img = image_copy[top:bottom, left:right, :]  # 获取一次检测区域
    img_reco_crop = cv2.resize(algo_img, (200, 200))  # array
    #img_reco_crop = Image.fromarray(np.uint8(img_reco_crop))  # array转Image

    return img_reco_crop
