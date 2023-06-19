from IPython import embed

from tools.SSD_one_object import SSD_one_object
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
def Detection_one_object(opt):

    predict_img = opt.image
    predict_video = opt.video
    if opt.model == 'SSD':
        ssd = SSD_one_object(opt)  # 一次检测
        if predict_img:
            while True:
                img = input('Input image filename:')
                try:
                    image = cv2.imread(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = ssd.detect_image(image)
                    r_image.save("img.jpg")
                    r_image.show()
        if predict_video:
            capture = cv2.VideoCapture(opt.video_path)
            fps = 0.0

            while (True):
                t1 = time.time()
                # 读取某一帧
                ref, frame = capture.read()
                # 格式转变，BGRtoRGB
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                # frame = Image.fromarray(np.uint8(frame))
                # 进行检测
                frame = np.array(ssd.detect_image(frame))
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", frame)
                if cv2.waitKey(25) & 0xff == ord('q'):
                    capture.release()
                    cv2.destroyAllWindows()
                    break