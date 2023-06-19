from tools.SSD_Two_Windows import SSD_Two_Windows
from tools.ssd1 import SSD1
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

def Detection_v2(opt):
    '''
    可将两次检测结果在视频中分两个窗口显示，而不是都在一个窗口中
    '''
    predict_img = opt.image
    predict_video = opt.video
    if opt.model == 'SSD':
        ssd = SSD_Two_Windows(opt)  # 一次检测
        ssd1 = SSD1(opt)  # 二次检测

        def EC_dection():
            img1 = cv2.imread('cut.jpg')
            r_image1 = ssd1.detect_image(img1)
            # r_image1.save("二次检测.jpg")
            cv2.imwrite("二次检测.jpg", r_image1)
            return r_image1
        if predict_img:
            while True:
                img = input('Input image filename:')
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = ssd.detect_image(image)
                    I = EC_dection()  # 二次检测
                    r_image.save("img.jpg")
                    plt.subplot(1, 2, 1)
                    plt.imshow(r_image)
                    # r_image.show()
                    plt.subplot(1, 2, 2)
                    plt.imshow(I)
                    # I.show()
                plt.show()
        if predict_video:
            # -------------------------------------#
            #   调用摄像头
            #   capture=cv2.VideoCapture("1.mp4")
            # -------------------------------------

            capture = cv2.VideoCapture(opt.video_path)

            fps = capture.get(cv2.CAP_PROP_FPS)
            # fps = 0.0
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            # 获取视频的宽和高
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter('test.avi', fourcc, fps, size)
            Frame_jiange = 6  # 每7帧取一幅图片
            NumFrame = 0
            img_reco_crop = None
            while (True):
                # t1 = time.time()
                # 读取某一帧
                res, frame = capture.read()
                if res == True:

                    # 格式转变，BGRtoRGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 转变成Image
                    frame = Image.fromarray(np.uint8(frame))  # frame= <class 'PIL.Image.Image'>
                    NumFrame = NumFrame + 1
                    if NumFrame % Frame_jiange == 0:
                        # 进行检测
                        frame = np.array(ssd.detect_image(frame))  # frame= <class 'numpy.ndarray'>
                        I = EC_dection()  # 二次检测
                        # I1 = np.array(I)  # 增加 数据类型转换
                        # RGBtoBGR满足opencv显示格式
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # frame= <class 'numpy.ndarray'>
                        # I1 = cv2.cvtColor(I1, cv2.COLOR_RGB2BGR)  # I1= <class 'numpy.ndarray'>
                        out.write(frame)

                        # fps = (fps + (1./(time.time()-t1))) / 2
                        print("fps= %.2f" % (fps))
                        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                            2)

                        cv2.imshow("video", frame)
                        # cv2.namedWindow("二次检测",cv2.WINDOW_NORMAL)
                        cv2.imshow("second dection", I)

                        if cv2.waitKey(25) & 0xff == ord('q'):
                            capture.release()
                            cv2.destroyAllWindows()
                            break