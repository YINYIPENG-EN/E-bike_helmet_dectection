from PIL import Image
from second_det import Second_Detec
from tools.ssd import SSD  # 一次检测
from tools.ssd1 import SSD1  # 二次检测
import matplotlib.pyplot as plt
import cv2
import numpy as np

def Detection(opt):

    predict_img = opt.image
    predict_video = opt.video
    if opt.model == 'SSD':
        ssd = SSD(opt)  # 一次检测
        ssd1 = SSD1(opt)  # 二次检测
        # 图像测试
        if predict_img:
            while True:
                img = input('Input image filename:')
                try:
                    image = cv2.imread(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image, axis = ssd.detect_image(image)  # r_image为array,axis[y1,x1,y2,x2]
                    img_reco_crop = Second_Detec(r_image, axis[0], axis[1], axis[2], axis[3])
                    second_res = ssd1.detect_image(img_reco_crop)
                    r_image = Image.fromarray(cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB))
                    second_res = Image.fromarray(cv2.cvtColor(second_res, cv2.COLOR_BGR2RGB))
                    r_image.save("一次检测结果.jpg", quality=100)
                    second_res.save("二次检测结果.jpg", quality=100)
                    plt.subplot(1, 2, 1)
                    plt.imshow(r_image)
                    plt.subplot(1, 2, 2)
                    plt.imshow(second_res, interpolation='none')
                plt.savefig("最终.jpg")
                plt.show()
        # 视频测试
        if predict_video:
            video_path = opt.video_path
            if video_path == '0':
                video_path = int(video_path)
            capture = cv2.VideoCapture(video_path)

            capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 15)
            fps = capture.get(cv2.CAP_PROP_FPS)
            # fps = 0.0
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            out = cv2.VideoWriter('test.avi', fourcc, fps, size)
            img_reco_crop = None

            while (True):
                # t1 = time.time()
                # 读取某一帧
                res, frame = capture.read()
                if res != True:
                    break
                if res == True:
                    result = ssd.detect_image(frame)
                    if type(result) is tuple:
                        frame, axis = result[0], result[1]  # frame为opencv格式，axis[y1,x1,y2,x2]
                        #img_reco_crop = Second_Detec(frame, axis[0], axis[1], axis[2], axis[3])  # 返回np.array 得到截取后的图像
                        img_reco_crop = ssd1.detect_image(Second_Detec(frame, axis[0], axis[1], axis[2], axis[3]))
                    if img_reco_crop is not None:
                        frame[(size[1] - 201):(size[1] - 1), (size[0] - 201):(size[0] - 1), :] = img_reco_crop
                        cv2.rectangle(frame, (size[0] - 201, size[1] - 201), (size[0] - 1, size[1] - 1), (225, 66, 66),
                                      1)
                    cv2.imshow("video", frame)
                    out.write(frame)
                    if cv2.waitKey(25) & 0xff == ord('q'):
                        capture.release()
                        cv2.destroyAllWindows()
                        break