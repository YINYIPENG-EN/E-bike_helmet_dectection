import argparse
from tools.Detection import Detection  # 单窗口显示
from tools.Detection_Double_Windows import Detection_v2  # 双窗口显示
from tools.Detection_One_Object import Detection_one_object  # 只检测一个目标
from tools.Model_Pruning import Pruning_Model  # 模型剪枝
from tools.FPS_test import FPS_SSD
from tools.get_dr_txt import mAP_SSD
from tools.train_pruning_fine import Pruning_fine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SSD', help='choose model')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--output', type=str, default='', help='output path')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--target_weights', type=str, default='pruned_trt_ckpt/ssd_target_512.engine', help='target detection weights path')
    parser.add_argument('--helmet_weights', type=str, default='pruned_trt_ckpt/ssd_helmet_512.engine', help='helmet detection weights path')
    parser.add_argument('--fps', action='store_true', default=False, help='fps test')
    # 对于电动车头盔检测模型，支持多分辨率输入
    parser.add_argument('--input_shape', type=int, default=512, help='target model input shape')
    parser.add_argument('--input_shape2', type=int, default=512, help='helmet model input shape')

    # -------------------------剪枝-----------------------------------------------------------------
    parser.add_argument('--pruning_model', action='store_true', default=False, help='pruning model')
    parser.add_argument('--pruning_weights', type=str, default='', help='pruning weights path')

    # -------------------------训练配置----------------------------------------------------
    parser.add_argument('--train', action='store_true', default=False, help='model train')
    parser.add_argument('--is_fine', action='store_true', default=False, help='pruning model fine train')
    parser.add_argument('--pruned_model_path', type=str, default='model_data/pruning_model.pth', help='pruned model path')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--Init_Epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--Freeze_Epoch', type=int, default=50, help='model freeze train epoch')
    parser.add_argument('--UnFeeze_epoch', type=int, default=100, help='model unfreeze train epoch')
    parser.add_argument('--Freeze_lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--UnFreeze_lr', type=float, default=1e-4, help='learning rate')

    # --------------------------预测配置-------------------------------------------------
    parser.add_argument('--predict', action='store_true', default=False, help='predict')
    parser.add_argument('--predict_2windows', action='store_true', default=False, help='Detection results show in double windows')
    parser.add_argument('--predict_single', action='store_true', default=False, help='Only detection one object ')
    parser.add_argument('--video', action='store_true', default=False, help='predict video')
    parser.add_argument('--video_path', type=str, default='', help='video_path')
    parser.add_argument('--image', action='store_true', default=False, help='predict image')
    parser.add_argument('--mAP', action='store_true', default=False, help='get mAP')

    # trt预测
    parser.add_argument('--trt', action='store_true', default=False, help='engine predict')

    opt = parser.parse_args()
    print(opt)
    if opt.predict:  # 预测模式
        Detection(opt)  # 单窗口显示

    if opt.predict_2windows:
        Detection_v2(opt)  # 双窗口显示

    if opt.predict_single:
        Detection_one_object(opt)  # 只检测一个目标(需要手动替换一下classes_path和权重)

    if opt.pruning_model:  # 对模型进行剪枝
        Pruning_Model(opt)

    if opt.fps:
        from PIL import Image
        ssd = FPS_SSD(opt)
        test_interval = 100
        img = Image.open('img.jpg')
        tact_time = ssd.get_FPS(img, test_interval)
        #print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')  # 异步预测
        print("predict time : {:.6f}ms,FPS:{}".format(tact_time, 1000/tact_time))  # 异步转同步

    if opt.train:
        if opt.is_fine:
            Pruning_fine(opt)
        else:
            from tools.train_pruning_fine import train
            train(opt)

    # mAP测试
    if opt.mAP:
        import os
        from tqdm import tqdm
        from PIL import Image

        print("正在获取ground_truth....\n")
        os.system('python get_gt_txt.py')
        print("获取ground_truth完成\n")
        ssd = mAP_SSD(opt)
        image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/trainval.txt').read().strip().split()

        if not os.path.exists("./input"):
            os.makedirs("./input")
        if not os.path.exists("./input/detection-results"):
            os.makedirs("./input/detection-results")
        if not os.path.exists("./input/images-optional"):
            os.makedirs("./input/images-optional")

        #for image_id in tqdm(image_ids):
        print("正在测试。。。\n")
        for image_id in image_ids:
            image_path = "./VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
            image = Image.open(image_path)
            # 开启后在之后计算mAP可以可视化
            # image.save("./input/images-optional/"+image_id+".jpg")
            ssd.detect_image(image_id, image)

        print("Conversion completed!")
        print("正在测试mAP....")
        os.system('python get_map.py')






