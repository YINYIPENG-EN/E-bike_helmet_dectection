import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets_student.ssd_student import get_ssd_student
from nets_student.ssd_training import Generator, MultiBoxLoss, LossHistory
from utils.config import Config
from utils.dataloader import SSDDataset, ssd_dataset_collate

warnings.filterwarnings("ignore")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# @torchsnooper.snoop()
def fit_one_epoch(model, criterion, epoch, optimizer, epoch_size, epoch_size_val, gen, genval, Epoch, loss_history, cuda):
    loc_loss = 0
    conf_loss = 0
    loc_loss_val = 0
    conf_loss_val = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    print('Start Teacher Train')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()

                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]

                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            # ----------------------#
            #   前向传播
            # ----------------------#
            out_teacher, _ = model(images)  # out是tuple类型,(batch_size,8732,4)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   计算损失
            # ----------------------#

            loss_l, loss_c = criterion(out_teacher,
                                       targets)  # loss_l= <class 'torch.Tensor'> tensor(2.1593, device='cuda:0', grad_fn=<DivBackward0>)
            # print("loss_l=", type(loss_l), loss_l)
            loss = (loss_l + loss_c)

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            pbar.set_postfix(**{'loc_loss': loc_loss / (iteration + 1),
                                'conf_loss': conf_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    model.eval()
    print('Start Teacher Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

                out_teacher, _ = model(images)
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out_teacher, targets)

                loc_loss_val += loss_l.item()
                conf_loss_val += loss_c.item()

                pbar.set_postfix(**{'loc_loss': loc_loss_val / (iteration + 1),
                                    'conf_loss': conf_loss_val / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    total_loss = loc_loss + conf_loss
    val_loss = loc_loss_val + conf_loss_val
    loss_history.append_loss(total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    print('Finish Teacher Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))

    torch.save(model, 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


# 剪枝微调
def Pruning_fine(opt):
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = opt.cuda
    # -------------------------------#
    #   Dataloder的使用
    # -------------------------------#
    Use_Data_Loader = True
    loss_history = LossHistory(r'./logs')
    model = torch.load(opt.pruned_model_path)
    model.phase = 'train'
    print('Finished!')

    annotation_path = '2007_train.txt'
    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, Cuda)  # 教师自己的损失函数，原SSD

    model = model.train()
    model = model.cuda()

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = opt.Freeze_lr
        Batch_size = opt.batch_size
        Init_Epoch = opt.Init_Epoch
        Freeze_Epoch = opt.Freeze_Epoch

        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        if Use_Data_Loader:
            train_dataset = SSDDataset(lines[:num_train], (Config["min_dim"], Config["min_dim"]), True)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=ssd_dataset_collate)

            val_dataset = SSDDataset(lines[num_train:], (Config["min_dim"], Config["min_dim"]), False)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=ssd_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train], (Config["min_dim"], Config["min_dim"]),
                            Config["num_classes"]).generate(True)
            gen_val = Generator(Batch_size, lines[num_train:], (Config["min_dim"], Config["min_dim"]),
                                Config["num_classes"]).generate(False)

        for param in model.vgg.parameters():
            param.requires_grad = False

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        for epoch in range(Init_Epoch, Freeze_Epoch):
            # def fit_one_epoch(model,criterion,epoch,optimizer,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
            fit_one_epoch(model, criterion, epoch, optimizer, epoch_size, epoch_size_val, gen, gen_val,
                          Freeze_Epoch,
                          loss_history, Cuda)
            lr_scheduler.step()

    if True:
        lr = opt.UnFreeze_lr
        Batch_size = opt.batch_size
        Freeze_Epoch = opt.Freeze_Epoch
        Unfreeze_Epoch = opt.UnFeeze_epoch

        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        if Use_Data_Loader:
            train_dataset = SSDDataset(lines[:num_train], (Config["min_dim"], Config["min_dim"]), True)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=ssd_dataset_collate)

            val_dataset = SSDDataset(lines[num_train:], (Config["min_dim"], Config["min_dim"]), False)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=ssd_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train], (Config["min_dim"], Config["min_dim"]),
                            Config["num_classes"]).generate(True)
            gen_val = Generator(Batch_size, lines[num_train:], (Config["min_dim"], Config["min_dim"]),
                                Config["num_classes"]).generate(False)

        for param in model.vgg.parameters():

            param.requires_grad = True

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(model, criterion, epoch, optimizer, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch,
                          loss_history,Cuda)
            lr_scheduler.step()


def train(opt):
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = opt.cuda
    devcie = 'cuda' if Cuda else 'cpu'
    # -------------------------------#
    #   Dataloder的使用
    # -------------------------------#
    Use_Data_Loader = True
    loss_history = LossHistory(r'./logs')
    model = get_ssd_student('train', Config['num_classes'])
    model_dict = model.state_dict()
    pretrained_weight = torch.load(opt.target_weights, map_location=devcie)
    pretrained_weight = {k: v for k, v in pretrained_weight.itmes() if pretrained_weight.keys() == model_dict.keys()}
    model_dict.update(pretrained_weight)
    model.load_state_dict(model_dict)
    print('完成预权重的加载')
    model.to(devcie)
    annotation_path = '2007_train.txt'
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, Cuda)  # 教师自己的损失函数，原SSD
    model.train()
    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = opt.Freeze_lr
        Batch_size = opt.batch_size
        Init_Epoch = opt.Init_Epoch
        Freeze_Epoch = opt.Freeze_Epoch

        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        if Use_Data_Loader:
            train_dataset = SSDDataset(lines[:num_train], (Config["min_dim"], Config["min_dim"]), True)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=ssd_dataset_collate)

            val_dataset = SSDDataset(lines[num_train:], (Config["min_dim"], Config["min_dim"]), False)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=ssd_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train], (Config["min_dim"], Config["min_dim"]),
                            Config["num_classes"]).generate(True)
            gen_val = Generator(Batch_size, lines[num_train:], (Config["min_dim"], Config["min_dim"]),
                                Config["num_classes"]).generate(False)

        for param in model.vgg.parameters():
            param.requires_grad = False

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(model, criterion, epoch, optimizer, epoch_size, epoch_size_val, gen, gen_val,
                          Freeze_Epoch,
                          loss_history, Cuda)
            lr_scheduler.step()

    if True:
        lr = opt.UnFreeze_lr
        Batch_size = opt.batch_size
        Freeze_Epoch = opt.Freeze_Epoch
        Unfreeze_Epoch = opt.UnFeeze_epoch

        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        if Use_Data_Loader:
            train_dataset = SSDDataset(lines[:num_train], (Config["min_dim"], Config["min_dim"]), True)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=ssd_dataset_collate)

            val_dataset = SSDDataset(lines[num_train:], (Config["min_dim"], Config["min_dim"]), False)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=ssd_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train], (Config["min_dim"], Config["min_dim"]),
                            Config["num_classes"]).generate(True)
            gen_val = Generator(Batch_size, lines[num_train:], (Config["min_dim"], Config["min_dim"]),
                                Config["num_classes"]).generate(False)

        for param in model.vgg.parameters():
            param.requires_grad = True

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(model, criterion, epoch, optimizer, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch,
                          loss_history, Cuda)
            lr_scheduler.step()
