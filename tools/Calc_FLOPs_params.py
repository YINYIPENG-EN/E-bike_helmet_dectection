import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import torch_pruning as tp

#model = torch.load(r'../剪枝后mAP/target/results/pruning_target.pth')  # 一次检测 15.87M
#model = torch.load(r'../剪枝后mAP/helmet/results/pruning_2th_model.pth')  # 二次检测16.30M
#model = torch.load(r'../model_data/whole_model.pth')  # 剪枝前一次23.75M
model = torch.load(r'../EC/model_data/whole_2th_model.pth')  # 剪枝前二次24.41M
num_params_before_pruning = tp.utils.count_params(model)
#total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (num_params_before_pruning/1e6))