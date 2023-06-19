# 这个代码是用来将权重和模型结构完整的保存

# 用import导入自己的网络模型
from nets_student import ssd_student_self  # 这里需要手动修改
from nets_student import ssd_student
import torch
from utils.config import Config
model = ssd_student_self.get_ssd_self_KD("detection", 2)  # 模型实例化

model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('E:/graduate/自蒸馏检测/model_data/94.21_512.pth').items()}) # 加载模型权重
#model.load_state_dict(torch.load('E:/尹以鹏/西科/毕业论文/权重/CSSD(自蒸馏)/一次检测/95.53/95.56(只有权值).pth'))
torch.save(model, r'../model_data/whole_target_model.pth')
print("模型保存完成")