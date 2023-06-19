import torch
import torch_pruning as tp
import torch.nn as nn


def Pruning_Model(opt):
    model = torch.load(opt.pruning_weights, map_location='cuda')
    # 如果phase为'test'剪枝不了，会报没有detect属性，这是因为在模型训练保存的时候，没有保存该预测部分，这部分已在检测单独写出了
    model.phase = 'train'
    model.eval()
    num_params_before_pruning = tp.utils.count_params(model)
    strategy = tp.strategy.L1Strategy()
    DG = tp.DependencyGraph()
    DG = DG.build_dependency(model, example_inputs=torch.randn(1, 3, 512, 512))  # input_size是网络的输入大小
    included_layers = model.vgg[23:]

    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m in included_layers:
            pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=strategy(m.weight, amount=0.5))
            print(pruning_plan)  # 执行剪枝
            pruning_plan.exec()
    num_params_after_pruning = tp.utils.count_params(model)
    print(" Params: %s => %s" % (num_params_before_pruning, num_params_after_pruning))
    torch.save(model, opt.output+'pruning_model.pth')