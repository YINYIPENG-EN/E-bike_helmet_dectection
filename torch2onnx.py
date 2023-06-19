import torch


output_path = "pruned_onnx_ckpt/ssd_target_512.onnx"
ckpt_path = 'model_data/target_512.pth'
device = torch.device('cuda')
model = torch.load(ckpt_path, map_location=device)  # 调用这个权重的时候会自动调用nets_student/ssd_student_self.py网络模型
del model.conv1
del model.upsample
model.phase = 'detection'
model.eval()

x = torch.zeros(1, 3, 512, 512).to(device)
output_names = ["output0", "output1", "output2"]
input_names = ["images"]
torch.onnx.export(model, x, output_path, verbose=True, input_names=input_names,
                  output_names=output_names, do_constant_folding=True, opset_version=12)

