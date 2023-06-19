import torch
import torch.nn as nn
import numpy as np
from loguru import logger as LOGGER
from collections import namedtuple, OrderedDict

class DetectMultiBackend(nn.Module):
    def __init__(self, weights, device=torch.device('cpu'), fp16=False):
        super(DetectMultiBackend, self).__init__()
        cuda = torch.cuda.is_available()
        if weights.split('.')[-1] == "onnx":
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(weights, None)
            output_names = [x.name for x in session.get_outputs()]
            print("Output_names: ", output_names)
            meta = session.get_modelmeta().custom_metadata_map  # metadata
        elif weights.split('.')[-1] == 'engine':
            import tensorrt as trt
            LOGGER.info(f'Loading {weights} for TensorRT inference...')
            device = torch.device('cuda:0')
            # 1.创建一个Binding对象，该对象包含'name', 'dtype', 'shape', 'data', 'ptr'这些属性
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            # 2.读取engine文件并记录log
            with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
                # 将engine进行反序列化，这里的model就是反序列化中的model
                model = runtime.deserialize_cuda_engine(
                    f.read())  # model <class 'tensorrt.tensorrt.ICudaEngine'> num_bindings=2,num_layers=163
            # 3.构建可执行的context(上下文：记录执行任务所需要的相关信息)
            context = model.create_execution_context()  # <IExecutionContext>
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)  # 获得输入输出的名字"images","output0"
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):  # 判断是否为输入
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)  # 放入输出名字 output_names = ['output0']
                shape = tuple(context.get_binding_shape(i))  # 记录输入输出shape
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)  # 创建一个全0的与输入或输出shape相同的tensor
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # 放入之前创建的对象中
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())  # 提取name以及对应的Binding
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        self.__dict__.update(locals())
    def forward(self, im):
        global y
        if self.weights.split('.')[-1] == 'onnx':  # onnx推理
            im = im.cpu().numpy()
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.weights.split('.')[-1] == 'engine':  # tensorRT推理
            b, ch, h, w = im.shape  # batch, channel, height, width
            if self.fp16 and im.dtype != torch.float16:
                im = im.half()
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            # 调用计算核心执行计算过程
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        if isinstance(y, (list, tuple)):  # 多输出
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x