import torch
from torch.nn import Module
from brevitas.nn import QuantIdentity, QuantConv2d
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Int8Bias


class ExampleModel(Module):

    def __init__(self):
        super().__init__()
        self.inp_quant = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)
        self.conv = QuantConv2d(5, 10, (3,3),weight_quant=Int8WeightPerTensorFixedPoint,bias_quant=Int8Bias,output_quant=Int8ActPerTensorFixedPoint,return_quant_tensor=True)
        self.conv2 = QuantConv2d(10, 10, (3,3),weight_quant=Int8WeightPerTensorFixedPoint,bias_quant=Int8Bias,output_quant=Int8ActPerTensorFixedPoint,return_quant_tensor=True)
        self.conv.cache_inference_quant_out = True
        self.conv.cache_inference_quant_bias = True
        self.conv2.cache_inference_quant_out = True
        self.conv2.cache_inference_quant_bias = True

    def forward(self, x):
        return self.conv2(self.conv(self.inp_quant(x)))


model = ExampleModel()
model.eval()  # to trigger caching
model(torch.randn(1, 5, 20, 20))

print(-torch.log2(model.inp_quant.quant_output_scale()))  # 6
print(model.inp_quant.quant_output_bit_width())  # 8
print(-torch.log2(model.conv.quant_weight_scale()))  # 10
print(model.conv.quant_weight_bit_width())  # 8
print(-torch.log2(model.conv.quant_bias_scale()))  # 16
print(model.conv.quant_bias_bit_width())  # 8
print(-torch.log2(model.conv.quant_output_scale()))  # 16
print(model.conv.quant_output_bit_width())  # 24