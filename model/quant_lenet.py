#Based on brevitas examples
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFixedPoint
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8WeightPerTensorFloat
from brevitas.quant import Int8Bias, Int8BiasPerTensorFloatInternalScaling, Int8BiasPerTensorFixedPointInternalScaling, Int16Bias, IntBias

BIAS_QUANTIZER = Int8Bias
WEIGHT_QUANTIZER = Int8WeightPerTensorFixedPoint
ACT_QUANTIZER = Int8ActPerTensorFixedPoint
class QuantLeNet(Module):
    def __init__(self,bit_width=3):
        super(QuantLeNet, self).__init__()
        self.features = nn.Sequential(
            qnn.QuantIdentity(bit_width=bit_width, act_quant=ACT_QUANTIZER, return_quant_tensor=True),
            make_conv(bit_width,3,32,3),
            qnn.QuantReLU(return_quant_tensor=True,output_quant=ACT_QUANTIZER),
            qnn.QuantMaxPool2d(kernel_size=2, stride=2,return_quant_tensor=True),
            make_conv(bit_width,32,64,3),
            qnn.QuantReLU(return_quant_tensor=True,output_quant=ACT_QUANTIZER),
            qnn.QuantMaxPool2d(kernel_size=2, stride=2,return_quant_tensor=True))
        self.classifier = nn.Sequential(
            qnn.QuantLinear(64*6*6, 120, bias=True, weight_bit_width=bit_width,return_quant_tensor=True, bias_quant=BIAS_QUANTIZER, weight_quant=WEIGHT_QUANTIZER,output_quant=ACT_QUANTIZER),
            qnn.QuantReLU(return_quant_tensor=True,output_quant=ACT_QUANTIZER),
            qnn.QuantLinear(120, 84, bias=True, weight_bit_width=bit_width,return_quant_tensor=True, bias_quant=BIAS_QUANTIZER, weight_quant=WEIGHT_QUANTIZER,output_quant=ACT_QUANTIZER),
            qnn.QuantReLU(return_quant_tensor=True,output_quant=ACT_QUANTIZER),
            qnn.QuantLinear(84, 10, bias=False, weight_bit_width=bit_width,return_quant_tensor=False))
        self.features[1].cache_inference_quant_bias=True
        self.features[4].cache_inference_quant_bias=True
        self.classifier[0].cache_inference_quant_bias = True
        self.classifier[2].cache_inference_quant_bias = True
        self.classifier[4].cache_inference_quant_bias = True

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.shape[0], -1)
        out = self.classifier(out)
        return out

def make_conv(bit_width,in_channels,out_channels,kernel):
    return qnn.QuantConv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size= kernel,
                            weight_bit_width=bit_width,
                            return_quant_tensor=True,
                            bias_quant=BIAS_QUANTIZER,
                            weight_quant=WEIGHT_QUANTIZER,
                            output_quant=ACT_QUANTIZER)