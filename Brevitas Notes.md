1. In 0.6.0 torch.flatten that breaks the propagation of QuantTensor. Try to replace it with x = x.flatten(1) or x = x.view(x.shape[0], -1) 
2. Any time you use a bias quantizer that depends on an external scale factor (like IntBias or Int16tBias) you need to pass as QuantTensor as input to that layer 
3. The right type of bias quantization (internal scale vs external scale) depends on how the target platform works. FINN, PyTorch, standard ONNX use externally defined scale factors, DPUs use internally defined scale factors. FINN also works with floating-point biases in most scenarios. 
4. Any value that is passed as keyword argument overwrites what's in the quantizer

* Alessandro, gitter

5. Dev barnch has some extra features -torch.cat()- to install: 
    pip3 install -e git://github.com/Xilinx/brevitas.git@dev#egg=brevitas
