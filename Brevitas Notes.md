1. In 0.6.0 torch.flatten that breaks the propagation of QuantTensor. Try to replace it with x = x.flatten(1) or x = x.view(x.shape[0], -1) 
2. Any time you use a bias quantizer that depends on an external scale factor (like IntBias or Int16tBias) you need to pass as QuantTensor as input to that layer 
3. The right type of bias quantization (internal scale vs external scale) depends on how the target platform works. FINN, PyTorch, standard ONNX use externally defined scale factors, DPUs use internally defined scale factors. FINN also works with floating-point biases in most scenarios. 
4. Any value that is passed as keyword argument overwrites what's in the quantizer

* Alessandro, gitter

5. `Int8Bias` means you have that `bias_bit_width()` is set at 8, while `quant_bias_scale() = quant_weight_scale() * quant_input_scale()`.

6. If you used the `Int8BiasPerTensorFixedPoint` quantizer you would have had that `quant_bias_scale()` was a standalone value, independent of input and weight.

7. Quantizing Bias to 8 bit is not recommended, might force loss to nan 

## Notes regarding scale factor
1. The type of scale factor depends on the quantizer. `Int8WeightPerTensorFloat` results in a floating point scale factor, which also does not allow us to have a specific integer vs fractional bits for weight & bias. On the other side `Int8WeightPerTensorFixedPoint` would make the scale factor a power of two number -PoT-. With this quantizer we can extract the number of fractional bits by taking `torch.log2(QuantConv.quant_weight_scale())`.[source](https://github.com/Xilinx/brevitas/issues/271#issuecomment-800473329)
2. Brevitas's quantizers have an option called `SCALING_PER_OUTPUT_CHANNEL`. If this option is set to True each channel would have it's own scale factor for weight, bias or activation based on the quantizer. This way each channel would have a different fractional number of bits, which is harder to implement on FPGA.
3. Relu layers scale factor can be restricted to power of two values by setting `restrict_scaling_type=RestrictValueType.POWER_OF_TWO`

## Notes regarding sharing quantizers
Recently brevitas started supporting sharing of quantizer objects, not the definition but the object it self. For example if a weight quantizer is shared among two Conv layers it's scale factor will be adjusted based on the weights of both layers not just one of them. This becomes useful if want to use `torch.cat` on quant tensors, for example in Unet.[source](https://github.com/Xilinx/brevitas/blob/master/notebooks/Brevitas_TVMCon2021.ipynb) at In[7].

## Notes regarding coping layers
To probably copy a layer from a model to a model use `layer.load_state_dict(other_layer.state_dict())`. Using `copy.deepcopy` would result in a drop in accuracy. This is useful when doing transfer learning or BatchNorm fusion.

## Relation between `int_weight` & `quant_weight`
Mathematically `int_weight=quant_weight/quant_weight_scale_factor`. To correctly convert `int_weight` to a hex format that `ap_fixed` would recognize correctly, we need to first convert it to binary 2'compliment. Note that python's `bin()` function does not return 2'compliment. For example, using a network trained with `Int8WeightPerTensorFixedPoint` & if we weight int bits = 2, `quant_weight=0.0937500000`, `quant_weight_scale=0.0156250000` we will get `int_weight=6`. To convert the int_weight probably we need to pad it correctly so `0110` becomes `00000110` which converts to 0x06. If we put the binary point in the correct place we get `00.000110` which correctly represents our quant_weight. Lets look at a negative quant_weight from the same network. `quant_weight=-0.0312500000; int_weight=-2` -2 gets converted to 11111110 or 0xFE which is our quant_weight in 2's complement.