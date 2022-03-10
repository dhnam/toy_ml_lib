from abc import ABCMeta, abstractmethod, ABC
from typing import final
import numpy as np
from tensor import Tensor
from model import ParamScanner
from array_func import Convolution, ConvBiasAddition

class Layer(ABC, ParamScanner):
    @final
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.apply(*args, **kwargs)

    @final
    def get_trainalbe_param(self) -> list[Tensor]:
        return self.scan_param()


    @abstractmethod
    def apply(self, *args, **kwargs) -> Tensor:
        pass

class Activation(Layer):
    pass


class LinearLayer(Layer):
    def __init__(self, in_features: int, out_features: int, name: str="Linear", has_bias=True, activation: Activation=None):
        self.shape = (in_features, out_features)
        self.name = name
        self.has_bias = has_bias
        self.activation = activation
        self.weight = Tensor(np.random.standard_normal(self.shape), self.name+"_weight", trainable=True)
        if self.has_bias:
            self.bias = Tensor(np.random.standard_normal(self.shape[-1]), self.name+"_bias", trainable=True)

    def apply(self, x:Tensor) -> Tensor:
        calc: Tensor = x @ self.weight
        calc.name = self.name + "_apply_weight"
        if self.has_bias:
            calc = calc + self.bias
            calc.name = self.name + "_apply_bias"
        # calc.name = self.name + "_out"
        if self.activation is not None:
            calc = self.activation()(calc)
            calc.name = self.name + "_activation"
        return calc

class ConvolutionLayer(Layer):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, int], name:str="Conv", has_bias: bool=True):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.name = name
        self.kernel_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.kernel = Tensor(np.random.standard_normal(self.kernel_shape), self.name+"_kernel", trainable=True)
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Tensor(np.random.standard_normal(out_channels), self.name+"_bias", trainable=True)
    
    def apply(self, x:Tensor) -> Tensor:
        # conv = Convolution(x, self.kernel)
        calc: Tensor = x.call_manual_func(Convolution, x, self.kernel)
        calc.name = self.name + "_apply_conv"
        if self.has_bias:
            calc = calc.call_manual_func(ConvBiasAddition, calc, self.bias)
            calc.name = self.name + "_apply_bias"
        return calc

class FlattenLayer(Layer):
    def __init__(self, name:str="Flatten"):
        self.name = name

    def apply(self, x:Tensor) -> Tensor:
        return np.ravel(x)
        

class Sigmoid(Activation):
    def apply(self, x:Tensor) -> Tensor:
        return 1. / (1. + np.exp(-x))

class ReLU(Activation):
    def apply(self, x: Tensor) -> Tensor:
        return np.maximum(x, 0)