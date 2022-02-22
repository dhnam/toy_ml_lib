from abc import ABCMeta, abstractmethod, ABC
from typing import final
import numpy as np
from tensor import Tensor
from model import ParamScanner

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
    def __init__(self, in_features: int, out_features: int, name: str=None, has_bias=True, activation: Activation=None):
        if name is None:
            name = "Linear"
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


class Sigmoid(Activation):
    def apply(self, x:Tensor) -> Tensor:
        return 1. / (1. + np.exp(-x))

class ReLU(Activation):
    def apply(self, x: Tensor) -> Tensor:
        return np.maximum(x, 0)