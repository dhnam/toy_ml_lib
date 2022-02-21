from abc import abstractmethod, ABC
from typing import final
import numpy as np
from tensor import Tensor

class Layer(ABC):
    @final
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.apply(*args, **kwargs)

    @final
    def get_trainalbe_param(self) -> list[Tensor]:
        trainable_param = []
        for next_name in dir(self):
            if next_name == "trainable_param":
                continue
            if isinstance(next_attr := getattr(self, next_name), Tensor) and next_attr.trainable:
                trainable_param.append(next_attr)
            elif isinstance(next_attr, Layer):
                trainable_param.extend(next_attr.get_trainalbe_param())
            else:
                try:
                    for next_item in next_attr:
                        if isinstance(next_item, Tensor) and next_item.trainable:
                            trainable_param.append(next_item)
                        elif isinstance(next_item, Layer):
                            trainable_param.extend(next_item.get_trainalbe_param())
                except TypeError:
                    pass

        return trainable_param


    @abstractmethod
    def apply(self, *args, **kwargs) -> Tensor:
        pass


class Activation(Layer):
    pass


class LinearLayer(Layer):
    def __init__(self, shape: int, name: str=None, has_bias=True, activation: Activation=None):
        self.shape = shape
        self.name = name
        self.has_bias = has_bias
        self.activation = activation

    def _layer_init(self, last_shape:int):
        self.layer = Tensor(np.random.standard_normal((last_shape, self.shape)), self.name, trainable=True)
        if self.has_bias:
            self.bias = Tensor(np.random.standard_normal(self.shape), self.name+"_bias", trainable=True)

    def apply(self, x:Tensor) -> Tensor:
        calc = x @ self.layer
        if self.has_bias:
            calc = calc + self.bias
        if self.activation is not None:
            calc = self.activation(calc)
        return calc


class Sigmoid(Activation):
    def apply(self, x:Tensor) -> Tensor:
        return 1. / (1. + np.exp(-x))

class ReLU(Activation):
    def apply(self, x: Tensor) -> Tensor:
        return np.maximum(x, 0)