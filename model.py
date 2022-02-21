from typing import final
import numpy as np
import abc
from tensor import Tensor
from layer import Layer


class Model(abc.ABC):
    def __init__(self):
        self.trainable_param: list[Tensor] | None = None
        pass

    @final
    def __call__(self, *args, **kwargs):
        return self._apply(*args, **kwargs)

    @final
    def _apply(self, *args, **kwargs):
        if self.trainable_param is None:
            self.trainable_param = []
            for next_name in dir(self):
                if next_name == "trainable_param":
                    continue
                if isinstance(next_attr := getattr(self, next_name), Tensor) and next_attr.trainable:
                    self.trainable_param.append(next_attr)
                elif isinstance(next_attr, Layer):
                    self.trainable_param.extend(next_attr.get_trainalbe_param())
                else:
                    try:
                        for next_item in next_attr:
                            if isinstance(next_item, Tensor) and next_item.trainable:
                                self.trainable_param.append(next_item)
                            elif isinstance(next_item, Layer):
                                self.trainable_param.extend(next_item.get_trainalbe_param())
                    except TypeError:
                        pass
        # Check something
        return self.apply(*args, **kwargs)

    @abc.abstractmethod
    def apply(self, *args, **kwargs):
        pass

class SequentialModel(Model):
    def __init__(self, layers: list[Layer]):
        super().__init__()
        self.layers = layers
        last_size = 1
        for layer in layers:
            layer._layer_init(last_size)
            last_size = layer.shape

    def apply(self, x: Tensor) -> Tensor:
        for next_layer in self.layers:
            x = next_layer(x)
        return x