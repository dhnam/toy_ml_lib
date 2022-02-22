from __future__ import annotations
from typing import final, TYPE_CHECKING
import numpy as np
import abc
from tensor import Tensor
if TYPE_CHECKING:
    from layer import Layer

class ParamScanner:
    def scan_param(self) -> list[Tensor]:
        trainable_param = []
        for next_name in dir(self):
            next_attr = getattr(self, next_name)
            trainable_param.extend(self._scan_attr_param(next_attr))

        return trainable_param

    def _scan_attr_param(self, attr) -> list:
        if isinstance(attr, Tensor) and attr.trainable:
            return [attr]
        elif isinstance(attr, ParamScanner):
            return attr.scan_param()
        else:
            params = []
            try:
                for next_item in attr:
                    if next_item is attr:
                       continue
                    params.extend(self._scan_attr_param(next_item))
            except TypeError:
                pass
            return params
        
class Model(abc.ABC, ParamScanner):
    def __init__(self):
        self.trainable_param: list[Tensor] | None = None
        pass

    @final
    def __call__(self, *args, **kwargs):
        return self._apply(*args, **kwargs)

    @final
    def _apply(self, *args, **kwargs):
        if self.trainable_param is None:
            self.trainable_param = self.scan_param()
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

    def apply(self, x: Tensor) -> Tensor:
        for next_layer in self.layers:
            x = next_layer(x)
        return x