from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Type
from func import *

if TYPE_CHECKING:
    from tensor import Tensor

class CalcGraph:
    def __init__(self, param: List[Optional[CalcGraph]], func: Type[Func], tensor: Tensor):
        self.param: List[Optional[CalcGraph]] = param
        self.func: Type[Func] = func
        self.tensor = tensor

    def __repr__(self):
        return f"<{self.func}, {self.param}>"
    
    def __call__(self):
        return self.func(*[x() for x in self.param])

    def backward(self, prop: np.ndarray):
        self.tensor.grad += prop
        backs = self.func.backward(prop, *[x() for x in self.param])
        for next_param, next_back in zip(self.param, backs):
            next_param.backward(next_back.view(np.ndarray))
    
    def zero_grad(self):
        self.tensor.grad = np.zeros_like(self.tensor)
        for next_param in self.param:
            next_param.zero_grad()


class CalcGraphLeaf(CalcGraph):
    def __init__(self, tensor: Tensor):
        super().__init__([None], FuncNil, tensor)

    def __repr__(self):
        return f"Leaf {self.tensor.shape}"

    def __call__(self):
        return np.asarray(self.tensor)

    def backward(self, prop:np.ndarray):
        self.tensor.grad += prop

    def zero_grad(self):
        self.tensor.grad = np.zeros_like(self.tensor)
