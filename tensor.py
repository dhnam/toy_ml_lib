import numpy as np
import abc
from typing import List, Optional

class Func(abc.ABC):
    @abc.abstractmethod
    def forward(self, *args):
        pass

    @abc.abstractmethod
    def backward(self, *args):
        pass

def ufunc_to_func(ufunc: np.ufunc):
    # This will redirect numpy ufunc to Func class...
    pass

class CalcGraph:
    def __init__(self, param: List[Optional[CalcGraph]], func: Func):
        self.param: List[Optional[CalcGraph]] = param
        self.func: Func = func


class Tensor(np.ndarray):
    # TODO: Refer to https://numpy.org/doc/stable/user/basics.subclassing.html for subclassing.
    # Have to implement __array_ufunc__ to not actually calculate but make calc graph
    # It has CalcGraph in it.
    pass

if __name__ == "__main__":
    #test = Tensor
    pass
