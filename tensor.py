from __future__ import annotations
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

class FuncLeaf(Func):
    def __init__(self, value):
        self.value = value
    
    def forward(self, *args):
        return self.value

    def backward(self, *args):
        return 1

    def __repr__(self):
        return f"Leaf value = {self.value}"

class FuncMatmul(Func):
    def forward(self, *args):
        return args[0] @ args[1]

    def backward(self, *args):
        #return (args[1], args[0])
        return (args[1].T, args[0].T)

    def __repr__(self):
        return "Matmul"


def ufunc_to_func(ufunc: np.ufunc):
    # This will redirect numpy ufunc to Func class...
    if ufunc == np.matmul:
        return FuncMatmul()
    return ufunc


class CalcGraph:
    def __init__(self, param: List[Optional[CalcGraph]], func: Func):
        self.param: List[Optional[CalcGraph]] = param
        self.func: Func = func

    def __str__(self):
        return f"<{self.func}, {self.param}>"

    def __repr__(self):
        return self.__str__()


class Tensor(np.ndarray):
    # TODO: Refer to https://numpy.org/doc/stable/user/basics.subclassing.html for subclassing.
    # Have to implement __array_ufunc__ to not actually calculate but make calc graph
    # It has CalcGraph in it.

    def __new__(cls, array):
        print("__new__ called for Tensor")
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        print("__array_finalize__")
        print(f"self type: {type(self)}")
        print(f"obj type: {type(obj)}")
        self.calc_graph = CalcGraph([None], FuncLeaf(self))

    def __array_wrap__(self, out_arr, context=None):
        print(f"__array_wrap__ Called\n{self=}\n{out_arr=}\n{context=}")
        res = super().__array_wrap__(out_arr, context)
        res.calc_graph.param = [x.calc_graph if isinstance(x, Tensor) else x.view(Tensor).calc_graph for x in context[1]]
        res.calc_graph.func = ufunc_to_func(context[0])
        return res


if __name__ == "__main__":
    test = Tensor([[1, 2], [3, 5]])
    print("====")
    test2 = np.asarray([[5, 6], [7, 8]])
    print("====")
    testres = test @ test2
    print(f"{testres=}, {type(testres)=}")
    print(f"{testres.calc_graph=}")
    pass
