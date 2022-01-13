from __future__ import annotations
import numpy as np
import abc
from typing import List, Optional, final

class funcMeta(abc.ABCMeta):
    def __repr__(cls):
        return cls.func_name

class Func(metaclass=funcMeta):
    func_name = "Func"

    @final
    def __new__(cls, *args):
        return cls.forward(*args)

    @staticmethod
    @abc.abstractmethod
    def forward(*args):
        pass

    @staticmethod
    @abc.abstractmethod
    def backward(*args):
        pass

    @final
    def __repr__(self):
        return self.func_name

class FuncNil(Func):
    func_name = "Nil"
    @staticmethod
    def forward(*args):
        return args
    
    @staticmethod
    def backward(*args):
        return 1

class FuncMatmul(Func):
    func_name = "Matmul"
    @staticmethod
    def forward(*args):
        return args[0] @ args[1]

    @staticmethod
    def backward(*args):
        return (args[1].T, args[0].T)


def ufunc_to_func(ufunc: np.ufunc):
    # This will redirect numpy ufunc to Func class...
    if ufunc == np.matmul:
        return FuncMatmul
    return ufunc


class CalcGraph:
    def __init__(self, param: List[Optional[CalcGraph]], func: Func, tensor: Tensor):
        self.param: List[Optional[CalcGraph]] = param
        self.func: Func = func
        self.tensor = tensor

    def __repr__(self):
        return f"<{self.func}, {self.param}>"
    
    def __call__(self):
        return self.func(*[x() for x in self.param])

    def backward(self, prop: np.ndarray):
        self.tensor.grad = prop
        backs = self.func.backward(*[x() for x in self.param])
        self.param[0].backward(prop @ backs[0].view(np.ndarray))
        self.param[1].backward(backs[1].view(np.ndarray) @ prop)


class CalcGraphLeaf(CalcGraph):
    def __init__(self, tensor: Tensor):
        super().__init__([None], FuncNil, tensor)

    def __repr__(self):
        return f"Leaf {self.tensor}"

    def __call__(self):
        return np.asarray(self.tensor)

    def backward(self, prop:np.ndarray):
        self.tensor.grad = prop



class Tensor(np.ndarray):
    # TODO: Refer to https://numpy.org/doc/stable/user/basics.subclassing.html for subclassing.
    # Have to implement __array_ufunc__ to not actually calculate but make calc graph
    # It has CalcGraph in it.

    def __new__(cls, array):
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if 'calc_graph' not in dir(self):
            self.calc_graph = CalcGraphLeaf(self)
        self.grad = None

    def __array_wrap__(self, out_arr, context=None):
        res = super().__array_wrap__(out_arr, context)
        param = [x.calc_graph if isinstance(x, Tensor) else x.view(Tensor).calc_graph for x in context[1]]
        func = ufunc_to_func(context[0])
        res.calc_graph = CalcGraph(param, func, res)
        return res

    def __call__(self, obj=None):
        self.__array_finalize__(self.calc_graph())
        return self

    def backward(self):
        self.calc_graph.backward(np.ones_like(self))


if __name__ == "__main__":
    test = Tensor([[1, 2], [3, 5]])
    print("====")
    test2 = np.asarray([[5, 6], [7, 8]])
    print("====")
    testres = test @ test2
    print(f"{testres=}, {type(testres)=}")
    print(f"{testres.calc_graph=}")
    print("========")
    print(testres())
    print(type(testres()))
    print(testres().calc_graph)
    print("==========")
    testres.backward()
    print(test.grad)

    pass
