from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
from func import *
from calc_graph import *

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

    def __array_wrap__(self, out_arr, context: Tuple[Callable, List[np.ndarray], int] | None=None):
        print(f"{self=}, {out_arr=}, {context=}")
        broadcasted = context[1]
        if context[0].signature is None:
            broadcasted: List[np.ndarray] = np.copy(np.broadcast_arrays(context[1]))

        param = []
        for i, next_array in enumerate(context[1]):
            next_tensor = next_array.view(Tensor)
            if isinstance(context[1][i], Tensor):
                next_tensor.calc_graph = context[1][i].calc_graph

        res: Tensor = super().__array_wrap__(out_arr, context)
        param = [x.calc_graph if isinstance(x, Tensor) else np.asarray(x).view(Tensor).calc_graph for x in context[1]]
        # take care of numpy broadcast?
        func = FuncFactory.generate(context[0])
        res.calc_graph = CalcGraph(param, func, res)
        return res

    def __call__(self, obj=None):
        self.__array_finalize__(self.calc_graph())
        return self

    def backward(self):
        self.calc_graph.backward(np.ones_like(self))

    def zero_grad(self):
        self.calc_graph.zero_grad()


if __name__ == "__main__":
    test = Tensor([[1, 2], [3, 4]])
    print("====")
    test2 = np.asarray([[5, 6], [7, 8]])
    print("====")
    testres: Tensor = test @ test2 @ np.asarray([[9], [10]])
    print(testres)
    print(testres.calc_graph)
    testres.backward()
    print(test.grad)

    testres.zero_grad()

    test3 = test + 3
    print(test3)
    print(test3.calc_graph)
    test3.backward()
    print(test.grad)
    pass
