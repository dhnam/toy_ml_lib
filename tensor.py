from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
from func import *
from calc_graph import *

tensorcount = 0
class Tensor(np.ndarray):
    # TODO: Refer to https://numpy.org/doc/stable/user/basics.subclassing.html for subclassing.
    # Have to implement __array_ufunc__ to not actually calculate but make calc graph
    # It has CalcGraph in it.
    def __new__(cls, array, name=None):
        global tensorcount
        obj = np.asarray(array).view(cls)
        if name is None:
            obj.name = "tensor" + str(tensorcount)
        else:
            obj.name = name
        
        tensorcount += 1
        return obj

    def __array_finalize__(self, obj):
        global tensorcount
        if 'calc_graph' not in dir(self):
            self.calc_graph = CalcGraphLeaf(self)
        self.grad = np.zeros(self.shape, dtype=np.float64)
        self.name = "tensor" + str(tensorcount)
        tensorcount += 1


    def broadcast_func(self, func: Callable, operand: list[np.ndarray]) -> list[np.ndarray]:
        if not isinstance(func, np.ufunc) or func.signature is None:
            return_arr: list[Tensor] = []
            broadcast = np.broadcast(*operand)
            for next_op in operand:
                if not isinstance(next_op, Tensor):
                        next_op = np.asarray(next_op).view(Tensor)
                if next_op.shape != broadcast.shape:
                    broadcasted_tensor: Tensor = np.copy(np.broadcast_to(next_op, broadcast.shape)).view(Tensor)
                    broadcasted_tensor.calc_graph = CalcGraph([next_op.calc_graph], broadcast_func_class_maker(next_op.shape, broadcast.shape), broadcasted_tensor)
                    return_arr.append(broadcasted_tensor)
                else:
                    return_arr.append(next_op)
            return return_arr
        else:
            return operand

    @property
    def T(self):
        transposed_tensor: Tensor = np.copy(self.transpose(), subok=True)
        transposed_tensor.calc_graph = CalcGraph([self.calc_graph], FuncTranspose, transposed_tensor)
        return transposed_tensor

    def __array_wrap__(self, out_arr, context: Tuple[Callable, list[np.ndarray], int] | None=None):
        broadcasted = self.broadcast_func(context[0], context[1])

        param = []
        for i, next_array in enumerate(broadcasted):
            next_tensor = next_array.view(Tensor)
            if isinstance(broadcasted[i], Tensor):
                next_tensor.calc_graph = broadcasted[i].calc_graph

        res: Tensor = super().__array_wrap__(out_arr, context)
        param = [x.calc_graph if isinstance(x, Tensor) else np.asarray(x).view(Tensor).calc_graph for x in broadcasted]
        # take care of numpy broadcast?
        func = FuncFactory.generate(context[0])
        res.calc_graph = CalcGraph(param, func, res)
        return res

    def __call__(self, obj=None):
        self.__array_finalize__(self.calc_graph())
        return self

    def backward(self):
        self.calc_graph.backward(np.ones_like(self, dtype=np.float64))

    def zero_grad(self):
        self.calc_graph.zero_grad()


if __name__ == "__main__":
    test = Tensor([[1, 2], [3, 4]], name="test")
    print("====")
    test2 = np.asarray([[5, 6], [7, 8]])
    print("====")
    testres: Tensor = test @ test2 @ np.asarray([[9], [10]]) + 2
    testres.name = "testres"
    print(testres)
    print(testres.calc_graph)
    testres.backward()
    print(test.grad)

    testres.zero_grad()

    test3: Tensor = test + 3
    test3.name = "test3"
    print("test 3")
    print(test3)
    print(test3.calc_graph)
    test3.backward()
    print(test.grad)

    test3.zero_grad()

    test4: Tensor = test @ test + test2 + test
    test4.name = "test4"
    print(test4)
    print(test4.calc_graph)
    test4.backward()
    print(test.grad)

    test4.zero_grad()

    print("=====")
    test5: Tensor = test + test2
    test5.name = "test5"
    test6 = test5.T
    test6.name = "test6"
    print(test6)
    print(test6.calc_graph)
    test6.backward()
    print(test.grad)
    
    test6.zero_grad()

    x: Tensor = Tensor([1, 2, 3])
    x.name = "x"
    y: Tensor = (2*x - 3)/(3*x + 2)
    y.name = "y"
    print(y)
    print(y.calc_graph)
    y.backward()
    print(x.grad)