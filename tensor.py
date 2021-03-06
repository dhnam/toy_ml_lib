from __future__ import annotations
from typing import Callable
import numpy as np
from func import *
from calc_graph import *
from array_func import ArrFuncFactory, Convolution

tensorcount = 0
class Tensor(np.ndarray):
    # TODO: Refer to https://numpy.org/doc/stable/user/basics.subclassing.html for subclassing.
    # Have to implement __array_ufunc__ to not actually calculate but make calc graph
    # It has CalcGraph in it.
    def __new__(cls, array, name=None, trainable=False):
        global tensorcount
        obj = np.asarray(array).view(cls)
        if name is None:
            obj.name = "tensor" + str(tensorcount)
        else:
            obj.name = name
        
        tensorcount += 1
        obj.trainable = trainable
        obj.graph_included = False
        if trainable:
            obj.graph_included = True
        return obj

    def __array_finalize__(self, obj):
        global tensorcount
        if 'calc_graph' not in dir(self):
            self.calc_graph = CalcGraphLeaf(self)
        self.grad = np.zeros(self.shape, dtype=np.float64)
        self.name = "tensor" + str(tensorcount)
        self.trainable = False
        self.graph_included = False
        if type(obj) is Tensor:
            self.name = obj.name + "#"
            self.calc_graph = obj.calc_graph
            # self.trainable = obj.trainable
            self.graph_included = obj.graph_included
        else:
            tensorcount += 1


    def broadcast_func(self, func: Callable, operand: list[np.ndarray]) -> list[np.ndarray]:
        #print(func)
        if not isinstance(func, np.ufunc) or func.signature is None:
            return_arr: list[Tensor] = []
            #print([operand_.shape for operand_ in operand if isinstance(operand_, np.ndarray)])
            broadcast = np.broadcast(*operand)
            #print(broadcast.shape)
            #print(operand)
            for next_op in operand:
                if not isinstance(next_op, Tensor):
                        next_op = np.asarray(next_op).view(Tensor)
                if next_op.shape != broadcast.shape:
                    broadcasted_tensor: Tensor = np.copy(np.broadcast_to(next_op, broadcast.shape)).view(Tensor)
                    broadcasted_tensor.graph_included = next_op.graph_included
                    broadcasted_tensor.calc_graph = CalcGraphFactory.make_graph([next_op.calc_graph], BroadcastFuncClassMaker(next_op.shape, broadcast.shape), broadcasted_tensor)
                    return_arr.append(broadcasted_tensor)
                else:
                    return_arr.append(next_op)
            #print([x.shape for x in return_arr])
            return return_arr
        else:
            return operand

    @property
    def T(self):
        transposed_tensor: Tensor = np.copy(self.transpose(), subok=True)
        transposed_tensor.calc_graph = CalcGraphFactory.make_graph([self.calc_graph], FuncTranspose, transposed_tensor)
        return transposed_tensor

    def __array_wrap__(self, out_arr, context: tuple[Callable, list[np.ndarray], int] | None=None):
        broadcasted = self.broadcast_func(context[0], context[1])

        param = []
        is_graph_included = False
        for next_array in broadcasted:
            # if not isinstance(next_array, Tensor):
            #     next_tensor = next_array.view(Tensor)
            # else:
            #     next_tensor = next_array
            if isinstance(next_array, Tensor):
            #     next_tensor.calc_graph = next_array.calc_graph
                if next_array.graph_included:
                    is_graph_included = True
            #         pass

        res: Tensor = super().__array_wrap__(out_arr, context)
        res.graph_included = is_graph_included
        param = [x.calc_graph if isinstance(x, Tensor) else np.asarray(x).view(Tensor).calc_graph for x in broadcasted]
        # take care of numpy broadcast?
        func = FuncFactory.generate(context[0])
        res.calc_graph = CalcGraphFactory.make_graph(param, func, res)
        return res

    def __array_function__(self, func, types, args, kwargs):
        arr_func = ArrFuncFactory.generate(func, *args, **kwargs)
        if arr_func is None:
            return super().__array_function__(func, types, args, kwargs)

        applied: Tensor = arr_func(*args, **kwargs).view(Tensor)
        is_graph_included = False
        if type(args[0]) in (list, tuple):
            param = [x.calc_graph if isinstance(x, Tensor) else np.asarray(x).view(Tensor).calc_graph for x in args[0]]
            for next_array in args[0]:
                if isinstance(next_array, Tensor):
                    if next_array.graph_included:
                        is_graph_included = True
        else:
            param = []
            for next_arg in args:
                if isinstance(next_arg, Tensor):
                    next_param = next_arg.calc_graph
                    if next_arg.graph_included:
                        is_graph_included = True
                else:
                    next_param = np.asarray(next_arg).view(Tensor).calc_graph
                param.append(next_param)
        applied.graph_included = is_graph_included
        applied.calc_graph = CalcGraphFactory.make_graph(param, arr_func, applied, kwargs)
        return applied

    def call_manual_func(self, func, *args, **kwargs):
        return self.__array_function__(func, [], args, kwargs)

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

    test7: Tensor = np.exp2(test)
    test7.name = "test7"
    print(test7)
    print(test7.calc_graph)

    test8: Tensor = np.concatenate([test, test2], axis=1).view(Tensor)
    test8.name = "Test8"
    print(test8)
    print(test8.calc_graph)
    test8.backward()
    print(test.grad)

    test_squeeze: Tensor = Tensor([[1, 2, 3, 4]], name="Test_squeeze")
    test9: Tensor = np.squeeze(test_squeeze)
    test9.name = "Test9"
    print(test9)
    print(test9.calc_graph)
    test9.backward()
    print(test_squeeze.grad)

    test.zero_grad()

    print(test)
    test_avg: Tensor = np.average(test)
    test_avg.name = "avg"
    print("======")
    print(test_avg)
    print(test_avg.calc_graph)
    test_avg.backward()
    print(test.grad)
    test_avg.zero_grad()

    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    test_sigmoid = Tensor([[-1, 0, 1, 1000000]], name="test_sigmoid")
    res_sigmoid = sigmoid(test_sigmoid)
    print(res_sigmoid)
    res_sigmoid.backward()
    print(test_sigmoid.grad)
    res_sigmoid.zero_grad()

    test_a = np.average(np.square(test - test2))
    print(test_a)
    test_a.backward()
    print(test.grad)
    test_a.zero_grad()
    
    test_b = np.average(np.square(test2 - test))
    print(test_b)
    test_b.backward()
    print(test.grad)
    test_b.zero_grad()

