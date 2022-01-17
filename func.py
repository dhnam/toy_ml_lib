import abc
from typing import final
import numpy as np

class FuncMeta(abc.ABCMeta):
    def __repr__(cls):
        return cls.func_name


class Func(metaclass=FuncMeta):
    func_name = "Func"

    @final
    def __new__(cls, *args):
        return cls.forward(*args)

    @staticmethod
    @abc.abstractmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        pass

    @final
    def __repr__(self):
        return self.func_name


class FuncNil(Func):
    func_name = "Nil"
    @staticmethod
    def forward(*args: np.ndarray):
        return args
    
    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray):
        return propa


class FuncMatmul(Func):
    func_name = "Matmul"
    @staticmethod
    def forward(*args: np.ndarray):
        return args[0] @ args[1]

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray):
        return (propa @ args[1].T, args[0].T @ propa)


class FuncAdd(Func):
    func_name = "Add"
    @staticmethod
    def forward(*args: np.ndarray):
        return args[0] + args[1]

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray):
        print(f"{args[0].shape=}, {args[1].shape}, {propa.shape=}")
        assert(args[0].shape == args[1].shape == propa.shape)
        return (propa, propa)


class FuncSubtract(Func):
    func_name = "Substract"
    @staticmethod
    def forward(*args: np.ndarray):
        return args[0] - args[1]
    
    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray):
        assert(args[0].shape == args[1].shape == propa.shape)
        return (propa, -propa)


class FuncFactory:
    @staticmethod
    def generate(ufunc: np.ufunc) -> Func:
        match ufunc:
            case np.matmul:
                return FuncMatmul
            case np.add:
                return FuncAdd
            case np.subtract:
                return FuncSubtract
            case _:
                return FuncNil

def broadcast_func_class_maker(shape_before: tuple[int, ...], shape_after: tuple[int, ...]):
    func_name = f"Broadcast({shape_before} -> {shape_after})"
    def tuple_to_identifier(tuple_: tuple[int]):
        str_ = str(tuple_)
        str_ = str_.replace(" ", "")
        str_ = str_.replace(",", "_")
        str_ = str_.replace("(", "")
        str_ = str_.replace(")", "")
        return str_
    class_name = f"FuncBroadcast_{tuple_to_identifier(shape_before)}__{tuple_to_identifier(shape_after)}"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        assert(len(args) == 1)
        assert(args[0].shape == shape_before)
        return np.copy(np.broadcast_to(args[0], shape_after))

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        shape_before_adjusted = shape_before
        diff = len(shape_after) - len(shape_before)
        if diff > 0:
            shape_before_adjusted = tuple([1] * diff + list(shape_before))
        reduce_dim = []
        for i, next_size in enumerate(shape_before_adjusted):
            if next_size == 1 and shape_after[i] != 1:
                reduce_dim.append(i)
        reduced = np.mean(propa, tuple(reduce_dim), keepdims=True)
        return (np.squeeze(reduced, tuple(range(diff))),)

    return type(class_name, (Func, ), {
        'func_name': func_name,
        'forward': forward,
        'backward': backward,
    })

if __name__ == "__main__":
    broadcast_func = broadcast_func_class_maker((2, 2), (2, 2, 2))
    a = np.asarray([[2, 2], [2, 2]])
    print(broadcast_func(a))
    print("======")
    grad = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(broadcast_func.backward(grad))
    