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
    def forward(*args: np.ndarray):
        pass

    @staticmethod
    @abc.abstractmethod
    def backward(propa: np.ndarray, *args: np.ndarray):
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