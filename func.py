from __future__ import annotations
import abc
from logging import warning
from typing import final, Callable
import numpy as np

class FuncFactory:
    IMPL_FUNC = {}
    @staticmethod
    def generate(ufunc: np.ufunc) -> Func:
        if ufunc in FuncFactory.IMPL_FUNC:
            return FuncFactory.IMPL_FUNC[ufunc]
        warning(ufunc)
        return FuncNil
        

class FuncMeta(abc.ABCMeta):
    def __repr__(cls):
        return cls.func_name


def implements(func: Callable):
    def decorator(decorated: type[Func]):
        FuncFactory.IMPL_FUNC[func] = decorated
    return decorator


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


@implements(np.matmul)
class FuncMatmul(Func):
    func_name = "Matmul"
    @staticmethod
    def forward(*args: np.ndarray):
        return args[0] @ args[1]

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray):
        return (propa @ args[1].T, args[0].T @ propa)

@implements(np.add)
class FuncAdd(Func):
    func_name = "Add"
    @staticmethod
    def forward(*args: np.ndarray):
        return args[0] + args[1]

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray):
        assert(args[0].shape == args[1].shape == propa.shape)
        return (propa, propa)

@implements(np.subtract)
class FuncSubtract(Func):
    func_name = "Substract"
    @staticmethod
    def forward(*args: np.ndarray):
        return args[0] - args[1]
    
    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray):
        assert(args[0].shape == args[1].shape == propa.shape)
        return (propa, -propa)


@implements(np.multiply)
class FuncMultiply(Func):
    func_name = "Multiply"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return args[0] * args[1]

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        assert(args[0].shape == args[1].shape == propa.shape)
        return (propa * args[1], propa * args[0])

@implements(np.true_divide)
class FuncTrueDivide(Func):
    func_name = "TrueDivide"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return args[0] / args[1]

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        assert(args[0].shape == args[1].shape == propa.shape)
        return (propa / args[1], -(propa * args[0] / np.square(args[1])))

class FuncTranspose(Func):
    func_name = "Transpose"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return args[0].T

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa.T,)

@implements(np.negative)
class FuncNegative(Func):
    func_name = "Negative"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return -args[0]

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (-propa,)

@implements(np.square)
class FuncSquare(Func):
    func_name = "Square"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.square(args[0])

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (2 * args[0] * propa,)


@implements(np.log)
class FuncLog(Func):
    func_name = "Log"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.log(args[0])

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * (1 / args[0]), )

@implements(np.log2)
class FuncLog2(Func):
    func_name = "Log2"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.log2(args[0])
    
    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * (1 / (args[0] * np.log(2))))

@implements(np.log10)
class FuncLog10(Func):
    func_name = "Log10"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.log10(args[0])
    
    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * (1 / (args[0] * np.log(10))))

@implements(np.sin)
class FuncSin(Func):
    func_name = "Sin"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.sin(args[0])

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * np.cos(args[0]),)

@implements(np.cos)
class FuncCos(Func):
    func_name = "Cos"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.cos(args[0])

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (-propa * np.sin(args[0]),)

@implements(np.tan)
class FuncTan(Func):
    func_name = "Tan"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.tan(args[0])

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * (1 / (np.square(np.cos(args[0])))),)

@implements(np.sinh)
class FuncSinh(Func):
    func_name = "Sinh"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.sinh(args[0])

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * np.cosh(args[0]),)

@implements(np.cosh)
class FuncCosh(Func):
    func_name = "Cosh"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.cosh(args[0])

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * np.sinh(args[0]),)

@implements(np.tanh)
class FuncTanh(Func):
    func_name = "Tanh"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.tanh(args[0])

    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (-propa * (1 / (np.square(np.cosh(args[0])))),)

@implements(np.exp)
class FuncExp(Func):
    func_name = "Exp"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.exp(args[0])
    
    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * np.exp(args[0]))

@implements(np.exp2)
class FuncExp2(Func):
    func_name = "Exp2"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.exp2(args[0])
    
    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * np.exp(args[0]) * np.log(2))

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
        reduced = np.sum(propa, tuple(reduce_dim), keepdims=True)
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

    print(FuncFactory.IMPL_FUNC)
    