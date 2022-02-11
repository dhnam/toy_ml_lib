from __future__ import annotations
import abc
from logging import warning
from typing import final, Callable, Any
import numpy as np

class FuncFactory:
    IMPL_FUNC = {}
    @staticmethod
    def generate(ufunc: np.ufunc) -> Func:
        if ufunc in FuncFactory.IMPL_FUNC:
            if not issubclass(FuncFactory.IMPL_FUNC[ufunc], FuncClassMaker):
                return FuncFactory.IMPL_FUNC[ufunc]
            else:
                return FuncFactory.IMPL_FUNC[ufunc]()
        warning(ufunc)
        return FuncNil
        

class FuncMeta(abc.ABCMeta):
    def __repr__(cls):
        return cls.func_name

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.forward(*args, **kwargs)


def implements(func: Callable):
    def decorator(decorated: type[Func]):
        FuncFactory.IMPL_FUNC[func] = decorated
    return decorator


class Func(metaclass=FuncMeta):
    func_name = "Func"

    @staticmethod
    @abc.abstractmethod
    def forward(*args: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def backward(propa: np.ndarray, *args: np.ndarray, **kwargs) -> tuple[np.ndarray]:
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
        return (np.where(np.isnan(propa * np.exp(args[0])), 0, propa * np.exp(args[0])),)

@implements(np.exp2)
class FuncExp2(Func):
    func_name = "Exp2"
    @staticmethod
    def forward(*args: np.ndarray) -> np.ndarray:
        return np.exp2(args[0])
    
    @staticmethod
    def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
        return (propa * np.exp(args[0]) * np.log(2))

class FuncClassMakerMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj.__call__(*args, **kwargs)


class FuncClassMaker(metaclass=FuncClassMakerMeta):
    @abc.abstractmethod
    def args_to_class_name(self) -> str:
        pass

    @abc.abstractmethod
    def make_forward(self) -> Callable[[Any], np.ndarray]:
        pass

    @abc.abstractmethod
    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        pass

    @abc.abstractmethod
    def args_to_func_name(self) -> str:
        pass
    
    @final
    def __call__(self, *args, **kwargs):
        func_name = self.args_to_func_name()
        class_name = self.args_to_class_name()
        return type(class_name, (Func, ), {
            'func_name': func_name,
            'forward': self.make_forward(),
            'backward': self.make_backward(),
        })


class BroadcastFuncClassMaker(FuncClassMaker):
    def __init__(self, shape_before: tuple[int, ...], shape_after: tuple[int, ...]):
        self.shape_before = shape_before
        self.shape_after = shape_after

    def args_to_class_name(self) -> str:
        def tuple_to_identifier(tuple_: tuple[int]):
            str_ = str(tuple_)
            str_ = str_.replace(" ", "")
            str_ = str_.replace(",", "_")
            str_ = str_.replace("(", "")
            str_ = str_.replace(")", "")
            return str_
        return f"FuncBroadcast_{tuple_to_identifier(self.shape_before)}__{tuple_to_identifier(self.shape_after)}"

    def make_forward(self) -> Callable[[Any], np.ndarray]:
        @staticmethod
        def forward(*args: np.ndarray) -> np.ndarray:
            assert(len(args) == 1)
            assert(args[0].shape == self.shape_before)
            return np.copy(np.broadcast_to(args[0], self.shape_after))

        return forward

    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        @staticmethod
        def backward(propa: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray]:
            shape_before_adjusted = self.shape_before
            diff = len(self.shape_after) - len(self.shape_before)
            if diff > 0:
                shape_before_adjusted = tuple([1] * diff + list(self.shape_before))
            reduce_dim = []
            for i, next_size in enumerate(shape_before_adjusted):
                if next_size == 1 and self.shape_after[i] != 1:
                    reduce_dim.append(i)
            reduced = np.sum(propa, tuple(reduce_dim), keepdims=True)
            return (np.squeeze(reduced, tuple(range(diff))),)
        
        return backward

    def args_to_func_name(self) -> str:
        return f"Broadcast({self.shape_before} -> {self.shape_after})"

@implements(np.maximum)
class MaximumFuncClassMaker(FuncClassMaker):
    def __init__(self, *args, **kwargs):
        self.pos = None

    def args_to_class_name(self) -> str:
        return "FuncMaximum"

    def make_forward(self) -> Callable[[Any], np.ndarray]:
        @staticmethod
        def forward(x1, x2, out=None, where=True):
            ret = np.maximum(x1, x2, out=out, where=where)
            self.pos = x1 >= x2
            return ret
        return forward

    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        @staticmethod
        def backward(propa: np.ndarray, *args):
            return (np.where(self.pos, propa, 0), np.where(np.invert(self.pos), propa, 0))
        return backward

    def args_to_func_name(self) -> str:
        return "FuncMaximum"


if __name__ == "__main__":
    broadcast_func = BroadcastFuncClassMaker((2, 2), (2, 2, 2))
    a = np.asarray([[2, 2], [2, 2]])
    print(broadcast_func(a))
    print("======")
    grad = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(broadcast_func.backward(grad))

    print(FuncFactory.IMPL_FUNC)
    b = np.asarray([[1, 3], [1, 3]])
    func = FuncFactory.generate(np.maximum)
    print(func(a, b))
    print(func.backward(np.asarray([[1, 2], [3, 4]])))
    