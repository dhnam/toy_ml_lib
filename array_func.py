from __future__ import annotations
from typing import Callable,  Any, TYPE_CHECKING
from func import Func, FuncClassMaker
import numpy as np
from itertools import accumulate

if TYPE_CHECKING:
    from tensor import Tensor

class ArrFuncFactory:
    IMPL_FUNC: dict[Callable, Callable[[list[np.ndarray], Any], type[Func]]] = {}
    @staticmethod
    def generate(func: Callable, *args, **kwargs) -> Func | None:
        if func in ArrFuncFactory.IMPL_FUNC:
            return ArrFuncFactory.IMPL_FUNC[func](*args, **kwargs)
        return None


def implements(func: Callable):
    def decorator(decorated: type[Func]):
        ArrFuncFactory.IMPL_FUNC[func] = decorated
    return decorator


@implements(np.concatenate)
class ConcatFuncClassMaker(FuncClassMaker):
    def __init__(self, *arrs: list[np.ndarray | Tensor], axis=0, **kwargs):
        self.arrs = arrs
        self.axis = axis
        self.kwargs = kwargs
        self.concat_len_list = list(accumulate([x.shape[axis] for x in arrs[0]]))[:-1]

    def args_to_class_name(self) -> str:
        def concat_len_to_identifier(lst: list[int]):
            identifier = str(lst)
            identifier = identifier.replace("[", "")
            identifier = identifier.replace(" ", "_")
            identifier = identifier.replace("]", "")
            return identifier
        return f"FuncConcat_axis{self.axis}_{concat_len_to_identifier(self.concat_len_list)}"

    def args_to_func_name(self) -> str:
        return f"ArrFuncConcat (axis {self.axis})"

    def make_forward(self) -> Callable[[Any], np.ndarray]:
        @staticmethod
        def forward(*arrs: list[np.ndarray | Tensor] | np.ndarray | Tensor, axis=0, **kwargs) -> np.ndarray:
            if len(arrs) != 1:
                arrs = [arrs]
            assert(all(len(x) >= axis for x in arrs[0]))
            lst = [x.view(np.ndarray) for x in arrs[0]]
            return np.concatenate(lst, axis=axis, **kwargs)
        return forward

    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        @staticmethod
        def backward(propa: np.ndarray, *args, axis=0, **kwargs) -> tuple[np.ndarray]:
            return tuple(np.split(propa, self.concat_len_list, axis=axis))
        return backward

@implements(np.squeeze)
class SqueezeFuncClassMaker(FuncClassMaker):
    def __init__(self, arr: np.ndarray | Tensor, *args, axis=None, **kwargs):
        self.arr = arr
        self.args = args
        self.axis = axis
        self.kwargs = kwargs
        self.shape = arr.shape
    
    def args_to_class_name(self) -> str:
        def shape_to_identifier(shape: tuple[int]):
            identifier = str(shape)
            identifier = identifier.replace("(", "")
            identifier = identifier.replace(",", "_")
            identifier = identifier.replace(")", "")
            return identifier
        return f"FuncSqueeze_axis{self.axis}_{shape_to_identifier(self.shape)}"

    def args_to_func_name(self) -> str:
        return f"ArrFuncSqueeze (axis {self.axis})"

    def make_forward(self) -> Callable[[Any], np.ndarray]:
        @staticmethod
        def forward(arr: np.ndarray | Tensor, *args, axis=0, **kwargs) -> np.ndarray:
            return np.squeeze(arr.view(np.ndarray), axis=axis)
        return forward

    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        @staticmethod
        def backward(propa: np.ndarray, *args, **kwargs) -> tuple[np.ndarray]:
            return (np.reshape(propa, self.shape),)
        return backward


@implements(np.average)
class AvgFuncClassMaker(FuncClassMaker):
    def __init__(self, arr: np.ndarray | Tensor, axis=None, weights=None, **kwargs):
        self.arr = arr
        self.axis = axis
        self.weights = weights

    def args_to_class_name(self) -> str:
        return f"FuncAvg_axis{self.axis}_weight_is_none_{self.weights is None}"

    def args_to_func_name(self) -> str:
        return f"ArrFuncAvg (axis {self.axis}, weight_is_none: {self.weights is None})"

    def make_forward(self) -> Callable[[Any], np.ndarray]:
        @staticmethod
        def forward(arr: np.ndarray | Tensor, axis=None, weights=None, **kwargs) -> np.ndarray:
            ret = np.average(arr.view(np.ndarray), axis, weights)
            if not isinstance(ret, np.ndarray):
                ret = np.asarray(ret)
            return ret
        return forward
    
    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        @staticmethod
        def backward(propa: np.ndarray, *args, **kwargs) -> tuple[np.ndarray]:
            if self.weights is None:
                if self.axis is None:
                    return (np.broadcast_to(propa / np.prod(self.arr.shape), self.arr.shape),)
                else:
                    return (np.broadcast_to(np.expand_dims(propa / self.arr[self.axis], self.axis), self.arr.shape),)
            else:
                raise NotImplementedError
        return backward
