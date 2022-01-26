from __future__ import annotations
from typing import Callable, final, Any, TYPE_CHECKING
from func import FuncMeta
import abc
import numpy as np
from itertools import accumulate

if TYPE_CHECKING:
    from tensor import Tensor

class ArrFuncFactory:
    IMPL_FUNC: dict[Callable, Callable[[list[np.ndarray], Any], type[ArrayFunc]]] = {}
    @staticmethod
    def generate(func: Callable, *args, **kwargs) -> ArrayFunc | None:
        if func in ArrFuncFactory.IMPL_FUNC:
            return ArrFuncFactory.IMPL_FUNC[func](*args, **kwargs)
        return None


def implements(func: Callable):
    def decorator(decorated: type[ArrayFunc]):
        ArrFuncFactory.IMPL_FUNC[func] = decorated
    return decorator

class ArrayFunc(metaclass=FuncMeta):
    func_name = "ArrFunc"

    @final
    def __new__(cls, *args, **kwargs):
        return cls.forward(*args, **kwargs)

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
    pass

@implements(np.concatenate)
def concat_func_class_maker(*arrs: list[np.ndarray | Tensor], axis=0, **kwargs):
    concat_len_list = list(accumulate([x.shape[axis] for x in arrs[0]]))[:-1]
    func_name = f"ArrFuncConcat (axis {axis})"
    def concat_len_to_identifier(lst: list[int]):
        identifier = str(lst)
        identifier = identifier.replace("[", "")
        identifier = identifier.replace(" ", "_")
        identifier = identifier.replace("]", "")
        return identifier
    class_name = f"FuncBroadcast_axis{axis}_{concat_len_to_identifier(concat_len_list)}"
    @staticmethod
    def forward(*arrs: list[np.ndarray | Tensor] | np.ndarray | Tensor, axis=0, **kwargs) -> np.ndarray:
        if len(arrs) != 1:
            arrs = [arrs]
        assert(all(len(x) >= axis for x in arrs[0]))
        lst = [x.view(np.ndarray) for x in arrs[0]]
        return np.concatenate(lst, axis=axis, **kwargs)

    @staticmethod
    def backward(propa: np.ndarray, *args, axis=0, **kwargs) -> tuple[np.ndarray]:
        print(np.split(propa, concat_len_list, axis=axis))
        return tuple(np.split(propa, concat_len_list, axis=axis))

    return type(class_name, (ArrayFunc, ), {
        'func_name': func_name,
        'forward': forward,
        'backward': backward,
    })


