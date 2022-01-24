from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Type
from func import *

if TYPE_CHECKING:
    from tensor import Tensor

class CalcGraph:
    def __init__(self, param: list[Optional[CalcGraph]], func: Type[Func], tensor: Tensor):
        self.param: list[Optional[CalcGraph]] = param
        self.func: Type[Func] = func
        self.tensor = tensor
        if param[0] is not None:
            self.value = self.func(*[x.value for x in self.param])

    def __repr__(self):
        return f"<{self.func}, {self.param}>"

    def __str__(self):
        ret_str = self.tensor.name + " " + str(self.value.shape)
        param_strs: list[tuple[str, int]] = []
        for next_param in self.param:
            next_str = str(next_param)
            max_len = 0
            for next_line in next_str.split("\n"):
                if len(next_line) > max_len:
                    max_len = len(next_line)
            param_strs.append((next_str, max_len + 1))
        ret_str += " " * (sum([x[1] for x in param_strs]) - len(ret_str) - len(param_strs) - 1) + "│" + "\n"
        name_str = str(self.func)
        ret_str += name_str
        is_first = True
        for next_str, next_len in param_strs:
            if is_first:
                is_first = False
                ret_str += "─" * (next_len - len(name_str) - 2)
                ret_str += "┬"
            else:
                ret_str += "─" * (next_len - 2)
                ret_str += "┬"
        ret_str = ret_str[:-1]
        ret_str += "┤"
        # ret_str += "─" * (sum([x[1] for x in param_strs]) - len(name_str) - len(param_strs) - 1) + "┤"

        i = 0
        while True:
            ret_str += "\n"
            end_count = len(param_strs)
            for next_str, next_len in param_strs:
                next_splits = next_str.split("\n")
                if len(next_splits) <= i:
                    ret_str += " " * (next_len - 2)
                    ret_str += "│"
                    end_count -= 1
                    continue
                next_line = next_splits[i]
                ret_str += next_line[:-1]
                ret_str += " " * (next_len - len(next_line) - 1)
                ret_str += next_line[-1]
            if end_count == 0:
                break
            i += 1
        ret_str = "\n".join(ret_str.split("\n")[:-1])
        return ret_str.strip()
        
    
    def __call__(self):
        return self.value

    def backward(self, prop: np.ndarray):
        self.tensor.grad += prop
        backs = self.func.backward(prop, *[x.value for x in self.param])
        for next_param, next_back in zip(self.param, backs):
            next_param.backward(next_back.view(np.ndarray))
    
    def zero_grad(self):
        self.tensor.grad = np.zeros_like(self.tensor, dtype=np.float64)
        for next_param in self.param:
            next_param.zero_grad()


class CalcGraphLeaf(CalcGraph):
    def __init__(self, tensor: Tensor):
        super().__init__([None], FuncNil, tensor)
        self.value: np.ndarray = np.copy(tensor)

    def __repr__(self):
        return f"{self.tensor.name} {self.tensor.shape}"

    def __str__(self):
        return self.__repr__() + "│"

    def __call__(self):
        return self.value

    def backward(self, prop:np.ndarray):
        self.tensor.grad += prop

    def zero_grad(self):
        self.tensor.grad = np.zeros_like(self.tensor, dtype=np.float64)
