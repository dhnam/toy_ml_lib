from __future__ import annotations
from typing import Iterator, Optional, TYPE_CHECKING
from func import *

if TYPE_CHECKING:
    from tensor import Tensor

class CalcGraph:
    def __init__(self, param: list[Optional[CalcGraph]], func: type[Func], tensor: Tensor, kwargs=None):
        self.param: list[Optional[CalcGraph]] = param
        self.func: type[Func] = func
        self._tensor = tensor
        self.kwargs = kwargs if kwargs is not None else {}
        if param[0] is not None:
            self.cache = self.func(*[x.cache for x in self.param], **self.kwargs)

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, val: np.ndarray | Tensor):
        temp = self._tensor
        np.copyto(self._tensor, val, casting='safe')
        #self._tensor: Tensor = val
        self._tensor.calc_graph = self
        self._tensor.name = temp.name

    def __iter__(self) -> Iterator[CalcGraph]:
        return CalcGraphIterator(self)

    def __repr__(self):
        return f"<{self.func}, {self.param}>"

    def __str__(self):
        ret_str = self.tensor.name + " " + str(self.cache.shape)
        if self.tensor.trainable:
            ret_str += " <T>"
        param_strs: list[tuple[str, int]] = []
        glob_max_len = len(ret_str) + 1
        name_str = str(self.func)
        glob_max_len = max(glob_max_len, len(name_str) + 1)
        for next_param in self.param:
            next_str = str(next_param)
            max_len = 0
            for next_line in next_str.split("\n"):
                if len(next_line) > max_len:
                    max_len = len(next_line)
            max_len = max(max_len, glob_max_len)
            glob_max_len = 0
            param_strs.append((next_str, max_len + 1))
        ret_str += " " * (sum([x[1] for x in param_strs]) - len(ret_str) - len(param_strs) - 1) + "│" + "\n"
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
        if self.param[0] is not None:
            self.cache = self.func(*[x() for x in self.param], **self.kwargs)
        return self.cache

    
    def __hash__(self):
        return hash(str(self.func))

    def backward(self, prop: np.ndarray):
        self.tensor.grad += prop
        backs = self.func.backward(prop, *[x() for x in self.param], **self.kwargs)
        for next_param, next_back in zip(self.param, backs):
            next_param.backward(next_back.view(np.ndarray))
    
    def zero_grad(self):
        self.tensor.grad = np.zeros_like(self.tensor, dtype=np.float64)
        for next_param in self.param:
            next_param.zero_grad()


class CalcGraphLeaf(CalcGraph):
    def __init__(self, tensor: Tensor):
        super().__init__([None], FuncNil, tensor)
        self.cache: np.ndarray = np.copy(tensor)

    def __repr__(self):
        ret_str = f"{self.tensor.name} {self.tensor.shape}"
        if self.tensor.trainable:
            ret_str += " <T>"
        return ret_str

    def __str__(self):
        return self.__repr__() + "│"

    def __call__(self):
        self.cache: np.ndarray = np.copy(self.tensor)
        return self.cache

    def backward(self, prop:np.ndarray):
        self.tensor.grad += prop

    def zero_grad(self):
        self.tensor.grad = np.zeros_like(self.tensor, dtype=np.float64)


class CalcGraphIterator:
    def __init__(self, graph: CalcGraph):
        self.graph = graph
        self.visited = []
        self.visit_stack: list[CalcGraph] = [self.graph]
        self.curr_pos = self.graph

    def __next__(self) -> CalcGraph:
        if len(self.visit_stack) == 0:
            raise StopIteration
        
        self.curr_pos = self.visit_stack.pop(0)
        if self.curr_pos in self.visited:
            return self.__next__()
        if self.curr_pos.param != [None]:
            for next_child in self.curr_pos.param:
                self.visit_stack.insert(0, next_child)
        self.visited.append(self.curr_pos)
        return self.curr_pos

    def __iter__(self):
        return self