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
                    return (np.broadcast_to(np.expand_dims(propa / self.arr.shape[self.axis], self.axis), self.arr.shape),)
            else:
                raise NotImplementedError
        return backward


@implements(np.ravel)
class RavelFuncClassMaker(FuncClassMaker):
    def __init__(self, arr: np.ndarray | Tensor):
        self.arr = arr
    
    def args_to_class_name(self)->str:
        def shape_to_identifier(shape: tuple[int]):
            identifier = str(shape)
            identifier = identifier.replace("(", "")
            identifier = identifier.replace(",", "_")
            identifier = identifier.replace(")", "")
            return identifier
        return f"FuncRavel_{shape_to_identifier(self.arr.shape)}"
    
    def args_to_func_name(self) -> str:
        return f"FuncRavel{self.arr.shape}"
    
    def make_forward(self) -> Callable[[Any], np.ndarray]:
        @staticmethod
        def forward(arr:np.ndarray|Tensor):
            return np.reshape(arr, (self.arr.shape[0], np.prod(self.arr.shape[1:])))
        return forward

    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        @staticmethod
        def backward(propa:np.ndarray, *args, **kwargs) -> tuple[np.ndarray]:
            return (np.reshape(propa, self.arr.shape), )
        return backward



class Convolution(FuncClassMaker):
    def __init__(self, x: np.ndarray | Tensor, kernel: np.ndarray | Tensor):
        self.x_shape = x.shape
        self.kernel_shape = kernel.shape
        self.ARR_CHANNEL = -3
        self.ARR_Y = -2
        self.ARR_X = -1
        self.KERNEL_OUT_CHANNEL = -4
        self.KERNEL_IN_CHANNEL = -3
        self.KERNEL_Y = -2
        self.KERNEL_X = -1
        self.FILT_SLICE = slice(2, 4)

    def args_to_class_name(self) -> str:
        return f"FuncConv2D"
    
    def args_to_func_name(self) -> str:
        return f"FuncConv2D {self.x_shape} * {self.kernel_shape}"

    def im2col(self, x: np.ndarray | Tensor) -> np.ndarray:
        passes_y = self.x_shape[self.ARR_Y] - self.kernel_shape[self.KERNEL_Y] + 1
        passes_x = self.x_shape[self.ARR_X] - self.kernel_shape[self.KERNEL_X] + 1
        passes = passes_x * passes_y
        kernel_size = self.kernel_shape[self.KERNEL_X] * self.kernel_shape[self.KERNEL_Y]
        channels = self.x_shape[self.ARR_CHANNEL]
        cols = np.zeros((passes, kernel_size * channels))
        for next_pass in range(passes):
            for next_channel in range(channels):
                y_coord = next_pass // passes_x
                x_coord = next_pass % passes_x
                next_filt = np.zeros(self.kernel_shape[self.FILT_SLICE])
                for next_y in range(self.kernel_shape[self.KERNEL_Y]):
                    for next_x in range(self.kernel_shape[self.KERNEL_X]):
                        next_filt[next_y, next_x] = x[next_channel][y_coord + next_y][x_coord + next_x]
                # print(cols[next_pass][next_channel * kernel_size : next_channel * kernel_size + kernel_size])
                cols[next_pass][next_channel*kernel_size : next_channel * kernel_size + kernel_size] = next_filt.flatten()
        return cols

    def col2im(self, col: np.ndarray | Tensor) -> np.ndarray:
        passes_y = self.x_shape[self.ARR_Y] - self.kernel_shape[self.KERNEL_Y] + 1
        passes_x = self.x_shape[self.ARR_X] - self.kernel_shape[self.KERNEL_X] + 1
        imgs = np.zeros((col.shape[0], passes_y, passes_x))
        for i, next_col in enumerate(col):
            imgs[i] = np.reshape(next_col, (passes_y, passes_x))
        return imgs

    def make_forward(self) -> Callable[[Any], np.ndarray]:
        @staticmethod
        def forward(x: np.ndarray | Tensor, kernel: np.ndarray | Tensor) -> np.ndarray:
            cols = []
            for next_x in x:
                cols.append(self.im2col(next_x))
            cols = np.asarray(cols)
            kernels = np.zeros((kernel.shape[self.KERNEL_OUT_CHANNEL], kernel.shape[self.KERNEL_IN_CHANNEL] * kernel.shape[self.KERNEL_X] * kernel.shape[self.KERNEL_Y]))
            for idx, next_kernel in enumerate(kernel):
                kernels[idx] = next_kernel.flatten()
            cols_conv = kernels @ np.transpose(cols, [0, 2, 1])
            output = []
            for next_cols in cols_conv:
                output.append(self.col2im(next_cols))
            return np.asarray(output)

        return forward

    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        @staticmethod
        def backward(propa: np.ndarray, x: np.ndarray | Tensor, kernel: np.ndarray | Tensor, *args, **kwargs) -> tuple[np.ndarray]:
            passes_y = self.x_shape[self.ARR_Y] - self.kernel_shape[self.KERNEL_Y] + 1
            passes_x = self.x_shape[self.ARR_X] - self.kernel_shape[self.KERNEL_X] + 1
            assert(propa.shape == (x.shape[0], kernel.shape[0], passes_y, passes_x))
            cols = []
            for next_x in x:
                cols.append(self.im2col(next_x))
            cols = np.asarray(cols)
            propa_cols = np.reshape(propa, (x.shape[0], propa.shape[1], propa.shape[2] * propa.shape[3]))
            # second one : propa @ cols.T
            # I don't know how it'll work on 3d
            # first one: kernel.T @ propa and col2im -> how would this even work
            kernels = np.zeros((kernel.shape[0], kernel.shape[1] * kernel.shape[2] * kernel.shape[3]))
            for idx, next_kernel in enumerate(kernel):
                kernels[idx] = next_kernel.flatten()
            grad_col = np.transpose((kernels.T @ propa_cols), [0, 2, 1])
            imgs = np.zeros(self.x_shape)
            kernel_size = self.kernel_shape[2] * self.kernel_shape[3]
            passes = passes_x * passes_y
            for j, next_grad_col in enumerate(grad_col):
                for i, next_col in enumerate(next_grad_col):
                    y_num = (i // passes) % passes_x
                    x_num = (i // passes) // passes_x
                    img_count = self.x_shape[1]
                    for img_num in range(img_count):
                        imgs[j, img_num, y_num:y_num+self.kernel_shape[2], x_num:x_num+self.kernel_shape[3]] += np.reshape(next_col[img_num*kernel_size:(img_num + 1)*kernel_size], self.kernel_shape[2:])
            propa_calc = propa.reshape((propa.shape[0], propa.shape[1], propa.shape[2]*propa.shape[3]))
            grad_kernel_calc = propa_calc @ cols
            grad_kernel_calc = np.sum(grad_kernel_calc, axis=0)
            grad_kernel = grad_kernel_calc.reshape(self.kernel_shape)
            return  imgs, grad_kernel

        return backward

implements(Convolution)(Convolution)

class ConvBiasAddition(FuncClassMaker):
    def __init__(self, conv: Tensor|np.ndarray, bias:Tensor|np.ndarray):
        self.conv_shape = conv.shape
        self.bias_shape = bias.shape
        assert(len(self.bias_shape) == 1 and self.conv_shape[1] == self.bias_shape[0])

    def args_to_func_name(self) -> str:
        return "Conv_Bias"

    def args_to_class_name(self) -> str:
        return "Conv_Bias"

    def make_forward(self) -> Callable[[Any], np.ndarray]:
        @staticmethod
        def forward(x: Tensor|np.ndarray, bias:Tensor|np.ndarray) -> np.ndarray:
            res = np.zeros(self.conv_shape)
            for i, next_x in enumerate(x):
                for j, next_img in enumerate(next_x):
                    res[i, j] = next_img + bias[j]
            return res

        return forward

    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        @staticmethod
        def backward(propa:np.ndarray, *args, **kwargs):
            assert(propa.shape == self.conv_shape)
            return propa, np.sum(propa, axis=(0,2,3))
        return backward

implements(ConvBiasAddition)(ConvBiasAddition)


@implements(np.take)
class TakeFuncClassMaker(FuncClassMaker):
    def __init__(self, arr: np.ndarray | Tensor, indices: int, axis: None|int=None):
        if type(indices) != int:
            if len(indices) == 1:
                indices = indices[0]
            else:
                raise NotImplementedError
        
        self.arr_shape = arr.shape
        self.indices = indices
        self.axis = axis

    def args_to_func_name(self) -> str:
        return f"Take_indice_{self.indices}_axis_{self.axis}"

    def args_to_class_name(self) -> str:
        return f"Take_indice_{self.indices}_axis_{self.axis}"

    def make_forward(self) -> Callable[[Any], np.ndarray]:
        @staticmethod
        def forward(arr: np.ndarray|Tensor, indices: int, axis: None|int=None) -> np.ndarray:
            return np.take(arr.view(np.ndarray), indices=indices, axis=axis)

        return forward

    def make_backward(self) -> Callable[[np.ndarray, Any], tuple[np.ndarray]]:
        @staticmethod
        def backward(propa: np.ndarray, *args, **kwargs) -> np.ndarray:
            shape_before = self.arr_shape
            if self.axis is not None:
                shape_before[self.axis] -= 1
            else:
                shape_before = (np.prod(self.arr_shape) - 1,)
            propa_next = np.zeros(shape_before)
            np.insert(propa_next, self.indices, propa, axis=self.axis)
            if self.axis is None:
                propa_next = propa_next.reshape(self.arr_shape)
            assert(propa_next.shape == self.arr_shape)
            return propa_next

        return backward


if __name__ == "__main__":
    a = np.asarray([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]])
    # a.shape : (N, in_channel, img_y, img_x)
    kernel = np.asarray([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[0, 1, 1], [1, 0, 1], [0, 1, 1]]]])
    # kernel.shape : (out_channel, in_channel, y_size, x_size)
    out = Convolution(a, kernel).forward(a, kernel)
    print(out)
    print(Convolution(a, kernel).backward(np.ones_like(out), a, kernel))