from typing import final
from calc_graph import CalcGraph
from tensor import Tensor
from model import Model
import abc
import numpy as np

class Optimizer(abc.ABC):
    def __init__(self, input_:Tensor | Model, lr=0.001):
        self.params: list[CalcGraph] = []
        self.need_param_init = False
        if isinstance(input_, Tensor):
            loss_tensor = input_
            for next_graph in loss_tensor.calc_graph:
                if next_graph.tensor.trainable:
                    self.params.append(next_graph)
        else:
            assert(isinstance(input_, Model))
            self.model = input_
            self.need_param_init = True

        self.lr = lr
    
    @final
    def step(self, zero_grad=True):
        if self.need_param_init:
            self.params.extend(x.calc_graph for x in self.model.trainable_param)
            self.need_param_init = False

        for next_graph in self.params:
            next_graph.tensor = self.optimize(next_graph.tensor)
            # print()
            # print(np.asarray(next_graph.tensor.view(np.ndarray)).flat[-1])

        if zero_grad:
            for next_graph in self.params:
                next_graph.zero_grad()

    @staticmethod
    def optimize(self, next_tensor: Tensor) -> np.ndarray:
        pass


class GradientDescentOptimizer(Optimizer):
    def optimize(self, next_tensor: Tensor) -> np.ndarray:
        # print(np.asarray(next_tensor.view(np.ndarray)).flat[-1])
        # print(next_tensor.calc_graph)
        # print(next_tensor)
        # print(next_tensor.grad)
        return next_tensor - self.lr * next_tensor.grad

class MomentumOptimizer(Optimizer):
    def __init__(self, input_:Tensor | Model, lr=0.001, momentum=0.9):
        super().__init__(input_, lr)
        self.momentum_dict: dict[CalcGraph, np.ndarray] = {}
        self.momentum = momentum

    def optimize(self, next_tensor: Tensor) -> np.ndarray:
        next_graph = next_tensor.calc_graph
        if next_graph not in self.momentum_dict:
            self.momentum_dict[next_graph] = next_tensor.grad
        else:
            self.momentum_dict[next_graph] = self.momentum * self.momentum_dict[next_graph] + next_tensor.grad
        return next_tensor - self.lr * self.momentum_dict[next_graph]

class NAGOptimizer(Optimizer):
    def __init__(self, input_:Tensor | Model, lr=0.001, momentum=0.9):
        super().__init_(input_, lr)
        self.momentum_dict: dict[CalcGraph, np.ndarray] = {}
        self.grad_before: dict[CalcGraph, np.ndarray] = {}
        self.momentum = momentum

    def optimize(self, next_tensor: Tensor) -> np.ndarray:
        next_graph = next_tensor.calc_graph
        if next_graph not in self.momentum_dict:
            self.momentum_dict[next_graph] = next_tensor.grad
        else:
            self.momentum_dict[next_graph] = self.momentum * self.momentum_dict[next_graph] + next_tensor.grad

        if next_graph not in self.grad_before:
            self.grad_before[next_graph] = next_tensor.grad
        self.grad_before[next_graph] = next_tensor.grad + self.momentum*self.momentum_dict[next_graph]
        # self.grad_before[next_graph] = self.momentum_dict[next_graph]
        res = next_tensor - self.lr * self.grad_before[next_graph]
        # self.grad_before[next_graph] = next_tensor.grad
        return res


class AdaGradOptimizer(Optimizer):
    def __init__(self, input_:Tensor | Model, lr=0.01, eps=1e-10):
        super().__init__(input_, lr)
        self.sum_dict: dict[CalcGraph, np.ndarray] = {} # G
        self.eps = eps

    def optimize(self, next_tensor: Tensor) -> np.ndarray:
        next_graph = next_tensor.calc_graph
        if next_graph not in self.sum_dict:
            self.sum_dict[next_graph] = np.zeros_like(next_tensor)
        self.sum_dict[next_graph] = self.sum_dict[next_graph] + np.square(next_tensor.grad)
        next_lr = self.lr / (np.sqrt(self.sum_dict[next_graph]) + self.eps)
        return next_tensor - next_lr * next_tensor.grad


class RMSPropOptimizer(Optimizer):
    def __init__(self, input_:Tensor | Model, lr=0.01, alpha=0.99, eps=1e-8):
        super().__init__(input_, lr)
        self.sum_dict: dict[CalcGraph, np.ndarray] = {} # G
        self.alpha = alpha
        self.eps = eps

    def optimize(self, next_tensor: Tensor) -> np.ndarray:
        next_graph = next_tensor.calc_graph
        if next_graph not in self.sum_dict:
            self.sum_dict[next_graph] = np.zeros_like(next_tensor)
        self.sum_dict[next_graph] = self.alpha * self.sum_dict[next_graph] + (1 - self.alpha) * np.square(next_tensor.grad)
        next_lr = self.lr / (np.sqrt(self.sum_dict[next_graph]) + self.eps)
        return next_tensor - next_lr * next_tensor.grad


class AdamOptimizer(Optimizer):
    def __init__(self, input_:Tensor | Model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(input_, lr)
        self.momentum_dict: dict[CalcGraph, np.ndarray] = {}
        self.velocity_dict: dict[CalcGraph, np.ndarray] = {} # G
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def optimize(self, next_tensor: Tensor) -> np.ndarray:
        next_graph = next_tensor.calc_graph
        if next_graph not in self.momentum_dict:
            self.momentum_dict[next_graph] = next_tensor.grad
        else:
            self.momentum_dict[next_graph] = self.beta1 * self.momentum_dict[next_graph] + (1 - self.beta1) * next_tensor.grad
        if next_graph not in self.velocity_dict:
            self.velocity_dict[next_graph] = np.zeros_like(next_tensor)
        self.velocity_dict[next_graph] = self.beta2 * self.velocity_dict[next_graph] + (1 - self.beta2) * np.square(next_tensor.grad)
        next_lr = self.lr / (np.sqrt(self.velocity_dict[next_graph] + self.eps))
        return next_tensor - next_lr * self.momentum_dict[next_graph]