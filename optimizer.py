from typing import final
from tensor import Tensor
import abc
import numpy as np

class Optimizer(abc.ABC):
    def __init__(self, loss_tensor:Tensor, lr):
        self.loss_tensor = loss_tensor
        self.lr = lr
    
    @final
    def step(self, backward=True, zero_grad=True):
        if backward:
            self.loss_tensor.backward()

        for next_graph in self.loss_tensor.calc_graph:
            if next_graph.tensor.trainable:
                next_graph.tensor = self.optimize(next_graph.tensor)
            # print(np.asarray(next_graph.tensor.view(np.ndarray)).flat[-1])

        if zero_grad:
            self.loss_tensor.zero_grad()

    @staticmethod
    def optimize(self, next_tensor: Tensor) -> np.ndarray:
        pass


class GradientDescentOptimizer(Optimizer):
    def optimize(self, next_tensor: Tensor) -> np.ndarray:
        # print(np.asarray(next_tensor.view(np.ndarray)).flat[-1])
        return next_tensor - self.lr * next_tensor.grad