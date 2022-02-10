from tensor import Tensor
from optimizer import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm

base_x = np.random.rand(100, 1) * 200 - 100
base_y = base_x * base_x * 2 + base_x * 13 + 5

x = Tensor(base_x + np.random.randn(100, 1) / 10)
x.name = "x"
y = Tensor(base_y + np.random.randn(100, 1) / 10)
y.name = "y"

lr = 1e-3
losses = []
layers = [Tensor(np.random.standard_normal((1, 10)), name="l1", trainable=True),
          Tensor(np.random.standard_normal((10, 10)), name="l2", trainable=True),
          Tensor(np.random.standard_normal((10, 1)), name="l3", trainable=True)]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def MSE(pred, real) -> Tensor:
    return np.average(np.square(real - pred))




fig, (ax1, ax2) = plt.subplots(2, 1)
x_axe = np.linspace(-100, 100, 200)
ims = []


def test(num):
    test_input = Tensor(np.asarray([[num]]))
    layer1 = sigmoid(test_input @ layers[0])
    layer2 = sigmoid(layer1 @ layers[1])
    out = layer2 @ layers[2]
    return out

def func(num):
    return num * num * 2 + num * 13 + 5

y_real = [func(x).tolist() for x in x_axe]

np.seterr(all="ignore")
def train(num):
    global losses
    tbar = tqdm(range(num))
    for _ in tbar:
        layer1 = sigmoid(x @ layers[0])
        layer1.name = "layer1"
        layer2 = sigmoid(layer1 @ layers[1])
        layer2.name = "layer2"
        out: Tensor = layer2 @ layers[2]
        out.name = "out"
        out_cp = out.copy()
        loss: Tensor = MSE(out, y)
        loss.name = "loss"
        losses.append(loss.copy())
        tbar.set_description(f"loss={losses[-1].tolist()},val={out.flatten()[5]}, calc_graph={loss.calc_graph.param[0].param[0].param[1].tensor.flatten()[5]}, grad={out.grad.flatten()[5]}"
                            f", same_id={id(out) == id(loss.calc_graph.param[0].param[0].param[1].tensor)}")
        optim = GradientDescentOptimizer(loss, lr)
        optim.step()

        im1, = ax1.plot([x.tolist() for x in losses], "r")
        y_pred = [test(x).tolist()[0][0] for x in x_axe]
        im2, im3 = ax2.plot(x_axe, y_real, 'r', x_axe, y_pred, 'b')
        # im2 = ax2.scatter(x.view(np.ndarray), out_cp.view(np.ndarray), c='b')
        # im3 = ax2.scatter(x.view(np.ndarray), y.view(np.ndarray), c='r')
        ims.append([im1, im2, im3])
    print(loss.calc_graph)



train(100)

ani = anim.ArtistAnimation(fig, ims, interval=50)

plt.show()