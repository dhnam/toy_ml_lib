from tensor import Tensor
from optimizer import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm


batch_size = 100

base_x = np.random.rand(batch_size, 1) * 200 - 100
base_y = base_x * base_x * 2 + base_x * 13 + 5

x = Tensor(base_x + np.random.randn(batch_size, 1) / 10)
x.name = "x"
y = Tensor(base_y + np.random.randn(batch_size, 1) / 10)
y.name = "y"

lr = 1e-2
losses = []
# np.random.uniform(-(1/np.sqrt()), (1/np.sqrt()), size=())
layers = [Tensor(np.random.uniform(-(1/np.sqrt(1)), (1/np.sqrt(1)), size=(1, 10)), name="l1", trainable=True),
          Tensor(np.random.uniform(-(1/np.sqrt(10)), (1/np.sqrt(10)), size=(10, 10)), name="l2", trainable=True),
          Tensor(np.random.uniform(-(1/np.sqrt(10)), (1/np.sqrt(10)), size=(10, 1)), name="l3", trainable=True)]


biases = [Tensor(np.random.uniform(-(1/np.sqrt(1)), (1/np.sqrt(1)), size=(10)), name="b1", trainable=True),
          Tensor(np.random.uniform(-(1/np.sqrt(10)), (1/np.sqrt(10)), size=(10)), name="b2", trainable=True),
          Tensor(np.random.uniform(-(1/np.sqrt(10)), (1/np.sqrt(10)), size=(1)), name="b3", trainable=True)]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def ReLU(x):
    return np.maximum(x, 0)

def activation(x):
    return ReLU(x)

def MSE(pred, real) -> Tensor:
    return np.average(np.square(real - pred))




fig, (ax1, ax2) = plt.subplots(2, 1)
x_axe = np.linspace(-200, 200, 400)
ims = []


def test(num):
    test_input = Tensor(np.asarray([[num]]))
    layer1 = activation(test_input @ layers[0] + biases[0])
    layer2 = activation(layer1 @ layers[1] + biases[1])
    out = layer2 @ layers[2] + biases[2]
    return out

def func(num):
    return num * num * 2 + num * 13 + 5

y_real = [func(x).tolist() for x in x_axe]

optimizer = AdamOptimizer

np.seterr(all="ignore")
def train(num):
    global losses
    tbar = tqdm(range(num))
    a = ""
    layer1 = activation(x @ layers[0] + biases[0])
    layer1.name = "layer1"
    layer2 = activation(layer1 @ layers[1] + biases[1])
    layer2.name = "layer2"
    out: Tensor = layer2 @ layers[2] + biases[2]
    out.name = "out"
    loss: Tensor = MSE(out, y)
    loss.name = "loss"
    optim = optimizer(loss, lr)
    for i in tbar:
        layer1 = activation(x @ layers[0] + biases[0])
        layer1.name = "layer1"
        layer2 = activation(layer1 @ layers[1] + biases[1])
        layer2.name = "layer2"
        out: Tensor = layer2 @ layers[2] + biases[2]
        out.name = "out"

        out_cp = out.copy()
        loss: Tensor = MSE(out, y)
        loss.name = "loss"
        losses.append(loss.copy())
        loss.backward()
        tbar.set_description(f"loss={losses[-1].tolist()},val={layers[2].flatten()[0]},grad={layers[2].grad.flatten()[0]}")
        optim.step(backward=False)

        # im1, = ax1.plot([x.tolist() for x in losses], "r")
        # y_pred = [test(x).tolist()[0][0] for x in x_axe]
        # im2, im3 = ax2.plot(x_axe, y_real, 'r', x_axe, y_pred, 'b')
        # im2.axes.set_ylim(np.min(y_real) - i * 100, np.max(y_real) + i * 100)
        # im3.axes.set_ylim(np.min(y_real) - i * 100, np.max(y_real) + i * 100)
        # im2 = ax2.scatter(x.view(np.ndarray), out_cp.view(np.ndarray), c='b')
        # im3 = ax2.scatter(x.view(np.ndarray), y.view(np.ndarray), c='r')
        # ims.append([im1, im2, im3])
        # print("\n========\n")


    # print(loss.calc_graph)

def train_step(i, ims, optim:Optimizer):
    global layers
    layer1 = activation(x @ layers[0] + biases[0])
    layer1.name = "layer1"
    layer2 = activation(layer1 @ layers[1] + biases[1])
    layer2.name = "layer2"
    out: Tensor = layer2 @ layers[2] + biases[2]
    out.name = "out"

    out_cp = out.copy()
    loss: Tensor = MSE(out, y)
    loss.name = "loss"
    losses.append(loss.copy())
    loss.backward()
    # tbar.set_description(f"loss={losses[-1].tolist()},val={out.flatten()[0]}, calc_graph={loss.calc_graph.param[0].param[0].param[1].tensor.flatten()[0]}, grad={out.grad.flatten()[0]}"
                        # f", same_id={id(out) == id(loss.calc_graph.param[0].param[0].param[1].tensor)}")
    optim.step(backward=False)

    im1, = ax1.plot([x.tolist() for x in losses], "r")
    ims[0].update_from(im1)
    # ims[0].set_data(im1.get_xdata(), im1.get_ydata())
    y_pred = [test(x).tolist()[0][0] for x in x_axe]
    ax2.set_ylim(np.min([y_real, y_pred]), np.max([y_real, y_pred]))
    min_, max_ = ax2.get_ylim()
    range_ = max_ - min_
    ax2.set_ylim(min_ - range_ / 10, max_ + range_ / 10)
    im2, im3 = ax2.plot(x_axe, y_real, 'r', x_axe, y_pred, 'b')
    ims[2].update_from(im3)
    # ims[2].set_data(im3.get_xdata(), im3.get_ydata())
    # im2.axes.set_ylim(np.min(y_real) - i * 100, np.max(y_real) + i * 100)
    # im3.axes.set_ylim(np.min(y_real) - i * 100, np.max(y_real) + i * 100)
    # im2 = ax2.scatter(x.view(np.ndarray), out_cp.view(np.ndarray), c='b')
    # im3 = ax2.scatter(x.view(np.ndarray), y.view(np.ndarray), c='r')
    # ims.append([im1, im2, im3])
    # print("\n========\n")
    return im1, im2, im3


train(5000)

plt.subplot(2, 1, 1)
plt.plot([x.tolist() for x in losses])
plt.subplot(2, 1, 2)

y_real = [func(x).tolist() for x in x_axe]
y_pred = [test(x).tolist()[0][0] for x in x_axe]
plt.plot(x_axe, y_real, 'r', x_axe, y_pred, 'b')


""" im1, = ax1.plot([x.tolist() for x in losses], "r")
im2, im3 = ax2.plot(x_axe, y_real, 'r', x_axe, y_real, 'b')

layer1 = activation(x @ layers[0] + biases[0])
layer1.name = "layer1"
layer2 = activation(layer1 @ layers[1] + biases[1])
layer2.name = "layer2"
out: Tensor = layer2 @ layers[2] + biases[2]
out.name = "out"
loss: Tensor = MSE(out, y)
loss.name = "loss"
losses.append(loss.copy())
optim = optimizer(loss, lr)

ani = anim.FuncAnimation(plt.gcf(), train_step, tqdm(range(100)), interval=50, fargs=([im1, im2, im3], optim), blit=True) """

# plt.show()

# ani = anim.ArtistAnimation(fig, ims, interval=50)
plt.show()