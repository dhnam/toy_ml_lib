from layer import LinearLayer, Sigmoid, ReLU
from tensor import Tensor
from optimizer import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import SequentialModel


batch_size = 100

base_x = np.random.rand(batch_size, 1) * 200 - 100
base_y = base_x * base_x * 2 + base_x * 13 + 5

x = Tensor(base_x + np.random.randn(batch_size, 1) / 10)
x.name = "x"
y = Tensor(base_y + np.random.randn(batch_size, 1) / 10)
y.name = "y"

lr = 1e-2
losses = []

def activation(x):
    return ReLU(x)

def MSE(pred, real) -> Tensor:
    return np.average(np.square(real - pred))




fig, (ax1, ax2) = plt.subplots(2, 1)
x_axe = np.linspace(-200, 200, 400)
ims = []

def test(num):
    global model
    test_input = Tensor(np.asarray([[num]]))
    return model(test_input)

def func(num):
    return num * num * 2 + num * 13 + 5

y_real = [func(x).tolist() for x in x_axe]

optimizer = AdamOptimizer

model = SequentialModel([
    LinearLayer(1, 10, name="L1", activation=ReLU),
    LinearLayer(10, 10, name="L2", activation=ReLU),
    LinearLayer(10, 1, name="L3"),
])

np.seterr(all="ignore")
def train(num):
    global model

    tbar = tqdm(range(num))
    a = ""
    optim = optimizer(model, lr)
    for i in tbar:
        out = model(x)
        loss: Tensor = MSE(out, y)
        loss.name = "loss"
        loss.backward()
        tbar.set_description(f"loss={losses[-1].tolist()},val={model.layers[2].weight.flatten()[0]},grad={model.layers[2].weight.grad.flatten()[0]}")
        optim.step()
        
        losses.append(loss.copy())


    # print(loss.calc_graph)


train(1000)

plt.subplot(2, 1, 1)
plt.plot([x.tolist() for x in losses])
plt.subplot(2, 1, 2)

y_real = [func(x).tolist() for x in x_axe]
y_pred = [test(x).tolist()[0][0] for x in x_axe]
plt.plot(x_axe, y_real, 'r', x_axe, y_pred, 'b')

plt.show()