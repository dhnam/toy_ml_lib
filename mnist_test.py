from layer import FlattenLayer, LinearLayer, ConvolutionLayer
from model import SequentialModel
from optimizer import AdamOptimizer
from tensor import Tensor
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from tqdm import tqdm

class State(Enum):
    MAGIC_NUM = 1
    NUMITEM = 2
    NUMROW = 3
    NUMCOL = 4
    ITEMS = 5

def open_image_file(path: str) -> np.ndarray:
    state = State.MAGIC_NUM
    state_bytes = {
        State.MAGIC_NUM: 4,
        State.NUMITEM: 4,
        State.NUMROW: 4,
        State.NUMCOL: 4,
        State.ITEMS: 1,
    }
    item_count = 0
    num_rows = 0
    num_cols = 0
    imgs = None
    img_idx = 0
    with open(path, "rb") as image_file:
        while next_bytes := image_file.read(state_bytes[state]):
            match state:
                case State.MAGIC_NUM:
                    assert next_bytes == (2051).to_bytes(4, "big")
                    state = State.NUMITEM
                case State.NUMITEM:
                    item_count = int.from_bytes(next_bytes, "big", signed=True)
                    state = State.NUMROW
                case State.NUMROW:
                    num_rows = int.from_bytes(next_bytes, "big", signed=True)       
                    state = State.NUMCOL
                case State.NUMCOL:
                    num_cols = int.from_bytes(next_bytes, "big", signed=True)
                    state = State.ITEMS
                    state_bytes[State.ITEMS] = num_rows * num_cols
                    imgs = np.zeros((item_count, num_rows, num_cols), dtype=np.int32)
                case State.ITEMS:
                    imgs[img_idx, :, :] = np.asarray(bytearray(next_bytes)).reshape((num_rows, num_cols))
                    img_idx += 1
    return imgs

            

def open_label_file(path: str) -> np.ndarray:
    state = State.MAGIC_NUM
    state_bytes = {
        State.MAGIC_NUM: 4,
        State.NUMITEM: 4,
        State.ITEMS: 1,
    }
    item_count = 0
    labels = None
    label_idx = 0
    with open(path, "rb") as image_file:
        while next_bytes := image_file.read(state_bytes[state]):
            match state:
                case State.MAGIC_NUM:
                    assert next_bytes == (2049).to_bytes(4, "big")
                    state = State.NUMITEM
                case State.NUMITEM:
                    item_count = int.from_bytes(next_bytes, "big", signed=True)
                    state = State.ITEMS
                    labels = np.zeros((item_count, ), dtype=np.int32)
                case State.ITEMS:
                    labels[label_idx] = int.from_bytes(next_bytes, "big", signed=False)
                    label_idx += 1
    return labels


def image_label_set(img_path, label_path) -> tuple[np.ndarray, np.ndarray]:
    imgs = open_image_file(img_path)
    labels = open_label_file(label_path)
    return imgs, labels

def img_preprocess(img):
    return Tensor(np.reshape(img / 255., (img.shape[0], 1, img.shape[1], img.shape[2])))

def softmax(x: Tensor, cate) -> Tensor:
    sum_ = np.average(np.exp(x), axis=1) * x.shape[1]
    cate_sum =np.exp(np.take(x, cate, axis=1))
    return np.asarray(cate_sum) / (sum_ + 0.00001) + 0.00001


def cross_entropy(pred:Tensor, real:Tensor) -> Tensor:
    sum_ = np.zeros((pred.shape[0],)).view(Tensor)
    sum_.name = "sum_"
    for next_cate in range(pred.shape[1]):
        softmax_ = softmax(pred, next_cate)
        softmax_.name = "softmax_"
        log_softmax = np.log(softmax_)
        log_softmax.name = "log_softmax"
        sum_ = sum_ + real[:, next_cate] * log_softmax
    return sum_



img, label = image_label_set("./mnist_database/t10k-images.idx3-ubyte", "./mnist_database/t10k-labels.idx1-ubyte")
img = img_preprocess(img)
idx = 42


model = SequentialModel(
    [
        ConvolutionLayer(1, 3, 5), # (N, 1, 28, 28) -> (N, 3, 24, 24)
        ConvolutionLayer(3, 5, 5), # (N, 3, 24, 24) -> (N, 5, 20, 20)
        FlattenLayer(), # (N, 5, 20, 20) -> (N, 2000)
        LinearLayer(2000, 1) # (N, 2000) -> (N, 1)
    ]
)

optim = AdamOptimizer(model)
bar = tqdm(range(100))
for i in bar:
    ints = np.random.randint(img.shape[0], size=2)
    next_sample = img[ints, :].view(np.ndarray).view(Tensor)
    y = model(next_sample)
    y.name = "y"
    y_real = np.reshape(label[ints], y.shape)
    loss = cross_entropy(y, y_real)
    with open("output.txt", "w") as f:
        print(loss.calc_graph,file=f)
    bar.set_description(f"{loss[0]}")
    loss.backward()
    optim.step()

