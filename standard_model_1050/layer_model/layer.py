import numpy as np

from .frame_work import BaseLayer
from .functions import cross_entropy_error, soft_max


class Relu(BaseLayer):
    def __init__(self):
        self._mask = None

    def forward(self, x: np.array):
        self._mask = x <= 0
        out = x.copy()
        out[self._mask] = 0
        return out

    def backward(self, d_out):
        d_out[self._mask] = 0
        dx = d_out
        return dx


class Sigmoid(BaseLayer):
    def __init__(self):
        self.out = None

    def forward(self, x: np.array):
        out = 1 / (1 + np.exp(-x))
        self.out = out  # yの値の相当する

    def backward(self, d_out: np.array):
        dx = d_out * self.out * (1 - self.out)
        return dx


class Affine(BaseLayer):  # numpyのdot計算を行う
    def __init__(self, W: np.array, b: np.array):
        self.W = W
        self.b = b
        self._x = None
        self._dw = None
        self._db = None

    def forward(self, x: np.array) -> np.array:
        self._x = x
        out = np.dot(self._x, self.W) + self.b
        return out

    def backward(self, d_out: np.array):
        dx = np.dot(d_out, self.W.T)
        self._dw = np.dot(self.x.T, d_out)
        self._db = np.sum(d_out, axis=0)
        return dx

    @property
    def x(self):
        return self._x

    @property
    def dw(self):
        return self._dw

    @property
    def db(self):
        return self._db


class SoftmaxWithLoss(BaseLayer):
    def __init__(self):
        self._loss = None
        self._y = None
        self._t = None

    def forward(self, y: np.array, t: np.array):
        self._y = soft_max(y)
        self._t = t
        self._loss = cross_entropy_error(y=self._y, t=self._t)
        return self._loss

    def backward(self, dout=1):
        batch_size = self._t.shape[0]
        if self._t.size == self._y.size:  # 教師データがone-hot-vectorの場合
            dx = (self._y - self._t) / batch_size
        else:
            dx = self._y.copy()
            dx[np.arange(batch_size), self._t] -= 1
            dx = dx / batch_size

        return dx
