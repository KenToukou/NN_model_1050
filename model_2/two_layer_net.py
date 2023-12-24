import os  # noqa
import sys  # noqa

import numpy as np
from functions import (
    cross_entropy_error,
    numerical_gradient,
    sigmoid,
    sigmoid_grad,
    soft_max,
)


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, out_put_size, wieght_init_std=0.01):
        self._params = {}
        self._params["W1"] = wieght_init_std * np.random.randn(input_size, hidden_size)
        self._params["b1"] = np.zeros(hidden_size)
        self._params["W2"] = wieght_init_std * np.random.randn(
            hidden_size, out_put_size
        )
        self._params["b2"] = np.zeros(out_put_size)

    def predict(self, x):
        W1, W2 = self._params["W1"], self._params["W2"]
        b1, b2 = self._params["b1"], self._params["b2"]
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = soft_max(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_decent(self, x, t):
        loss_W = lambda W: self.loss(x, t)  # noqa
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self._params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self._params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self._params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self._params["b2"])
        return grads

    def gradient(self, x, t):
        W1, W2 = self._params["W1"], self._params["W2"]
        b1, b2 = self._params["b1"], self._params["b2"]
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = soft_max(a2)

        # backward
        dy = (y - t) / batch_num
        grads["W2"] = np.dot(z1.T, dy)
        grads["b2"] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads["W1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis=0)

        return grads
