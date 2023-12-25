from collections import OrderedDict

import numpy as np

from .layer_model import Affine, Relu, Sigmoid, SoftmaxWithLoss


class MultiLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size_list: list,
        output_size: int,
        activation: str = "relu",
        weight_int_std: str = "relu",
        weight_decay_lambda: int = 0,
    ):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.out_put_size = output_size
        self.hidden_lyer_num = len(self.hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params: dict = {}
        self._init_weight(weight_int_std)
        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_lyer_num + 1):
            self.layers[f"Affine_{idx}"] = Affine(
                self.params[f"W_{idx}"], self.params[f"b_{idx}"]
            )
            self.layers[f"Activation_function_{idx}"] = activation_layer[activation]()
        idx = self.hidden_lyer_num + 1
        self.layers[f"Affine_{idx}"] = Affine(
            self.params[f"W_{idx}"], self.params[f"b_{idx}"]
        )
        self.last_layer = SoftmaxWithLoss()

    def _init_weight(self, weight_int_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.out_put_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_int_std
            if str(weight_int_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_int_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params[f"W_{idx}"] = scale * np.random.randn(
                all_size_list[idx - 1], all_size_list[idx]
            )
            self.params[f"b_{idx}"] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_lyer_num + 2):
            W = self.params[f"W_{idx}"]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)
        d_out = 1
        d_out = self.last_layer.backward(dout=d_out)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            d_out = layer.backward(d_out)
        grads: dict = {}
        for idx in range(1, self.hidden_lyer_num + 2):
            grads[f"W_{idx}"] = self.layers[f"Affine_{idx}"].dw + (
                self.weight_decay_lambda * self.layers[f"Affine_{idx}"].W
            )
            grads[f"b_{idx}"] = self.layers[f"Affine_{idx}"].db
        return grads
