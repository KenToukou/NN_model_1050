from abc import ABCMeta, abstractclassmethod


class BaseLayer(metaclass=ABCMeta):
    @abstractclassmethod
    def forward(self):
        pass

    @abstractclassmethod
    def backward(self):
        pass
