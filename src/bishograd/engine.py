import math
from typing import Self


class Hako:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Hako(data : {self.data})"

    def __add__(self, other) -> Self:
        other = other if isinstance(other, Hako) else Hako(other)
        out = Hako(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> Self:
        other = other if isinstance(other, Hako) else Hako(other)
        out = Hako(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        tanh_value = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Hako(tanh_value, (self,), "tanh")

        def _backward():
            self.grad += (1 - tanh_value**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Hako(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        vis = set()

        def build_topo(curr):
            if curr not in vis:
                vis.add(curr)
                for child in curr._prev:
                    build_topo(child)
                topo.append(curr)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
