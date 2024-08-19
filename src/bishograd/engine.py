import math


class Hako:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Hako(data : {self.data:.2f})"

    def __add__(self, other):
        other = other if isinstance(other, Hako) else Hako(other)
        out = Hako(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Hako) else Hako(other)
        out = Hako(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Hako(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** (-1)

    def __rtruediv__(self, other):
        return other * self ** (-1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def tanh(self):
        x = self.data
        tanh_value = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Hako(tanh_value, (self,), "tanh")

        def _backward():
            self.grad += (1 - tanh_value**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Hako(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += out.grad if out.data > 0 else 0

        out._backward = _backward
        return out

    def sigmoid(self):
        out = Hako(1 / (1 + math.exp(-self.data)), (self,), "Sig")

        def _backward():
            self.grad += out.data(1 - out.data) * out.grad

        out._backward = _backward
        return out

    def linear(self):
        out = Hako(self.data, (self,), "linear")

        def _backward():
            self.grad += out.grad

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
