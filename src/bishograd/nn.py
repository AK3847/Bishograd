import random
from src.bishograd.engine import Hako


class Neuron:
    def __init__(self, nin, activation="relu"):
        self.w = [Hako(random.uniform(-0.5, 0.5)) for _ in range(nin)]
        self.b = Hako(random.uniform(-0.5, 0.5))
        self.activation = activation

    def __call__(self, x):
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        if self.activation == "relu":
            out = act.relu()
        elif self.activation == "sigmoid":
            out = act.sigmoid()
        elif self.activation == "tanh":
            out = act.tanh()
        elif self.activation == "linear":
            out = act.linear()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, activation="relu") -> None:
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layers in self.layers:
            x = layers(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0
