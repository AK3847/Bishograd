import random
from src.bishograd.engine import Hako


class Neuron:
    def __init__(self, nin):
        self.w = [Hako(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Hako(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        out = act.tanh()
        return out


class Layer:
    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs


class MLP:
    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layers in self.layers:
            x = layers(x)
        return x
