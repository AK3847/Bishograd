import random
from bishograd.engine import Hako


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
    def __init__(self, nin, nouts, activations=None) -> None:
        sz = [nin] + nouts
        if activations is None:
            activations = ["linear"] * (len(nouts) - 1) + ["sigmoid"]
        self.layers = [
            Layer(sz[i], sz[i + 1], activation=activations[i])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layers in self.layers:
            x = layers(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0

    def train(self, epochs=50, lr_rate=0.001, x_input=None, y_output=None, stats=True):
        if stats:
            print(f"{'-'*8}Training Model{'-'*8}")
            print(
                f"Parameters = {len(self.parameters())}\nEpochs = {epochs}\nLearning Rate: {lr_rate}"
            )
            print(f"{'-'*30}")
        if x_input is None or y_output is None:
            raise ValueError("Invalid input for training data")
        for epoch in range(epochs + 1):
            y_pred = [self(x) for x in x_input]
            loss = sum((y1 - y2) ** 2 for y1, y2 in zip(y_pred, y_output))
            self.zero_grad()
            loss.backward()
            for param in self.parameters():
                param.grad = max(min(param.grad, 1), -1)
                param.data -= lr_rate * param.grad
            print(f"epoch: {epoch}/{epochs} ---- Loss: {loss.data:.4f}")
