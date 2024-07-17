from typing import Self


class Hako:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Hako(data : {self.data})"

    def __add__(self, other) -> Self:
        other = other if isinstance(other, Hako) else Hako(other)
        out = Hako(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other) -> Self:
        other = other if isinstance(other, Hako) else Hako(other)
        out = Hako(self.data * other.data, (self, other), "*")
        return out
