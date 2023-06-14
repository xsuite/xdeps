import numpy as np

class FunctionPieceWiseLinear:

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def __call__(self, x):
        return np.interp(x, self.x, self.y, left=self.y[0], right=self.y[-1])

    @classmethod
    def from_dict(cls, dct):
        return cls(dct['x'], dct['y'])

    def to_dict(self):
        return {'x': self.x, 'y': self.y}