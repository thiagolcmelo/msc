import numpy as np

class Test:
    def __init__(self):
        self.dt = 10
        exp = lambda x: self.dt * x
        self.expo = lambda x: np.exp(exp(x))
    
    def run(self, x):
        return self.expo(x)

    def set_dt(self, dt):
        self.dt = dt

a = Test()
print(a.run(10))
a.set_dt(2)
print(a.run(10))