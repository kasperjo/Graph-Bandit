import numpy as np

class Reward:
    def __init_(self):
        pass
    def sample(self):
        raise NotImplementedError
        
class GaussianReward(Reward):
    def __init__(self,mu,sig):
        self._mu = mu
        self._sig = sig
    def sample(self,n_samples=1):
        r = np.random.randn(n_samples)*self._sig + self._mu
        if n_samples == 1:
            r = r[0]
        return r