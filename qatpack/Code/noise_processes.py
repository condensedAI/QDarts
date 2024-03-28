import numpy as np

class OU_process:
    def __init__(self, sig, tc, dt, num_points):
        self.sig = sig
        self.tc = tc
        self.num_points = num_points
        self.dt = dt
        self.x = self.sig * np.random.normal(0,1)
        self.a = (np.exp(-self.dt/self.tc))
        self.b = np.sqrt(1-np.exp(-2*self.dt/self.tc))*self.sig

    def next_val(self):
        n = np.random.normal(0,1)
        self.x = self.x*self.a +self.b*n
        return self.x
    
    def __call__(self):
        vals = np.zeros(self.num_points)
        for i in range(0, self.num_points):
            vals[i] = self.next_val()
        return vals
    


