import numpy as np        

class OU_process:
    """
    Implements the Ornstein-Uhlenbeck noise process
    """
    def __init__(self, sig, tc, dt, num_points, num_sensors):
        self.sig = sig
        self.tc = tc
        self.num_points = num_points
        self.dt = dt
        self.a = (np.exp(-self.dt/self.tc))
        self.b = np.sqrt(1-np.exp(-2*self.dt/self.tc))*self.sig
        self.x = self.sig * np.random.randn(num_sensors)
    def slice(self, P,m):
        return self
    def next_val(self):
        n = np.random.randn(len(self.x))
        self.x = self.x*self.a +self.b*n
        return self.x
    
    def __call__(self,v):
        vals = np.zeros((self.num_points,len(self.x)))
        for i in range(0, self.num_points):
            vals[i] = self.next_val()
        return vals
    
class Cosine_Mean_Function:
    """ Decorator that models an additive mean term that depends on the gate voltages. This term is added
        to noise values sampled from the decorated noise model
    
        The mean term is given as a set of cosine functions:
        mu_i(v)= sum_j a_ij cos(2pi (w_ij^Tv+b_ij))
        
        The user supplies the weight tensor W with elements W_ijk so that W[i,j] is the vector w_ij and a matrix a with the amplitude values a_ij.
        Finally, b is the matrix of offsets b_ij, which can be left as None, 
        in which case it is sampled uniformly between 0 and 1. Thus, the mean field uses the same amplitudes and weights
        for
    """
    def __init__(self, noise_model, a, W, b=None):
        self.noise_model = noise_model
        self.a = a
        self.W = W
        self.b = b if not (b is None) else np.random.uniform(size=a.shape)
    def slice(self, P,m):
        new_W = np.einsum('ijk,kl->ijl',self.W,P)
        new_b = self.b + np.einsum('ijk,k->ij',self.W,m)
        return Cosine_Mean_Function(self.noise_model, self.a, new_W, new_b)
    
    def __call__(self,v):
        
        
        #compute noise
        noise_values = self.noise_model(v)
        
        #add the mean
        activation = 2*np.pi*(np.einsum('ijk,k->ij',self.W,v)+self.b)
        mean = np.sum(2*np.pi*self.a*np.cos(activation),axis=1)
        noise_values += mean.reshape(1,-1)
        
        return noise_values


