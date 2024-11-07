import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractNoiseProcess(metaclass=ABCMeta):
    r"""
    Models a noise process that produces possibly dependent samples :math:`\epsilon(v)_t`. The noise process
    can depend on the device voltages v and the noise can depend on all previous samples in the sequence. We assume
    that :math:`\epsilon(v)_t` is vector valued and the number of elements is stored in the ``num_elements`` attribute

    A sequence is started by calling ``start_sequence``, at which point the newly sampled points are independent from
    all previous samples.

    Note that currently, the elements in the noise process are assumed to be drawn with equal time difference between samples.

    Attributes
    ----------
        num_elements: int
            the dimensionality of the noise variable
    """

    def __init__(self, num_elements):
        self.num_elements = num_elements

    @abstractmethod
    def slice(self, P, m):
        """Restricts the noise to the affine subspace :math:`v=m+Pv`.

        Parameters
        ----------
        P : MxK np.array of floats
            The linear coefficient matrix.
        m: M np.array of floats
            offset of the affine transformation.

        Returns
        -------
        A noise process object describing the noise on the affine subspace. The current noise object remains unchanged.
        """
        pass

    @abstractmethod
    def start_sequence(self):
        """
        Restart the sequence so that the next sample drawn is independent from all previous samples.
        """
        pass

    @abstractmethod
    def __call__(self, v):
        r"""Returns the next element of the noise process.

        Parameters
        ----------
        v : K np.array of floats
            The current voltage parameter of the device

        Returns
        -------
        An element of the noise process :math:`\epsilon(v)_t`. This is assumed to be a vector of M elements,
        """
        pass


class OU_process(AbstractNoiseProcess):
    r"""Implements the Ornstein-Uhlenbeck noise process

    This noise is independent of v. We have that

    .. math::
        \epsilon(v)_{t+1}=a \epsilon(v)_t + \sqrt{1-a^2} \sigma \epsilon_t

    Where :math:`\epsilon_t` is independent gaussian noise and

    .. math::
        a=e^{-\\frac{\Delta t}{t_c}}

    It is possible to generate multiple independent samples of the same process simultaneously.
    """

    def __init__(self, sigma, tc, dt, num_elements):
        """Instantiates the OU process with its process parameters.

        Arguments
        ---------
        sigma: float
            Standard deviation of the OU process
        tc: float
            time correlation parameters, the higher the more samples are correlated
        dt: float
            time step between samples. The higher, the less samples are correlated
        num_elements: int
            How many independnet samples are computed simultaneously
        """
        super().__init__(num_elements)
        self.sigma = sigma
        self.tc = tc
        self.dt = dt
        self.a = np.exp(-self.dt / self.tc)
        self.b = np.sqrt(1 - np.exp(-2 * self.dt / self.tc)) * self.sigma

    def slice(self, P, m):
        return self

    def start_sequence(self):
        self.x = self.sigma * np.random.randn(self.num_elements)

    def next_val(self):
        n = np.random.randn(len(self.x))
        self.x = self.x * self.a + self.b * n
        return self.x

    def __call__(self, v):
        return self.next_val()


class Cosine_Mean_Function(AbstractNoiseProcess):
    r"""Decorator of an random process that models an additive mean term that depends on the gate voltages.
    This term is added to noise values sampled from the decorated noise model

    The mean term of the ith noise element is given as a set of cosine functions:

    :math:`\mu_i(v)= \sum_j a_{ij} \cos(2\pi (w_{ij}^T v+b_{ij}))`

    and the resulting noise is

    :math:`\epsilon(v)_{t}=\mu(v) + \epsilon(v)^D_{t}`

    where :math:`\epsilon(v)^D_{t}` is the decorated noise process.

    The user supplies the weight tensor W with elements :math:`W_{ijk}` so that W[i,j] is the vector :math:`w_{ij}`
    and a matrix a with the amplitude values :math:`a_{ij}`. Finally, b is the matrix of offsets :math:`b_{ij}`, which can be left as None,
    in which case it is sampled uniformly between 0 and 1.
    """

    def __init__(self, noise_model, a, W, b=None):
        """Initialized the cosine mean function.

        Parameters
        ----------
        noise_model: AbstractNoiseProcess
            The noise process with N dimensions to decorrate.
        a: NxM np.array of float
            Amplitudes of the M overlapping cosine functions
        W: NxMxK np.array of float
            K dimensional weights of the M consine functions for the N outputs.
        b: NxM np.array of float or None
            Phases of the M overlapping cosine functions. If none, it is drawn uniformly between 0 and 1.
        """
        super().__init__(noise_model.num_elements)
        self.noise_model = noise_model
        self.a = a
        self.W = W
        self.b = b if b is not None else np.random.uniform(size=a.shape)

    def slice(self, P, m):
        new_W = np.einsum("ijk,kl->ijl", self.W, P)
        new_b = self.b + np.einsum("ijk,k->ij", self.W, m)
        return Cosine_Mean_Function(self.noise_model, self.a, new_W, new_b)

    def start_sequence(self):
        self.noise_model.start_sequence()

    def __call__(self, v):
        # compute noise
        noise_values = self.noise_model(v)

        # add the mean
        activation = 2 * np.pi * (np.einsum("ijk,k->ij", self.W, v) + self.b)
        mean = np.sum(2 * np.pi * self.a * np.cos(activation), axis=1)
        noise_values += mean.reshape(1, -1)

        return noise_values
