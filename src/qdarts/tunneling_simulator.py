import numpy as np
from abc import ABCMeta, abstractmethod
from qdarts.util_functions import find_label
from qdarts.simulator import AbstractPolytopeSimulator


def softmax(v, axis=None):
    max_v = np.max(v)
    y = np.exp(v - max_v)
    return y / np.sum(y, axis)


class AbstractSensorSim(metaclass=ABCMeta):
    """Base class defining the interface for all sensor simulations"""

    def __init__(self, num_sensors):
        """Initialized a sensor configuration with num_sensors sensor dots"""
        self.num_sensors = num_sensors

    @abstractmethod
    def slice(self, P, m):
        """Takes an affine subspace of the simulated model."""
        pass

    @abstractmethod
    def start_measurement(self):
        """Reinitializes the sensor as to generate independent noise samples"""
        pass

    @abstractmethod
    def precompute_sensor_state(self, state, A, b, basis_labels):
        """Allows the sensor to precompute internal information that is valid for a whole ground state polytope.

        This allows the sensor to precompute and cache information that is valid for all voltages v that are inside a
        ground state polytope P. The returned values are cached in the polytope objects of the simulator and
        supplied as sensor_state argument during a call of sample_sensor_equilibrium and sample_sensor_configuration.
        The supplied information provides all information of the basis labels considered by the simulation for P(n),
        and the linear functions defining the facets of P(n) Av+b. Note that as everywhere else, these linear functions define
        energy differences for each considered basis label to the ground state.

        Parameters
        ----------
        state: np.array of ints
            the state of N dots, identifying the ground state polytope for which to generate the sensor state information
        A: LxK np.array of floats
            LxK linear parameters of the energy difference function for the K sensor gates
        b: np.array of floats
            The affine offsets of the energy difference function for the L basis states
        basis_labels: LxN np.array of ints
            The labels of the L basis states
        """
        pass

    @abstractmethod
    def sample_sensor_equilibrium(self, v, H, mixed_state, sensor_state, beta):
        """Computes a noisy average of the sensor response for a given mixed state.

        This is intended to simulate a long (>1mus) time integration of the sensor signal, thus
        we can assume that states are thermalized but the signal is still affected by noise.

        Parameters
        ----------
        v: np.array of floats
            vector of K gate voltages defining the current system
        H: LxL np.array of floats
            Hamiltonian of the system defined by v. Labels and basis are the same as in precompute_sensor_state
        mixed_state: LxL np.array of floats
            Mixed state matrix computed via the Hamiltonian using expm(-beta*H)
        sensor_state:
            Cached information returned by precompute_sensor_state. All information therein are internal to the
            sensor simulator
        beta: float
            scaled inverse temperature parameter
        """
        pass

    @abstractmethod
    def sample_sensor_configuration(
        self, sampled_configuration, v, H, mixed_state, sensor_state, beta
    ):
        """samples a sensor response for a given sampled elecron configuration

        This is intended to simulate a short (<<1mus) time integration of the sensor signal,
        where we can not assume that electrons transitions during the measurement. In this case,
        the user supplied the relevant configuration and the sensor returns a sampled signal for this configuration.
        Care should be taken that the configuration sampled has all information needed in the base to compute the sensor
        signal, e.g., there should be a state with one more or less electrons on each sensor dot.


        Parameters
        ----------
        sampled_configuration: np.array of ints
            vector of N elements describing the sampled electron configuration.
        v: np.array of floats
            vector of K gate voltages defining the current system
        H: LxL np.array of floats
            Hamiltonian of the system defined by v. Labels and basis are the same as in precompute_sensor_state
        mixed_state: LxL np.array of floats
            Mixed state matrix computed via the Hamiltonian using expm(-beta*H)
        sensor_state:
            Cached information returned by precompute_sensor_state. All information therein are internal to the
            sensor simulator
        beta: float
            scaled inverse temperature parameter
        """
        pass


class NoisySensorDot(AbstractSensorSim):
    """Simulates a sensor signal by computing the conductance of the sensor dots.

    This class implements the interface of AbstractSensorSim and for most points, the
    documentation there should be referred to. This simulation combines a simple
    estimation of the conductance g with two noise sources. A fast noise source
    that simulates gaussian white noise that is drawn for each query of the sensor response
    and a slow noise source that models time dependent noise between different invocations
    of the sensor.

    The shape of the simulated sensor peak can be configured ´via config_peak, in which
    height and with of the peak can be adapted. Currently, all sensor dots share these
    parameters.

    The noise can be configured via config_noise. The noise is modeled as an additive
    noise on the sensor peak position in voltage space. Thus, at peaks or valleys, the noise
    is small while on the sides of the peak, where the derivatives are largest, the noise will
    affect measurements the most. Additional signal  noise is then modeled by adding white gaussian noise.
    """

    def __init__(self, sensor_dot_ids):
        super().__init__(len(sensor_dot_ids))
        self.sensor_dot_ids = sensor_dot_ids
        self.g_max = 1
        self.fast_noise_var = 0.0
        self.peak_width_multiplier = 1
        self.slow_noise_gen = None
        self.signal_noise_scale = 0.0

    def config_noise(self, sigma, signal_noise_scale, slow_noise_gen=None):
        self.fast_noise_var = sigma**2
        self.slow_noise_gen = slow_noise_gen
        self.signal_noise_scale = signal_noise_scale
        # initialize noise
        self.start_measurement()

    def config_peak(self, g_max, peak_width_multiplier):
        self.g_max = g_max
        self.peak_width_multiplier = peak_width_multiplier

    def start_measurement(self):
        if self.slow_noise_gen is not None:
            self.slow_noise_gen.start_sequence()

    def slice(self, P, m):
        # if there is no slow noise, there is nothing to slice
        if self.slow_noise_gen is None:
            return self

        # otherwise create a copy of this with a sliced slow noise model
        sliced_sensor_dot = NoisySensorDot(self.sensor_dot_ids)
        sliced_sensor_dot.g_max = self.g_max
        sliced_sensor_dot.fast_noise_var = self.fast_noise_var
        sliced_sensor_dot.peak_width_multiplier = self.peak_width_multiplier
        sliced_sensor_dot.signal_noise_scale = self.signal_noise_scale
        sliced_sensor_dot.slow_noise_gen = self.slow_noise_gen.slice(P, m)
        return sliced_sensor_dot

    def precompute_sensor_state(self, state, A, b, labels):
        sensor_state = {}
        for i, sensor_id in enumerate(self.sensor_dot_ids):
            labels_nosens = np.delete(labels, sensor_id, axis=1)
            labels_unique, inverse_index = np.unique(
                labels_nosens, return_inverse=True, axis=0
            )

            labels_sens = labels[:, sensor_id]
            sorted_ind = np.lexsort((labels_sens, inverse_index))

            relevant_label_indices = []
            prev = []
            nex = []
            cur = -1
            last = None
            last_2 = None
            for ind in sorted_ind:
                lab = labels_nosens[ind]
                if np.any(lab != cur):
                    cur = lab
                    last = None
                    last_2 = None
                else:
                    if last_2 is not None:
                        relevant_label_indices.append(last)
                        prev.append(last_2)
                        nex.append(ind)
                    last_2 = last
                    last = ind
            terms = np.array(relevant_label_indices, dtype=int)
            prev = np.array(prev, dtype=int)
            nex = np.array(nex, dtype=int)
            terms_labels = labels[terms, :]
            sensor_state[sensor_id] = (terms, prev, nex, terms_labels)
        return sensor_state

    def _precompute_g(self, v, H, sensor_state, beta):
        results = np.zeros(len(self.sensor_dot_ids))
        gs = {}
        slow_noise = np.zeros((results.shape[0], 1))
        if self.slow_noise_gen is not None:
            slow_noise = self.slow_noise_gen(v)
        for i, sensor_id in enumerate(self.sensor_dot_ids):
            terms, neighbour_prev, neighbour_next, _ = sensor_state[sensor_id]

            # compute sensor detuning between every state and their neighbour
            eps_prev = np.abs(np.diag(H)[terms] - np.diag(H)[neighbour_prev])
            eps_next = np.abs(np.diag(H)[terms] - np.diag(H)[neighbour_next])
            eps = np.minimum(eps_prev, eps_next)
            # add noise
            eps = eps + slow_noise[i : i + 1]
            if self.fast_noise_var > 0:
                fast_noise = np.random.randn(*eps.shape) * np.sqrt(self.fast_noise_var)
                eps += fast_noise
            eps *= beta

            # we approximate the logistic peak of g with the peak of a normal distribution of same width
            # todo: we can fully go back to the logistic peak
            var_logistic = (1 / 0.631 * self.peak_width_multiplier) ** 2

            def norm_pdf(x, mu, var):
                return (
                    1 / np.sqrt(2 * np.pi * var) * np.exp(-((x - mu) ** 2) / (2 * var))
                )

            gs[sensor_id] = self.g_max * 4 * norm_pdf(0, eps, var_logistic)
        return gs

    def sample_sensor_equilibrium(self, v, H, mixed_state, sensor_state, beta):
        results = np.zeros(len(self.sensor_dot_ids))
        gs = self._precompute_g(v, H, sensor_state, beta)

        for i, sensor_id in enumerate(self.sensor_dot_ids):
            terms, neighbour_prev, neighbour_next, _ = sensor_state[sensor_id]
            g = gs[sensor_id]
            p = np.diag(mixed_state)[terms]
            results[i] = np.sum(p * g) / np.sum(p)
        var_logistic = (1 / 0.631 * self.peak_width_multiplier) ** 2
        scale = (
            self.g_max * self.signal_noise_scale * 4 / np.sqrt(2 * np.pi * var_logistic)
        )
        results += scale * np.random.randn(len(results))
        return results

    def sample_sensor_configuration(
        self, sampled_configuration, v, H, mixed_state, sensor_state, beta
    ):
        results = np.zeros(len(self.sensor_dot_ids))
        gs = self._precompute_g(v, H, sensor_state, beta)

        for i, sensor_id in enumerate(self.sensor_dot_ids):
            terms, neighbour_prev, neighbour_next, terms_labels = sensor_state[
                sensor_id
            ]
            label_pos = find_label(terms_labels, sampled_configuration)
            results[i] = gs[sensor_id][label_pos]
        var_logistic = (1 / 0.631 * self.peak_width_multiplier) ** 2
        scale = (
            self.g_max * self.signal_noise_scale * 4 / np.sqrt(2 * np.pi * var_logistic)
        )
        results += scale * np.random.randn(len(results))
        return results


class TunnelBarrierModel:
    """Model of the tunnel barriers of a device

    This class defines a mapping between gate voltages of the device and
    the tunnel coupling between the dots. To be more exact, the tunnel
    coupling between dots i, and j and the gate voltages v is given by

    T_ij = exp(W_ij^Tv+b_ij)

    where W_ij is a vector of couplings and b_ij is an offset.
    """

    def __init__(self, gate_offsets, gate_levers=None):
        """Creates a tunnel barrier model.

        Parameters
        ----------
        gate_offsets : NxN np.array of floats
            the offsets b_ij provided as matrix.
        gate_levers : NxNxK np.array of floats or None
            Here, K is the number of plunger gates. The first two indices describe the index of the tunnel coupling matrix ij.
            If None, it is assumed to be 0.
        """
        self.gate_offsets = gate_offsets
        self.gate_levers = gate_levers

    def slice(self, P, m):
        """Takes an affine subspace of the simulated model.

        Let v=Pv'+m. Computes a new parameterization such, that
        T_ij = exp(W'_ij^Tv'+b'_ij)

        Parameters
        ----------
        P : KxM np.array of floats
            The linear transformation matrix
        m : np.array of floats or None
            Array of size K storing the affine offset.
        """
        if self.gate_levers is None:
            return self
        sliced_levers = self.gate_levers.reshape(-1, self.gate_levers.shape[-1]) @ P

        sliced_levers = sliced_levers.reshape(
            self.gate_levers.shape[0], self.gate_levers.shape[1], P.shape[1]
        )
        m_applied = self.gate_levers.reshape(-1, self.gate_levers.shape[-1]) @ m
        sliced_offsets = self.gate_offsets + m_applied.reshape(
            self.gate_levers.shape[0], self.gate_levers.shape[1]
        )
        return TunnelBarrierModel(sliced_offsets, sliced_levers)

    def get_tunnel_matrix(self, v):
        """Returns the tunnel matrix for a given gate voltage

        Parameters
        ----------
        v : np.array of floats or None
            Array of size K storing the gate voltages

        """
        if self.gate_levers is None:
            return np.exp(self.gate_offsets)
        else:
            barrier_potentials = self.gate_levers @ v
            return np.exp(barrier_potentials + self.gate_offsets)


class LocalSystem:
    """Class describing a quantum system defined by the gate voltages of a simulated device.

    For a given set of gate voltages, the simulator first computes a set of core states that are most
    likely relevant for the computation of the hamiltonian and then extends it by adding additional
    states. These are then used to define a basis of the vector space for the Hamiltonian, which
    is then used to compute the mixed state. Finally, the mixed state is then used to simulate a sensor signal.

    This class stores all this information and includes some minimal tools to query information on
    different sub-bases. This class is tightly coupled to tunneling_simulator.

    Attributes
    ----------

    v: np.array of floats
        gate voltages that define the parameters of this system
    state: np.array of ints
        the ground state configuration of v
    beta: float
        the scaled inverse temperature 1/k_bT
    H: LxL np.array of floats
        Hamiltonian over the subspace spanned by the L basis state of the extended basis.
        See methods basis_labels and core_basis_indices
    """

    def __init__(self, v, H, state, sim):
        """Creates the LocalSystem.

        This is an internal function used by the tunneling simulator.
        """
        self.v = v
        self.H = H
        self.state = state.copy()
        self._sim = sim
        self.beta = self._sim.beta

    def _compute_mixed_state(self, H):
        diffs = np.diag(H) - np.min(np.diag(H))
        sel = np.where(diffs < 2 * self._sim.poly_sim.get_maximum_polytope_slack())[0]
        H_sel = H[:, sel][sel, :]

        eigs, U = np.linalg.eigh(H_sel)
        ps = softmax(-eigs * self.beta)
        rho_sel = U @ np.diag(ps) @ U.T

        rho = np.zeros(H.shape)
        indizes = H.shape[0] * sel[:, None] + sel[None, :]
        np.put(rho, indizes.flatten(), rho_sel.flatten())
        return rho

    @property
    def mixed_state(self):
        """Computes an approximate mixed state matrix over the full basis.

        the mixed state matrix, defined as expm(-beta*H)

        Note that this function approximated the true mixed state matrix by inly taking basis eleemnts into account
        that have a small energy difference to the ground state. This is a multiple of the polytope slack used by the
        capacitive simulation.
        """
        return self._compute_mixed_state(self.H)

    def compute_mixed_state_of_subset(self, subset_indices):
        """Computes the mixed state for a subset, ignoring the existance of any other state entirely.

        The result is a KxK matrix where K is the length of subset_indices.
        This function is not equivalent to selecting a subset of mixed_state, since this assumes that
        the states not referenced by subset_indices are ruled out for some other reason, i.e., they are
        assigned probability 0 and probabilities are renormalized to sum to 1 over the elements in the subset.


        Parameters
        ----------
        subset_indices : np.array of ints
            The L' indices into the basis element matrix as returned by basis_labels
        m : np.array of floats or None
        """

        return self._compute_mixed_state(self.H[subset_indices, :][:, subset_indices])

    @property
    def basis_labels(self):
        """The labels of the basis elements, indentified by their ground state electron configuration"""
        return (
            self._sim.boundaries(self.state).additional_info["extended_polytope"].labels
        )

    @property
    def core_basis_indices(self):
        """Indices into basis_labels that define the subset of core basis elements."""
        return (
            self._sim.boundaries(self.state)
            .additional_info["extended_polytope"]
            .core_basis_indices
        )

    def sample_sensor_equilibrium(self):
        """Samples a boisy averaged sensor response from the current system over all basis elements.

        This returns the average signal with added sensor noise. This is an approximation to long
        average measurements at a single point.
        """
        sensor_state = self._sim.boundaries(self.state).additional_info["sensor_state"]
        return self._sim.sensor_sim.sample_sensor_equilibrium(
            self.v, self.H, self.mixed_state, sensor_state, self.beta
        )

    def sample_sensor_configuration(self, sampled_configuration):
        """Samples the sensor signal for a given sampled electron configuration.

        For a short time simulation it is more prudent to externally sample a state from the basis and then generate a sensor signal
        from it. This function allows this. Note that only selecting states from the set of core_basis_indices is safe as otherwise
        the sensor might miss information required to correctly compute the response.

        Parameters
        ----------
            sampled_configuration: list of ints
                the sampled state for which to generate the sensor response.
        """
        sensor_state = self._sim.boundaries(self.state).additional_info["sensor_state"]
        return self._sim.sensor_sim.sample_sensor_configuration(
            sampled_configuration,
            self.v,
            self.H,
            self.mixed_state,
            sensor_state,
            self.beta,
        )


class ApproximateTunnelingSimulator(AbstractPolytopeSimulator):
    """Simulator for approximate charge tunneling in a quantum dot device.

    The simulator extends the supplied capacitive simulation by creating a Hamiltionian H,
    where on the diagonals are the capacitive energies of the simualation, while the off-diagonals
    have added tunnel coupling parameters. Locally the hamiltonian is approximated via L basis states,
    where each state is an electron configurtion on the dots. This mixed state is then used to create a sensor simulation.

    It is possible to query the state of single hamiltonian, their mixed state and their sensor simulation via the class
    LocalSystem, returned by compute_local_system, but the primary use of tis class lies in its ability to compute
    1D or 2D sensor scans via sensor_scan and sensor_scan_2D.

    For computing the tunnel coupling parameters, this class can make use of an additional Tunnel barrier simulation, but it
    is also possible to just supply a NxN constant matrix of tunnel couplings between all D dots in the array.

    Finally, the class follows the interface of AbstractPolytopeSimulator, which means it is possible to directly query the information
    of the underlying polytopes of the simulation. This is there to unify slicing between simulators.

    Implementation details:

    The basis used for a gate voltage v is queried by finding its ground states n and then the facets of the
    ground state polytope P(n) create the basis. Thus, this basis becomes extended as the slack variable in the underlying
    capacitance simulation is increased. This is called the core state set.
    Additionally, the simulation allows to add additional states. For example, for most sensor simulations to work, we also need
    other higher energy states to compute correct conductance. These additional states can be added by modifying the vector
    num_additional_neighbours. if the ith element in this vector is R>0, and s is a state in the core basis, then
    the extended basis will also include the states :math:`s+ke_i` where :math:`|k|<=R` and :math:`e_i` is the ith basis vector.

    The tunnel couplings T are included into the Hamiltonian the following way: let :math:`s_i` and :math:`s_j` be two states in the basis of the Hamiltonian
    that differ only in the value of the electron configuration at dots i and j.
    More exactly, we have that :math:`s_i` and :math:`s_j` are related by moving an electron from state :math:`s_i` to :math:`s_j` or vice versa.
    Let :math:`H_{kl}` be the off-diagonal matrix element of those states. Then we have :math:`H_{kl} = T_{ij}`.
    In all other cases, tunnel coupling is 0.

    The mixed state is then again computed approximately, for more info on that, see documentation of LocalSystem.

    The sensor signal of the computed mixed state is computed via the sensor_sim.

    Attributes
    ----------

    beta: float
        Scaled inverse temperature 1/k_BT
    T: float
        Temperature
    poly_sim:
        the capacitive simulation object
    barrier_sim:
        the barrier simulation object. Note that even if the supplied object to init was a matrix, this will be a TunnelBarrierModel.
    sensor_sim:
        the sensor simulation object
    num_additional_neighbours: np.array of ints
        for each dot defines how many additional states should be added for each state in the core basis. This is done
        by adding or subtracting electrons on the ith element where the maximum is given by the ith element of num_additional_neighbours.
        We advise to set this to 2 for sensor dots. Note that computation time can quickly explode when increasing this parameter.
        Outside of sensor dots, we advise therefore to increase the slack in the capacitive simulation.
    """

    def __init__(self, polytope_sim, barrier_sim, T, sensor_sim):
        """Creates a tunneling simulation

        Parameters
        ----------
        polytope_sim:
            capacitance simulator object that computes ground state polytopes and capacitive energy differences
        barrier_sim: Object or Matrix
            Either a DxD basis that describes a constant tunnel coupling between all D dots. Note that the diagonal of this matrix is zero.
            Alternatively an object with a method barrier_sim.get_tunnel_matrix(v) returning a DxD matrix, and which supports the slice operation.
        T: float
            Temperature in Kelvin. Good values are < 0.1
        sensor_sim: Derived from AbstractSensorSim
            A sensor simulation that follows the interface of AbstractSensorSim and which computes the sensor signal.
        """
        self.poly_sim = polytope_sim
        # compatibility to earlier code that uses matrices
        if isinstance(barrier_sim, np.ndarray):
            self.barrier_sim = TunnelBarrierModel(np.log(barrier_sim + 1.0e-20))
        else:
            self.barrier_sim = barrier_sim
        eV = 1.602e-19
        kB = 1.380649e-23 / eV
        self.beta = 1.0 / (kB * T)
        self.T = T
        self.sensor_sim = sensor_sim

        # clean up potentially stored conflicting data
        for poly in self.poly_sim.cached_polytopes():
            poly.additional_info.pop("features_out_info", None)

        self.num_additional_neighbours = np.zeros(self.poly_sim.num_dots, dtype=int)

        super().__init__(self.poly_sim.num_dots, self.poly_sim.num_inputs)

    def slice(self, P, m, proxy=False):
        """Restricts the simulator to the affine subspace v=m+Pv'

        Computes the slice through the simulated device by setting v=m+Pv', where v is the plunger gate voltages of the
        original device and v' is the new coordinate system. This is implemented here by slicing all the different parts
        of the simulation, capacitance model, barrier model and sensor model.

        Parameters
        ----------
        P : MxK np.array of floats
            The linear coefficient matrix. Here M is the number of voltage elements in v in the full simulation
            and K the dimensionality of the subspace.
        m: offset of the affine trnsformation.
        proxy: bool
            Whether a proxy is returned. A proxy shares the cache, if possible. This is the case when P is invertible,
            especially this entails M=K. If cache sharing is possible, the simulation computes the original polytope and then
            applies the affine transformation. This can reduce run time a lot if several slices need to be computed for the
            same simulation.

        Returns
        -------
        A simulator object describing the simulation on the affine subspace. The current simulation object remains unchanged.
        """
        sliced_poly_sim = self.poly_sim.slice(P, m, proxy)
        sliced_barrier_sim = self.barrier_sim.slice(P, m)
        sliced_sensor_sim = self.sensor_sim.slice(P, m)
        sliced_tunneling_sim = ApproximateTunnelingSimulator(
            sliced_poly_sim, sliced_barrier_sim, self.T, sliced_sensor_sim
        )
        sliced_tunneling_sim.num_additional_neighbours = (
            self.num_additional_neighbours.copy()
        )
        return sliced_tunneling_sim

    def _compute_tunneling_op(self, state_list):
        """Computes the mapping between tunnel coupling and hamiltonian off diagonal elements
        and also also computes a multiplicative weight for each tunnel coupling based on the number
        of affected electrons during the state transitions.

        We currently only add tunnel coupling between two states n,m if they describe the transition of a single
        electron between two dots i and j. In this case the tunnel coupling is w*T_ij where w=1 if the
        total number of electrons on dots i and j is odd, otherwise w=sqrt(2).
        Parameters
        ----------
        state_list: list of vectors of ints
            The list of states that describe a subset of the fokh basis of the Hamiltonian.

        Returns
        -------
        TOp: a mapping TOP(n,m)=i*num_dots+j, the index in the flattened matrix of tunnel couplings
        TOpW: weight matrix W(n,m).

        """
        N = state_list.shape[0]
        n_dots = state_list.shape[1]
        TOp = np.zeros((N, N), dtype=int)
        TOpW = np.ones((N, N))

        sums = np.sum(state_list, axis=1)
        for i, s1 in enumerate(state_list):
            for j, s2 in zip(range(i + 1, len(state_list)), state_list[i + 1 :]):
                if sums[i] != sums[j]:
                    continue

                if np.sum(np.abs(s1 - s2)) == 0:
                    continue
                abs_diff = np.abs(s1 - s2)
                if np.sum(abs_diff) != 2:
                    continue

                # compute lookup indices in tunneling strength matrix
                idxs = np.where(abs_diff > 0)[0]
                if len(idxs) == 1:
                    ind = idxs[0] * n_dots + idxs[0]
                else:
                    ind = idxs[0] * n_dots + idxs[1]
                TOp[i, j] = ind
                TOp[j, i] = ind

                # compute weight. If the total number of electrons on the affected dots
                # is even, the tunneling strength is multiplied by sqrt(2)
                pos_changes = s1 != s2
                num_electrons_affected = np.sum(pos_changes * s1)
                if num_electrons_affected % 2 == 0:
                    TOpW[i, j] = np.sqrt(2)
                    TOpW[j, i] = TOpW[i, j]
        return TOp, TOpW

    def _create_state_list(self, state, direct_neighbours):
        """Creates the extended basis"""
        state_list = np.vstack([direct_neighbours, [np.zeros(len(state), dtype=int)]])

        additional_states = []
        for i in range(self.poly_sim.num_dots):
            e_i = np.eye(1, self.poly_sim.num_dots, i, dtype=int)
            for k in range(1, 1 + self.num_additional_neighbours[i]):
                additional_states.append(state_list + k * e_i)
                additional_states.append(state_list - k * e_i)

        if len(additional_states) > 0:
            for add in additional_states:
                state_list = np.vstack([state_list, add])
            state_list = np.unique(state_list, axis=0)
        state_list += state[None, :]
        state_list = state_list[np.all(state_list >= 0, axis=1)]

        # mark the subset of original polytope transitions + the current state

        core_index_set = [int(find_label(state_list, state)[0])]
        for core_transition in direct_neighbours:
            core_index_set.append(
                int(find_label(state_list, core_transition + state)[0])
            )
        return state_list, np.array(core_index_set, dtype=int)

    def boundaries(self, state):
        """
        Returns the polytope P(n) of a given state n with all its boundaries, labels and meta information.

        If the polytope is not cached, it needs to be computed. This can take some time for large devices.

        Parameters
        ----------
        state: list of ints
            The state n for which to compute the polytope P(n)

        Returns
        -------
        The polytope P(n)
        """
        state = np.asarray(state)
        polytope = self.poly_sim.boundaries(state)
        # cache features_out info in polytope structure
        if "extended_polytope" not in polytope.additional_info.keys():
            # create a list of all neighbour states of interest for use in the Hamiltonian
            state_list, polytope_base_indx = self._create_state_list(
                state, polytope.labels
            )

            # create full set of transition equations
            A, b = self.poly_sim.compute_transition_equations(state_list, state)

            TOp, TOpW = self._compute_tunneling_op(state_list)
            extended_polytope = type("", (object,), {})()
            extended_polytope.A = A
            extended_polytope.b = b
            extended_polytope.TOp = TOp
            extended_polytope.TOpW = TOpW
            extended_polytope.labels = state_list
            extended_polytope.core_basis_indices = polytope_base_indx
            polytope.additional_info["extended_polytope"] = extended_polytope

            # also compute the sensor info
            polytope.additional_info["sensor_state"] = (
                self.sensor_sim.precompute_sensor_state(state, A, b, state_list)
            )
        return polytope

    def _create_hamiltonian(self, v, A, b, TOp, TOpW):
        """Computes the hamiltonian at the given gate voltages"""
        tunnel_matrix = self.barrier_sim.get_tunnel_matrix(v)
        N = A.shape[0]
        energy_diff = -(A @ v + b)
        if tunnel_matrix is None:
            return np.diag(energy_diff)
        else:
            t_term = ((tunnel_matrix.reshape(-1)[TOp.reshape(-1)]).reshape(N, N)) * TOpW
            return np.diag(energy_diff) - t_term

    def get_displacement(self, H, dH):
        """Computes the displacement of the ground state"""
        ind0 = np.argsort(np.diag(H))[:2]
        tc = H[ind0[0], ind0[1]]
        eps = H[ind0[0], ind0[0]] - H[ind0[1], ind0[1]]

        dind0 = np.argsort(np.diag(dH))[:2]
        dtc = dH[dind0[0], dind0[1]]
        deps = dH[dind0[0], dind0[0]] - dH[dind0[1], dind0[1]]

        return deps / np.sqrt(dtc**2 + deps**2) - eps / np.sqrt(tc**2 + eps**2)

    def compute_local_system(self, v, state, search_ground_state=True):
        """Computes a full description of the local quantum system and returns the LocalSystem object.

        This is a full locla simulation of the device and can be used to query sensor values but also the mixed state matrix.
        See LocalSystem for more info.

        Note that unlike in most other places, v does not need to belong to the ground state polytope of state.
        This might be useful for the computation of signals in which the device is far out of equilibrium.

        Parameters
        ----------
        v: np.array of floats
            The vector of gate voltages of the device
        state: np.array of ints
            The ground state polytope relative to which the local system is computed. This is in most cases the ground state.
        search_ground_state: bool
            If True, verifies that state is the ground state of v and searches it otherwise. If you know that this is the case,
            you can safely set it to False for a speed-up. In the general case, setting this to false will compute the
            LocalSystem relative to a different basis state.
        """
        if search_ground_state:
            state = self.poly_sim.find_state_of_voltage(v, state_hint=state)
        polytope = self.boundaries(state)
        extended_polytope = polytope.additional_info["extended_polytope"]
        H = self._create_hamiltonian(
            v,
            extended_polytope.A,
            extended_polytope.b,
            extended_polytope.TOp,
            extended_polytope.TOpW,
        )
        system = LocalSystem(v, H, state, self)
        return system

    def sensor_scan(
        self,
        v_start,
        v_end,
        resolution,
        v_start_state_hint,
        cache=True,
        start_new_measurement=True,
        insitu_axis=None,
    ):
        """Computes a 1D sensor ramp scan.

        Computes a linear set of points between v_start and v_end and for each point computes the sensor signal.
        To be more exact, for each point, the ground state polytope is computed which is then used to define the local_system.
        Returns the sensor signal for each sensor and dot

        Parameters
        ----------
        v_start: np.array of floats
             Vector of gate voltages of the device describing the first measurement point
        v_end: np.array of floats
             Vector of gate voltages of the device describing the last measurement point
        resolution: int
            number of measured points on the linear scan between v_start and v_end, including both end points.
        v_start_state_hint: np.array of int
            Guess for the state n for which holds that v_start is element of P(n). The simulator will use this
            guess as a starting point for the search of the correct state if this guess is wrong. Note that P(n)
            must intersect with the affine slice, if slicing was used.
        cache: bool
            Whether the simulation should try to cache the computed polytopes. This might lead to a slower computation time
            for a scan compared to not using caching, but consecutive scans with similar ranges tend to be quicker.
        start_new_measurement: bool
            Whether the seimulated sensor measurement should be independent of any previous measurements.
        insitu_axis: list or None
            The axis along which the modulated signal is applied, insitu_axis @ plane_axes. If none, the code uses standard sensor dot approach. If a list it computes changes in quantum capacitance
        """
        # prepare start state
        state = self.poly_sim.find_state_of_voltage(
            v_start, state_hint=v_start_state_hint
        )

        P = (v_end - v_start).reshape(-1, 1)
        if cache:
            sim_slice = self.slice(P, v_start, proxy=cache)
        else:
            sim_slice = self

        if start_new_measurement:
            sim_slice.sensor_sim.start_measurement()

        if insitu_axis is None:
            values = np.zeros((resolution, self.sensor_sim.num_sensors))
        else:
            values = np.zeros((resolution, 1))
        for i, v0 in enumerate(np.linspace([0.0], [1.0], resolution)):
            if cache:
                v = v0
            else:
                v = v_start + P @ v0
            if not sim_slice.poly_sim.inside_state(v, state):
                state = sim_slice.poly_sim.find_state_of_voltage(v, state_hint=state)

            system = sim_slice.compute_local_system(v, state, search_ground_state=False)
            if insitu_axis is None:
                values[i] = system.sample_sensor_equilibrium()
            else:
                # inisitu reflecometry
                dv = 0.0001
                system2 = sim_slice.compute_local_system(
                    np.array(v) + dv * np.array(insitu_axis),
                    state,
                    search_ground_state=False,
                )
                a = sim_slice.get_displacement(system.H, system2.H)
                values[i] = a
        return values

    def sensor_scan_2D(
        self,
        P,
        m,
        minV,
        maxV,
        resolution,
        state_hint_lower_left,
        cache=True,
        insitu_axis=None,
    ):
        """Computes the sensor signal on a 2D grid of points.

        For the exact computation of points, see sensor_scan.

        The grid is defined the following way: Let w_ij be a 2D vector that is part of a regular
        rectangular grid spanned by the lower left corner given by minV and the upper right corner given
        by maxV and let (m,n) be the number of points in both grid directions. We have that w_00=minV and w_m-1,n-1=maxV.

        This grid is then affinely transformed into the K-dimensional space of gate vectors via
        v_ij = m+ P w_ij

        and thus P must be a Kx2 matrix and m a K-vector.

        Parameters
        ----------
        P: Kx2 np.array of floats
            linear transformation of grid-points into the K-dimensional voltage space
        m: np.array of floats:
            affine offset of the grid
        minV: np.array of floats
            2D vector describing the minimum value of the grid points
        maxV: np.array of floats
            2D vector describing the maximum value of the grid points
        resolution: int or list of ints
            if integer, describes the same number of points in both grid directions. If a list of 2 elements,
            describes the number of points along each axes of the grid.
        state_hint_lower_left: np.array of int
            Guess for the state n for point described by the grid position minV. The simulator will use this
            guess as a starting point for the search of the correct state if this guess is wrong. Note that P(n)
            must intersect with the affine slice, if slicing was used.
        cache: bool
            Whether the simulation should try to cache the computed polytopes. This might lead to a slower computation time
            for a scan compared to not using caching, but consecutive scans with similar ranges tend to be quicker.
        insitu_axis: list or None
            The axis along which the modulated signal is applied. If none, the code uses standard sensor dot approach. If a list it computes changes in quantum capacitance.
        """
        if P.shape[1] != 2:
            raise ValueError("P must have two columns")
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        # obtain initial guess for state
        line_start = self.poly_sim.find_state_of_voltage(
            m + P @ minV, state_hint=state_hint_lower_left
        )
        # now slice down to 2D for efficiency
        sim_slice = self.slice(P, m, proxy=cache)
        if insitu_axis is None:
            sim_slice.sensor_sim.start_measurement()
            values = np.zeros(
                (resolution[0], resolution[1], self.sensor_sim.num_sensors)
            )

        else:
            values = np.zeros((resolution[0], resolution[1], 1))

        for i, v2 in enumerate(np.linspace(minV[1], maxV[1], resolution[1])):
            v_start = np.array([minV[0], v2])
            v_end = np.array([maxV[0], v2])
            line_start = sim_slice.poly_sim.find_state_of_voltage(
                v_start, state_hint=line_start
            )
            values[:,i] = sim_slice.sensor_scan(
                v_start,
                v_end,
                resolution[0],
                line_start,
                cache=False,
                start_new_measurement=False,
                insitu_axis=insitu_axis,
            )
        return values
