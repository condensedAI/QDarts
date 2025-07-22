import numpy as np
from qdarts.util_functions import is_invertible_matrix
from abc import ABCMeta, abstractmethod


class AbstractPolytopeSimulator(metaclass=ABCMeta):
    """Base class for all simulation objects that can compute and return polytopes.

    The class only has a single method boundaries which returns the boundary description of the polytope
    with meta information as well as two attributes:

    Attributes
    ----------

    num_dots: int
        number of dots in the device, i.e., number of entries in the state vector of the polytope
    num_inputs: int
        number of gate voltages in the device. The polytope lives in a space that is num_inputs dimensonal.
    """

    def __init__(self, num_dots, num_inputs):
        self.num_dots = num_dots
        self.num_inputs = num_inputs

    @abstractmethod
    def boundaries(self, state):
        """
        Returns the polytope P(n) of a given state n with all its boundaries, labels and meta information.

        Parameters
        ----------
        state: list of ints
            The state n for which to compute the polytope P(n)

        Returns
        -------
        The polytope P(n)
        """
        pass

    @abstractmethod
    def slice(self, P, m, proxy=False):
        """Restricts the simulator to the affine subspace v=m+Pv'

        Computes the slice through the device by setting v=m+Pv', where v is the plunger gate voltages of the
        original device and v' is the new coordinate system. Must be implemented by derived classes.

        Parameters
        ----------
        P : MxK np.array of floats
            The linear coefficient matrix. Here M is the number of voltage elements in v in the full simulation
            and K the dimensionality of the subspace.
        m: offset of the affine trnsformation.
        proxy: bool
            Whether a proxy is returned. A proxy can share computation between instances, if supported by the derived class

        Returns
        -------
        A simulator object describing the simulation on the affine subspace. The current simulation object remains unchanged.
        """
        pass


class AbstractCapacitiveDeviceSimulator(AbstractPolytopeSimulator):
    """Base class for all objects that create device simulations from a Capacitive Model.

    This class includes all tools to compute and cache polytopes from the provided capacitive model.
    Polytopes are queried using a call to boundaries() which queries the internal cache and then
    computes the polytope on demand.

    The computed polytope P(n) is the set of voltages v for which n is the ground state of the capacitance
    energy function E(v,n), i.e., n=min_n' E(v,n'). A facet of the polytope is given by the equality
    E(v,n')-E(v,n) = 0 for a suitable choice of n'. As a result, when shooting a ray through
    a facet of the polytope created by state n', there is a transition from state n->n' in the ground state.

    The computation of the polytope discards by default all states and inequalitis that do not form a facet of P(n).
    However, this can be relaed by allowing a maximum slack, which also keeps facets for which
    min_v E(v,n')-E(v,n) < max_slack, where v is restricted to elements in the polytope P(n). i.e., max slack keeps facet
    in which the energy difference is small.

    This class supports slicing of the voltage space into affine subspaces.
    """

    def __init__(self, capacitance_model):
        super().__init__(capacitance_model.num_dots, capacitance_model.num_inputs)
        self.capacitance_model = capacitance_model
        self.cache = {}

    @abstractmethod
    def slice(self, P, m, proxy=False):
        """Restricts the simulator to the affine subspace v=m+Pv'

        Computes the slice through the device by setting v=m+Pv', where v is the plunger gate voltages of the
        original device and v' is the new coordinate system. Must be implemented by derived classes.

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
        pass

    @abstractmethod
    def compute_polytope(self, state):
        """
        Computes the polytope for a given state.

        Is implemented by the derived class and called when the polytope for a state is not found in cache.

        Parameters
        ----------
        state : list of ints
            the state identifying the polytope


        Returns
        -------
        A Polytope object containing the full computed polytope.
        """
        pass

    def compute_transition_equations(self, state_list, state_from):
        """Computes the energy difference equations from target states to all states in the list.

        For a given state and list of neighbour states, computes the linear equations Av+b that compute the energy differences
        Between the target state_from and the other states. That is, if state_list contains a list of states n', this
        function constains linear equations E(v,n')-E(v,state_from)

        Parameters
        ----------
        state_list: numpy array of ints of size NxK
            A list containing N states for which to compute the energy differences

        Returns
        -------
        A: NxK np.array, containing the linear factors for each state in state_list
        b: np.array, containing the N offsets, one for each equation.
        """
        return self.capacitance_model.compute_transition_equations(
            state_list, state_from
        )

    @abstractmethod
    def get_maximum_polytope_slack(self):
        """Returns the maximum slack value for inclusing of a facet into the polytope.

        Returns the maximum energy distance the closest point of a transition can have to the polytope
        before it is discarded. Setting to 0 means that only transitions that actually touch the polytope
        are kept.
        """
        pass

    @abstractmethod
    def set_maximum_polytope_slack(self, maximum_slack):
        """Sets the maximum slack value for inclusing of a facet into the polytope.

        Sets the maximum distance the closest point of a transition can have to the polytope
        before it is discarded. Setting to 0 means that only transitions that actually touch the polytope
        are kept.

        Note that changing this value clears the cache.
        """
        pass

    def cached_polytopes(self):
        """
        Returns a sequence including all computed and cached polytopes for inspection and modification.
        """
        return self.cache.values()

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
        # Convert to array to be sure
        state = np.asarray(state).astype(int)

        # lookup key of this state
        dict_key = tuple(state.tolist())
        # See if we already have this key in our prepared list
        if dict_key not in self.cache.keys():
            self.cache[dict_key] = self.compute_polytope(state)

        # obtain polyope from dict
        polytope = self.cache[dict_key]

        # slice is allowed to be lazy but then we need to verify the polytope now.
        if polytope.must_verify:
            polytope = self.capacitance_model.verify_polytope(
                polytope, self.get_maximum_polytope_slack()
            )
            self.cache[dict_key] = polytope

        return polytope

    def inside_state(self, v, state):
        """Returns true if a point v is fully within the polytope of a given state.

        Parameters
        ----------
        state: list of ints
            The state n identifying the polytope P(n)
        v: np.array of floats
            The point v


        Returns
        -------
        The truth value of whether v is element of P(n)
        """
        polytope = self.boundaries(state)
        if len(polytope.labels) == 0:
            return False
        f = polytope.A @ v + polytope.b
        return np.all(f < 1.0e-8)

    def find_boundary_intersection(
        self,
        old_v: np.ndarray,
        new_v: np.ndarray,
        state: np.ndarray,
        epsilon: float = 1.0e-6,
        deep_search: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes an intersection of a ray with the boundary of a polytope and computes the new state

        For a given state and a voltage old_v within the polytope of this state and a point new_v outside the polytope,
        computes the intersection of the ray old_v+t*(new_v-old_v) with the boundary of the polytope.
        the intersection point and new target state is computed.


        Parameters
        ----------
        old_v: np.array of floats
            A point within the current polytope
        new_v: np.array of floats
            Another point on the ray
        state: list of ints
            The ground state n of old_v. It is assumed that v is element of P(n)
        epsilon: float
            slack value added to the ray t to ensure that the point is numerically clearly outside the polytope.
        deep_search: bool
            whether an iterative search is performed for the new point in case none of the direct neighbours of the polytope match.
            If false, will throw an exception in that case. An exception is also raised when the deep search failed.


        Returns
        -------
        The first intersection point of the ray with the polytope, together with the new state
        """
        if not self.inside_state(old_v, state):
            raise ValueError("old_v must be in the provided state.")

        polytope = self.boundaries(state)

        direction = new_v - old_v
        direction /= np.linalg.norm(direction)
        selection = np.where(polytope.slacks<1.e-8)[0]
        

        A_line = polytope.A[selection] @ direction
        b_line = polytope.b[selection] + polytope.A[selection] @ old_v
        positive = np.where(A_line > 0)[0]
        ts = -b_line[positive] / A_line[positive]
        transition_idx = np.argmin(ts)
        selection = selection[positive]

        # construct point of closest hit
        transition_state = state + polytope.labels[selection[transition_idx]]
        v_intersect = old_v + (1 + epsilon) * ts[transition_idx] * direction
        if self.inside_state(v_intersect, transition_state):
            return transition_state, v_intersect

        # the new point might have went through a corner, so we check all states whose transitions are now violated

        rel_energy = polytope.A @ v_intersect + polytope.b
        idx_order = np.argsort(rel_energy)
        for idx in idx_order:
            # pass 1: ignore transitions that don't touch the polytope.
            if polytope.slacks[idx] > 1e-6:
                continue
            if rel_energy[idx] < -1.0e-8:
                continue
            transition_state = state + polytope.labels[idx]
            if self.inside_state(v_intersect, transition_state):
                return transition_state, v_intersect

        for idx in idx_order:
            # pass 2: now try the near-hits
            if polytope.slacks[idx] < 1e-6:
                continue
            if rel_energy[idx] < -1.0e-8:
                continue
            transition_state = state + polytope.labels[idx]
            if self.inside_state(v_intersect, transition_state):
                return transition_state, v_intersect
        if not self.inside_state(v_intersect, transition_state):
            if not (deep_search):
                print(old_v, new_v, state)
                raise LookupError()

            transition_state = self.find_state_of_voltage(
                new_v, state, deep_search=False
            )

        return transition_state, v_intersect

    def find_state_of_voltage(self, v, state_hint, deep_search=True):
        """Searches the ground state for a given voltage, given an initial guess.

        For a given state voltage, computes the state for which is within the polytope of the state.
        Note that the choice of the hint is not arbitrary, since the search starts from a point in state_hint
        in order to find iteratively intersections with the boundary that are closer to v. A specific requirement
        is that the polytope must not be empty, i.e., in case of a sliced simulator, the polytope must intersect
        with the affine space. This can sometimes be tricky and we recommend perform this type of computations
        only on spaces where all plungers are available and then perform the slicing through v.

        Parameters
        ----------
        v: np.array of floats
            Voltage vector for which to find the ground state
        state_hint: list of ints
            a likely candidate for the state.
        deep_search: bool
            whether an iterative search is performed in case none of the direct neighbours of a polytope match.
            If false, will throw an exception in that case. An exception is also raised when the deep search failed.
        """
        state = state_hint
        polytope = self.boundaries(state)
        if len(polytope.labels) == 0:
            raise ValueError("polytope of state_hint does not intersect with plane")

        # Check if hint was correct
        # If not we have to search.
        # We hope that the solution is close enough and find the transitions
        v_inside = polytope.point_inside.copy()
        while not self.inside_state(v, state):
            state, v_inside = self.find_boundary_intersection(
                v_inside, v, state, deep_search=deep_search
            )

        return state


class CapacitiveDeviceSimulator(AbstractCapacitiveDeviceSimulator):
    """
    This class simulates a quantum dot device based on a capacitance model.

    The simulator interally keeps track of the Coulomb diamonds (polytopes) and their transitions (facets),
    and takes care of keeping track of which transitions are feasible, with what precision, etc.
    This allows one to ask questions such as: "which transition does this facet correspond to?" and
    "what is the orthogonal axis in voltage space (i.e. virtual gate) that tunes across it?".
    The simulator will return, for each transition, a point on the transition line and the virtual gate.

    It also has the ability to take 2D slices through high dimensional voltage spaces to construct 2D
    projections of charge stability diagrams. See documentation of AbstractCapacitiveDeviceSimulator for more details.
    """

    def __init__(self, capacitance_model):
        super().__init__(capacitance_model)
        self.maximum_slack = 0.0

    def compute_polytope(self, state):
        return self.capacitance_model.compute_polytope_for_state(
            state, self.maximum_slack
        )

    def get_maximum_polytope_slack(self):
        return self.maximum_slack

    def set_maximum_polytope_slack(self, maximum_slack):
        self.maximum_slack = maximum_slack
        self.cache = {}

    def slice(self, P, m, proxy=None):
        # if proxy is not set, we check whether P is invertible
        # if it is invertible, then reusing the cache is the most efficient
        # in the general case where we don't know whether the original simulator
        # will be used still.

        # checking invertibility also allows us to quickly transform the cache
        is_invertible = is_invertible_matrix(P)
        if proxy is None:
            proxy = is_invertible

        if proxy:
            sliced_proxy = CapacitiveDeviceSimulatorProxy(self, P, m)
            return sliced_proxy
        else:
            sliced_simulator = CapacitiveDeviceSimulator(
                self.capacitance_model.slice(P, m)
            )
            sliced_simulator.maximum_slack = self.maximum_slack
            # slice all precomputed polytopes in a lazy manner.
            for key, polytope in self.cache.items():
                if is_invertible:
                    sliced_simulator.cache[key] = polytope.invertible_transform(P, m)
                else:
                    sliced_simulator.cache[key] = polytope.lazy_slice(P, m)

            return sliced_simulator


class CapacitiveDeviceSimulatorProxy(AbstractCapacitiveDeviceSimulator):
    """
    This class is a slice proxy for the CapacitiveDeviceSimulator class. It gets returned by
    any slice operation, when a proxy is requested. This is unlikely to be used by the user
    directly and mostly used during plotting. The advantage of a proxy is that it can make better use of
    caching at the expense of higher computation cost: all queries for polytopes are computed by the original simulator
    and thus if several different slices of the same simulator are needed, they can share computed polytopes.

    For the methods, see the documentation of AbstractCapacitiveDeviceSimulator
    """

    def __init__(self, simulator, P, m):
        super().__init__(simulator.capacitance_model.slice(P, m))
        self.simulator = simulator
        self.P = P
        self.m = m

    def get_maximum_polytope_slack(self):
        return self.simulator.get_maximum_polytope_slack()

    def set_maximum_polytope_slack(self, maximum_slack):
        self.simulator.set_maximum_polytope_slack(maximum_slack)
        self.cache = {}

    def compute_polytope(self, state):
        # query or compute original polytope
        polytope = self.simulator.boundaries(state)

        # transform lazyly

        polytope_sliced = polytope.lazy_slice(self.P, self.m)
        polytope_sliced = self.capacitance_model.verify_polytope(
            polytope_sliced, self.get_maximum_polytope_slack()
        )
        return polytope_sliced

    def slice(self, P, m, proxy=None):
        if proxy is None:
            proxy = True

        if proxy:
            return CapacitiveDeviceSimulatorProxy(self, P, m)
        else:
            new_P = self.P @ P
            new_m = self.m + self.P @ m
            return self.simulator.slice(new_P, new_m, False)
