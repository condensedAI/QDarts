import numpy as np
from qdarts.util_functions import (
    compute_maximum_inscribed_circle,
    compute_polytope_slacks,
)
from qdarts.polytope import Polytope
from abc import ABCMeta, abstractmethod


def is_sequence(seq):
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


# internal unit conversion of capacitances from attoFarrad to Farrad/eV
eV = 1.602e-19
to_Farrad_per_eV = 1e-18 / eV


class AbstractCapacitanceModel(metaclass=ABCMeta):
    """Base Class for all capacitance models.

    This class provides all required meta information to compute capacitive energies of a system with electron configuration
    n and gate voltages v, E(v,n). The only required to this model is that E(v,n)-E(v,n') is a linear function in v and that energies
    are measured in eV.
    The class provides basic abilities: enumeration of possible transition states from a given state and computation/verification of a polytope P(n) for a state.
    For this, the user only needs to provide functions to generate transition equations and a function that allows to slice the voltage space.

    As a base class it only needs to know the number of gates/inputs and the number of dots of the array. Additionally
    the user must supply voltage bounds that ensure that all computed polytopes are bounded. In practical devices these would be voltage limits
    e.g., for device protection.

    Attributes
    ----------
    num_dots: int
            the number of discrete dot locations on the device. This is the number of elements in the electron state n.
    num_inputs: int
        The number of gate voltages of the device.
    bounds_limits: N np.array of float
        right hand side of the bound inequalities.
        set of linear inequalities A that provide bounds for the voltage space.
    """

    def __init__(self, num_dots, num_inputs, bounds_limits, bounds_normals):
        """Initializes the model.

        Parameters
        ----------
        num_dots: int
            the number of discrete dot locations on the device. This is the number of elements in the electron state n.
        num_inputs: int
            The number of gate voltages of the device.
        bounds_limits: N np.array of float
            right hand side of the bound inequalities. If bounds_normals is none, this is interpreted as lower bounds and thus N=num_inputs
        bounds_normals: N x num_inputs np.array of float
            set of linear inequalities A that provide bounds for the voltage space. A valid voltage v fulfils Av+b<0, where b is
            bounds_limits. Can be none, in which case A=-Id.
        """
        self.num_dots = num_dots
        self.num_inputs = num_inputs

        if not is_sequence(bounds_limits):
            bounds_limits = bounds_limits * np.ones(num_inputs)

        if bounds_normals is None:
            if num_inputs != len(bounds_limits):
                raise ValueError(
                    "if bounds_normals is not given, bounds_limits must be either a scalar or a sequence of length same as number of gates"
                )
            bounds_normals = -np.eye(self.num_inputs)

        self.bounds_normals = np.asarray(bounds_normals)
        self.bounds_limits = np.asarray(bounds_limits)

    @abstractmethod
    def compute_transition_equations(self, state_list, state):
        """
        For a given state n and a list of other states (n_1,...n_N), computes the set of
        linear equations E(v,n)-E(v,n_i). Must be implemented by derived classes.

        Parameters
        ----------
        state: K np.array of int
            State n
        state_list: NxK np.array of int
            list of other states (n_1,...n_N)

        Returns
        -------
        (A,b) set of linear equations represented by matrix A and offset b. The ith element computes
        E(v,n)-E(v,n_i) as a function of v.
        """
        pass

    @abstractmethod
    def slice(self, P, m):
        """Restricts the model to the affine subspace v=m+Pv'

        Computes the slice through the device by setting v=m+Pv', where v is the plunger gate voltages of the
        original device and v' is the new coordinate system. Must be implemented by derived classes. Note
        that derived classes also need to apply the affine transformation to the bounds variables

        Parameters
        ----------
        P: MxK np.array of floats
            The linear coefficient matrix. Here M is the number of voltage elements in v in the full simulation
            and K the dimensionality of the subspace.
        m: M np.array of floats
            offset of the affine trnsformation.

        Returns
        -------
        A model object describing the simulation on the affine subspace.
        """
        pass

    def enumerate_neighbours(self, state):
        """Generates a state_list object for compute_transition_equations,

        Enumerates the set of neighbours of a transition to return all possible
        state transitions. In this class it is implemented by returning all possible
        states that can be reached by any combintion of adding or removing an electron on any dot.
        As a result, this list has 3^D-1 elements, where D is the number of dots.

        Derived classes may overwrite this if they want to consider a more restricted set of transitions.

        Parameters
        ----------
        state: D np.array of int
            The state for which to enumerate all neighbours

        Returns
        -------
        List of lists of neighbours. By default only a list including a list of all neighbours is returned.
        Derived classes may decide to instead return several lists each representing a batch of transitions.
        Aong those lists, states do not need to be unique but instead represent groups of elements that can
        filtered efficiently by the solver, e.g., by adding all states that can be used to filter a lot of other
        states in the list quickly. This is only ever relevant when larger arrays need to be computed.
        """
        d = state.shape[0]
        # Compute list of possible state transitions for the provided state
        # For simplicity, we restrict to only single electron additions/subtractions per dot
        # This leaves 3^d-1 states
        state_list = np.zeros((1, d), dtype=int)
        for i in range(d):
            state_list1 = state_list.copy()
            state_listm1 = state_list.copy()
            state_listm1[:, i] = -1
            state_list1[:, i] = 1
            state_list = np.vstack([state_list, state_list1])
            if state[i] >= 1:
                state_list = np.vstack([state_list, state_listm1])

        # First element is all-zeros, we don't want it
        state_list = state_list[1:]

        return [state_list + state]

    def compute_polytope_for_state(self, state, maximum_slack):
        r"""For a given state, computes P(N)

        Calls enumerate_neighbours and compute_transition_equations to obtain a list of
        possible transitions and then removes from this list iteratively
        all transitions that are not sufficiently close to the polytope.

        This is computed by computing the slack. The slack is 0 if the ith transition is a facet of
        the polytope, otherwise it is a positive number computed as

        :math:`s_i = min_v A_i^Tv +b_i, v \in P(n)`

        This function retains all facets that have slack smaller than maximum_slack.
        Since enregy differences are measure din eV, the slack represents the minimum
        energy difference between the ground state and the state represented by the transition
        for any point inside the polytope.

        Parameters
        ----------
        state: D np.array of int
            The electron configuration n for which to compute P(n)
        maximum_slack: float
            The maximum distance in eV after which the transition is discarded

        Returns
        -------
        A Polytope object representing P(n). See documentation of Polytope.
        """
        # get the potentially bacthed list of states
        state_lists = self.enumerate_neighbours(state)
        if len(state_lists)>1:
            raise NotImplementedError("batching of States not implemented, yet.")
        state_list = state_lists[0]
        A, b = self.compute_transition_equations(state_list, state)

        # check, whether there are superfluous transitions
        # TODO: Oswin: i don't remember what the significance of this was.
        zero_const = np.all(np.abs(A) < 1.0e-8, axis=1)
        if np.any(zero_const):
            A = A[~zero_const]
            b = b[~zero_const]
            state_list = state_list[~zero_const]
        # ... and check for this batch whether we can filter out non-touching ones
        slacks = compute_polytope_slacks(
            A, b, self.bounds_normals, self.bounds_limits, maximum_slack
        )
        keep = slacks <= maximum_slack + 1.0e-8

        # if we have kept nothing, this means there is a set of equations that is not fullfillable
        # this happens often when slicing, e.g, a polytope is not within the sliced subspace.
        if not np.any(keep):
            return Polytope(state)

        A = A[keep]
        b = b[keep]
        slacks = slacks[keep]
        state_list = state_list[keep]

        

        # create final polytope
        poly = Polytope(state)
        touching = slacks < 1.0e-8
        point_inside, _ = compute_maximum_inscribed_circle(
            A[touching], b[touching], self.bounds_normals, self.bounds_limits
        )
        poly.set_polytope(
            state_list - state, A, b, slacks, point_inside
        )
        return poly

    def verify_polytope(self, polytope, maximum_slack):
        """Verifies a polytope.

        After slicing, polytopes that have been computed earlier also need to be sliced. It is inefficient to recompute
        the polytopes from scratch, as slicing can only remove, but never add transitions. verify_polytope allows to take
        a polytope that has been modified via polytope.layz_slice and verify/filter all transitions. This recomputes
        all slack variables and removes all transitions that have slack larger than maximum slack

        Note that this does not touch any other internal information stored in the polytope.
        This function doe snothing if polytope.must_verify=False

        TOOD: this should be moved somewhere else.

        Parameters
        ----------
        polytope: Polytope
            The polytope P(n) to be verified
        maximum_slack: float
            The maximum distance in eV after which a transition of the polytope is discarded

        Returns
        -------
        The updated polytope after filtering out transitions.
        """
        if not polytope.must_verify:
            return polytope
        slacks = compute_polytope_slacks(
            polytope.A,
            polytope.b,
            self.bounds_normals,
            self.bounds_limits,
            maximum_slack,
        )
        keep = slacks <= maximum_slack + 1.0e-8
        touching = slacks <= 1.0e-6
        point_inside, _ = compute_maximum_inscribed_circle(
            polytope.A[touching],
            polytope.b[touching],
            self.bounds_normals,
            self.bounds_limits,
        )

        verified_polytope = Polytope(polytope.state)
        verified_polytope.set_polytope(
            polytope.labels[keep],
            polytope.A[keep],
            polytope.b[keep],
            slacks[keep],
            point_inside,
        )
        return verified_polytope


class CapacitanceModel(AbstractCapacitanceModel):
    """Implementation of a slight generalization of the constant interaction model.

    The constant interaction model defines
    :math:`E(v,n)=1/2 n^T C_{DD}^{-1}n - n^T  C_{DD}^{-1}C_{DG}v`

    where :math:`C_{DD}` and  :math:`C_{DG}` are part of the maxwell capacitance matrix created by the system
    of plunger gates G and quantum dots D. Thus, :math:`C_{DD}` are the interdot capacitances (mawell) and
    :math:`C_{DG}` the dot to gate capacitances.

    This model is a generalization of the constant interaction model as it makes :math:`C_{DD}` and  :math:`C_{DG}` a function
    of electron state n. The speed of this change from the constant interaction is governed by a parameter k for each dot. The larger
    k is, the smaller the deviation. if k=None, this is exactly the constant interaction model.
    """

    def __init__(
        self,
        C_g,
        C_D,
        bounds_limits,
        bounds_normals=None,
        ks=None,
        transform_C_g=None,
        offset=None,
    ):
        """Initializes the model

        The parameters here are normal capacitances and not maxwell capacitances given in atto Farrad.

        Parameters
        ----------
        C_g: DxK np.array of float
            Capacitances in atto Farrad between the K gates and D dots.
        C_D: DxD np.array of float
            Capacitances in atto Farrad between the D dots. Self capacitances are possible via the diagonal matrix elements.
        bounds_limits: N np.array of float
            right hand side of the bound inequalities. If bounds_normals is none, this is interpreted as lower bounds and thus N=num_inputs
        bounds_normals: N x num_inputs np.array of float
            set of linear inequalities A that provide bounds for the voltage space. A valid voltage v fulfils Av+b<0, where b is
            bounds_limits. Can be none, in which case A=-Id.'
        ks: D np.array of float or None.
            How quickly the capacitances change as deviation from the constant interaction model for each dot. Can be None in which this is just the constant interaction model.
            Larger integers give smaller changes. Realistic values are 3-5.
        transform_C_g:
            Internal. Used to implement slicing. Should be None.
        offset:
            Internal. Used to implement slicing. Should be None.
        """
        # Set the transformation matrix to the identity if not provided
        if transform_C_g is None:
            transform_C_g = np.eye(C_g.shape[1])

        super().__init__(
            C_D.shape[0], transform_C_g.shape[1], bounds_limits, bounds_normals
        )

        # Set instance properties
        self.C_g_atto = np.asarray(C_g)
        self.C_D_atto = np.asarray(C_D)
        self.transform_C_g = np.array(transform_C_g)

        # Check that an offset is provided for every gate
        self.offset = np.zeros(self.transform_C_g.shape[0])
        if offset is not None:
            if len(offset) != self.transform_C_g.shape[0]:
                raise ValueError(
                    "The offset you provided does not have an offset for every gate of the device (prior to slicing)."
                )
            self.offset = np.array(offset)

        # Convert units from attoFarrad to Farrad per eV
        self.C_g = self.C_g_atto * to_Farrad_per_eV
        self.C_D = self.C_D_atto * to_Farrad_per_eV

        # Check if value for non-constant capacitance is provided
        self.ks = ks
        if ks is not None:
            self.ks = np.array(ks)
            # if np.any(ks<1):
            #    raise ValueError("The ks values must be larger than 1")

            # TODO: What are S values?
            # Cache S values for up to 1000 total dots
            self.S_cache = np.zeros((1000, self.num_dots))
            self.S_cache[0, :] = 1
            self.S_cache[1, :] = 1

            r = 2.6
            alphas = 1 - 0.137 * (1 + r) / (ks + r)
            for n in range(2, self.S_cache.shape[0]):
                Sprev = self.S_cache[n - 2]
                self.S_cache[n] = n / (
                    2 * alphas * (ks + 2) / (n + ks) + (n - 2) / Sprev
                )

    def _compute_capacitances(self, state):
        N = len(state)

        S = np.eye(N)
        if self.ks is not None:
            S_values = self.S_cache[state, range(N)]
            S = np.diag(S_values)

        sum_C_g = np.sum(self.C_g, axis=1)

        # General transform by changing dot capacitances
        Cn_g = S @ self.C_g
        Cn_D = S @ self.C_D @ S
        Csum = S @ S @ sum_C_g + np.sum(Cn_D, axis=1) + np.diag(Cn_D)

        Cn_inv = np.linalg.inv(np.diag(Csum) - Cn_D)
        return Cn_inv, Cn_g

    def compute_transition_equations(self, state_list, state_from):
        # Get number of targets
        N = state_list.shape[0]

        # Compute normal and offset for the from_state
        C0_inv, C0_g = self._compute_capacitances(state_from)
        q0 = state_from @ C0_inv
        A0 = q0 @ C0_g
        b0 = q0 @ state_from - 2 * q0 @ C0_g @ self.offset

        # Now compute the normals and offsets for the target states
        A = np.zeros((N, self.num_inputs))
        b = np.zeros(N)
        for i, n in enumerate(state_list):
            # Compute the capacitances for the target state
            Cn_inv, Cn_g = self._compute_capacitances(n)
            qn = n @ Cn_inv
            An = qn @ Cn_g

            # Compute the normal
            A[i] = (An - A0) @ self.transform_C_g
            # Compute the offset
            b[i] = (b0 - qn @ n) / 2 + qn @ Cn_g @ self.offset

        return A, b

    def slice(self, P, m):
        new_offset = self.offset + self.transform_C_g @ m
        new_transform = self.transform_C_g @ P

        new_boundsA = self.bounds_normals @ P
        new_boundsb = self.bounds_limits + self.bounds_normals @ m

        # throw out almost orthogonal bounds
        sel = np.linalg.norm(new_boundsA, axis=1) > 1.0e-7 * np.linalg.norm(
            self.bounds_normals, axis=1
        )
        new_boundsA = new_boundsA[sel]
        new_boundsb = new_boundsb[sel]

        return CapacitanceModel(
            self.C_g_atto,
            self.C_D_atto,
            new_boundsb,
            new_boundsA,
            self.ks,
            new_transform,
            new_offset,
        )
