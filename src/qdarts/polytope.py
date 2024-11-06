import numpy as np


class Polytope:
    r"""Represents the polytope P(n) defined by all gate voltages v in a device that have
    capacitive ground state n. They are stored as a set of linear inequalities (A,b), and
    A point v in P(n) fulfills

    :math:`Av+b <0`

    Each inequality represents a facet of the polytope and each facet marks a transition from P(n) to
    some other ground state polytope P(n'). The state difference t=n'-n is stored as label for each
    inequality.

    Not each inequality stored must touch the polytope. There might be others that could be removed without changing P(n).
    The distance from the polytope is given by the slack variable s that for each inequality either is 0 if the side is touching
    (or some number numerically close to 0, e..g, 1.e-8) and otherwise we have

    :math:`s_i = min_v A_i^Tv +b_i, v \in P(n)`

    If the inequalities measure difference in capacitive energy from the ground state (which is default in the simulator) the slack
    therefore indicates the minimum energy gap between the transition state indicated by the inequality and the ground state.

    Finally, for optimization reason, not every polytope might be fully computed and must be verified. This should never happen to a user
    and is mostly an internal detail of the simulator. This holds as well for additional information that can be stored inside a dict in the
    polytope. The simulators can store additional info in the polytope via this way.

    Attributes
    ----------
        state: D np.array of int
            The D dimensional array that stores the electron configuration n of the current ground state polytope. All points inside
            the polytope have this state as ground state.
        labels: NxD np.array of int
            Matrix of transitions. Each of the N inequalities is labeled by the state difference t=n'-n.
        A: NxK np.array of float
            Linear factors of the N inequalities for a K-dimensional gate space.
        b: N np.array of float
            constant offsets of the N linear inequalities
        slacks: N np.array of float
            Measures the distance of the inequality from the boundary of the polytope. ~0 if touching.
        point_inside: K np.array of float
            A point inside the polytope.
        must_verify: bool
            Internal variable for bookkeeping whether the polytope needs recomputing (mostly after slice). Should ALWAYS be False for polytopes
            queried from the simulator. TODO: can we remove this?
        additional_info: dict
            Internal additional information that later steps of the simulation can store inside a polytope for bookkeeping.
    """

    def __init__(self, state):
        # empty polytope
        self.state = state
        self.labels = np.array([])
        self.A = np.array([])
        self.b = np.array([])
        self.slacks = np.array([])
        self.point_inside = np.array([])
        self.must_verify = False
        self.additional_info = {}

    def set_polytope(self, labels, A, b, slacks, point_inside, must_verify=False):
        """Sets the internal variables of the polytope.

        Helper function to ensure thateverything is set as it should be.
        """
        self.labels = labels
        self.A = A
        self.b = b
        self.slacks = slacks
        self.point_inside = point_inside
        self.must_verify = must_verify

    def lazy_slice(self, P, m):
        """
        Slices a polytope lazyily, i.e., without recomputing the slacks and boundaries.

        As a result, after this must_verify is True. P is not required to be invertible.

        Parameters
        ----------
        P: KxK np.array of float
            Invertible linear transformation matrix
        m: K np.array of float
            Affine offset of the transformation.
        """
        sliced = Polytope(self.state)
        if self.A.shape[0] == 0:
            sliced.set_polytope(self.labels, self.A, self.b, np.array([]), None, False)
        else:
            sliced.set_polytope(
                self.labels,
                self.A @ P,  # we know the line equations
                self.b + self.A @ m,  # and their offsets
                None,
                None,  # but nothing else
                True,  # user must verify this later.
            )
        return sliced

    def invertible_transform(self, P, m):
        """
        Apply an invertible affine transformation to the polytope. This can be done without changing slacks and thus no verification is needed.

        Changes the space of the polytope via the transformation :math:`v=Av'+b`. Returns the polytope in the coordinate system of v'

        Parameters
        ----------
        P: KxK np.array of float
            Invertible linear transformation matrix
        m: K np.array of float
            Affine offset of the transformation.
        """
        if self.must_verify:
            return self.lazy_slice(P, m)

        transformed = Polytope(self.state)

        transformed_point_inside = np.linalg.inv(P) @ (self.point_inside - m)
        transformed.set_polytope(
            self.labels,
            self.A @ P,  # we know the line equations
            self.b + self.A @ m,  # and their offsets
            self.slacks,  # slacks are constant under invertible transforms.
            transformed_point_inside,
        )
        return transformed
