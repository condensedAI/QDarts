import cvxpy as cp
import scipy
import clarabel
import numpy as np
from scipy.spatial import HalfspaceIntersection


def is_sequence(seq):
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


def is_invertible_matrix(A, max_cond=1.0e8):
    """Returns true if A is an invertible matrix.

    Parameters
    ----------
    max_cond: float
        conditioning of A (fraction of maximum and minimum absolute eigenvalue) above which
        it is assumed that A is numerically rank-deficient.
    """
    if A.shape[0] != A.shape[1]:
        return False
    else:
        return np.linalg.cond(A) < max_cond


def solve_linear_problem(prob):
    """Internal helper function to solve supplied linear cvxpy problems"""
    try:
        prob.solve(verbose=False, solver=cp.CLARABEL, max_iter=1000000)
    except cp.SolverError:
        prob.solve(solver=cp.GLPK)

def solve_linear_ineq_problem(cx,bx, A_ineq,b_ineq):
    """Internal helper function to solve LP with no equality constraints"""

    # convert to SCS format
    N = len(cx)
    M = A_ineq.shape[0]
    P = scipy.sparse.csc_matrix((N,N))
    q = -cx
    A = scipy.sparse.csc_matrix(A_ineq)
    b = -b_ineq
    
    cone = [clarabel.NonnegativeConeT(M)]

    # Set solver parameters
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    
    #initializeand solve
    solver = clarabel.DefaultSolver(P,q,A,b,cone,settings)
    solution = solver.solve()
    
    status = str(solution.status)
    success = False
    if status == "Solved" or status == "AlmostSolved":
        success = True
    
    if success :
        return True, -solution.obj_val+bx, solution.x
    else:
        return False, None, None
    
def _compute_polytope_slacks_1D(A, b):
    """Special case called by compute_polytope_slacks when A has a single column"""
    w = A.reshape(-1)
    close_to_zero = np.abs(w) < 1.0e-6 * np.max(w)
    # compute the single point fulfilling the constraint
    x = -b / (w + close_to_zero * 1.0e-6 * np.max(w))

    # check whether the constraint is a lower or upper bound
    is_lower = w < 0
    # count the constraints for checking whether we have both lower and upper bounds
    num_lower = np.sum(is_lower)

    # now find the lower and upper bounds of the interval that fulfill all constraints
    if num_lower > 0:
        lower_x = np.max(x[is_lower])
    else:
        lower_x = -np.inf
    if num_lower < len(A):
        upper_x = np.min(x[~is_lower])
    else:
        upper_x = np.inf
    # handling of infeasibility
    # since the slacks are defined as smallest violation of the constraints,
    # if the polytope is infeasible, the slack for the lower bound constraints
    # must be computed based on the upper bound and vice versa
    if lower_x > upper_x:
        temp = lower_x
        lower_x = upper_x
        upper_x = temp

    # compute slacks:
    slacks_lower = -(is_lower * (w * lower_x + b))
    slacks_upper = -((~is_lower) * (w * upper_x + b))

    slacks = slacks_lower + slacks_upper
    return slacks


def _compute_polytope_slacks_2D(A, b, bounds_A, bounds_b):
    """Special case in 2D solved via halfspace intersection"""
    
    # first we use halfspace intersection to compute all corners of the final polytope
    # find a point fulfilling all constraints
    feasible_point, _ = compute_maximum_inscribed_circle(A, b, bounds_A, bounds_b)
    # bring all constraints together in matrix form
    halfspaces_poly = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    halfspaces_bounds = np.concatenate([bounds_A, bounds_b.reshape(-1, 1)], axis=1)
    halfspaces = np.concatenate([halfspaces_poly, halfspaces_bounds], axis=0)
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    intersections = hs.intersections

    # now compute slacks. since the slack is computed as the solution of a linear programming problem,
    # the solution must lie on one of the vertices.
    slacks_intersections = A @ (intersections.T) + b.reshape(-1, 1)

    min_slacks = np.min(-slacks_intersections, axis=1)
    slacks = np.maximum(min_slacks, np.zeros(min_slacks.shape))

    return slacks


def compute_polytope_slacks(A, b, bounds_A, bounds_b, maximum_slack):
    """Computes the slacks of each candidate transition of a ground state polytope.

    The polytope is given by all points x, such that :math:`Ax+b<0`. There might be boundaries
    such that for no x holds that :math:`A_ix+b_i = 0`. In this case the definition of the polytope is
    the same when it is removed. However, sometimes we are interested in keeping transitions that
    are near misses - i.e., there exists an x such that, the inequality is almost fulfilled.
    In this case, we can relax this by allowing a positive slack and
    accept transitions to still be relevant for the polytope when we find an x, such that

    :math:`A_ix+b_i <= s`

    if this inequality holds exactly with slack :math:`s=0`, we say that the ith transition touches
    the polytope and the larger slack is, the more distant is the polytope

    Computing the slack can be difficult in the presence of unbounded polytopes. For this reason additional linear bound
    constraints need to be provided that ensure that all polytopes are bounded.

    The function computes the minimum slack for all transitions in :math:`A_i` and :math:`b_i`.
    The default slack is needed if for some reason it is not possible to compute the slack due to numerical
    difficulties.

    Parameters
    ----------
    A: NxK np.array of floats
        The linear coefficients of the N affine linear equations in K dimensions
    b: N np.array of floats
        The constant offsets of the N affine linear equations
    bounds_A: N'xK np.array of floats
        The linear coefficients of N' additional constraint that ensure that polytopes are bounded.
    bounds_b: N' np.array of floats
        The constant offsets of the N' bounds
    maximum_slack: float
        Value for the maximum acceptable slack for transitions to be considered near the polytope.
        TODO: review whether this parameter is needed.
    """

    # to find whether an equation a^Tx+b<= 0 touches the polytope, we need to find
    # a point that is on the intersection between polytope and the line where
    # a^Tx+b=0. For this we systematiclly go throuh the full list of
    # equations and for each solve an LP with this goal. if an equation
    # is found not to touch the polytope it is removed from the list
    # of candidates for all future solves.

    # we slightly generalize this problem by weakening the equation via
    # a slack variable
    # a^Tx+b = eps
    # eps > 0
    # and minimize for eps.
    # if eps = 0, then the equation touches the polytope. If not, then
    # it can be removed. this allows us to introduce a maximum slack value
    # which allows us to keep close matches for later stages in the simulation, e.g., to keep
    # neighbours that are possibly relevant for tunneling equations. Note: for eps>0 this formulation
    # produces the x value in the polytope closest to the equation in function value a^Tx+b.

    # we return the vector of eps values for all equations so that the user can filter transitions afterwards

    # first special cases.
    # only one constraint? feasible qwith slack 0
    if len(b) == 1:
        return np.zeros(1)

    # 1D and 2D problems can be solved efficiently
    if A.shape[1] == 1:
        return _compute_polytope_slacks_1D(A, b)
    if A.shape[1] == 2:
        return _compute_polytope_slacks_2D(A, b, bounds_A, bounds_b)

    # now we know there is a polyope and we can compute its sides
    N = len(A)
    touching = np.ones(
        N, dtype=bool
    )  # equations with eps~=0. At the beginning we assume all are touching
    slacks = (maximum_slack + 1) * np.ones(
        N
    )  # slack value (updated when equation is computed)
    A_touch = A
    b_touch = b
    for k in range(N):
        success, value, x = solve_linear_ineq_problem(A[k],b[k], A_touch,b_touch)
        touching[k]=False
        if success:
            slacks[k] = np.maximum(-value,0.0)
            if slacks[k] < 1.0e-6:
                touching[k] = True
        if not touching[k]:
            A_touch = A[touching, :]
            b_touch = b[touching]
    return slacks


def compute_maximum_inscribed_circle(A, b, bounds_A, bounds_b):
    """Computes the maximum inscribed circle in a polytope intersected with a set of linear inequalities.

    The maximum inscribed circle is a crude measure for position and size of a polytope.
    It computes the circle with maximum radius r and midpoint m, such that all its points
    lie inside the polytope. The function returns the (m,r) maximizing this. This choice is very often not unique.

    Since the polytope given by linear equations A,b might be unbounded, the function takes another
    set of linear equations for establishing lower and upper bounds. In essence, this is the same as
    adding the additional equalities to A and b and computing the maximum inscribed circle for that polytope.



    Parameters
    ----------
    A: NxK np.array of floats
        The linear coefficients of the N affine linear equations in the K-dimensional polytope
    b: N np.array of floats
        The constant offsets of the N affine linear equations of the polytope
    bounds_A: MxK np.array of floats
        The linear coefficients of the M added linear inequality constraints
    bounds_b: M np.array of floats
        The constant offsets of the M added linear inequality constraints
    """
    if len(A) == 0:
        return None, 0.0

    norm_A = np.linalg.norm(A, axis=1)
    norm_bounds = np.linalg.norm(bounds_A, axis=1)

    r = cp.Variable()
    v = cp.Variable(A.shape[1])
    constraints = [
        A @ v + b + r * norm_A
        <= 0,  # linear boundaries are only allowed to intersect the sphere once
        bounds_A @ v + bounds_b + norm_bounds * r
        <= 0,  # also stay away from bound constraints
        r >= 0,  # Radius is strictly positive
    ]
    prob = cp.Problem(cp.Maximize(r), constraints)
    solve_linear_problem(prob)
    return v.value, r.value


def find_label(labels, label):
    """helper function that finds the position of a state in a matrix of states"""
    dist = np.sum(np.abs(labels - np.array(label)), axis=1)
    return np.where(dist < 1.0e-5)[0]


def find_point_on_transitions(polytope, indizes):
    """Finds a point on a facet (or intersection point of multiple facets) of a polytope

    Given a precomputed polytope with facets (A,b) and their slacks,
    computes a point where a set of facet inequalities are exactly equal.
    Among all points that fulfill this, we pick the mid point defined
    by the maximum inscribed circle on the facet (or the subfacet
    created by the intersection of facets).

    TODO: it is not quite clear what happens when the indizes are not touching.

    Parameters
    ----------

    polytope: A polytope object
        The polytope for which the intersections are computed
    indices: list of int
        The subset of facets for which a common point is to be found.
        note that it is quietly assumed that the facets are touching the polytope.
    """
    slacks = np.delete(polytope.slacks, indizes)
    A = np.delete(polytope.A, indizes, axis=0)
    b = np.delete(polytope.b, indizes)
    A = A[slacks < 1.0e-8]
    b = b[slacks < 1.0e-8]
    A_eq = polytope.A[indizes, :]

    b_eq = polytope.b[indizes]

    norms = np.linalg.norm(A, axis=1)

    eps = cp.Variable()
    x = cp.Variable(A.shape[1])
    prob = cp.Problem(
        cp.Maximize(eps), [A @ x + b + norms * eps <= 0, A_eq @ x + b_eq == 0, eps >= 0]
    )
    solve_linear_problem(prob)
    return x.value


def fix_gates(simulator, gate_ids, gate_values, proxy=False):
    """Fixes a number of gate voltages in the simulator object

    Returns a new simulation where the values of the given gates are fixed to a constant.
    This is done by computing the apropriate parameters for slice, and therefore the
    operation can not be undone in the returned simulation.

    Please keep in mind that by doing this, all indices of gate voltages at entries
    after the deleted entries change, i.e., in a device with 4 plungers, removing the
    third plungers will lead to a simulator with 3 plungers where the last plunger has
    index 3. It is therefore advisable to order parameters such that fix_gates is always
    applied to the end.

    Parameters
    ----------
    simulator: AbstractPolytopeSimulator
        The simulator object for which gates are to be fixed
    gate_ids: list of int
        The indices of the gates in the voltage vector
    gate_values: np.array of float
        The values of the fixed gates
    proxy: bool
        whether or not the returned simulator should be a proxy, i.e., share cache if possible
        Todo: is this ever possible?
    """
    v = np.zeros(simulator.num_inputs)
    v[gate_ids] = gate_values

    P = np.zeros((simulator.num_inputs, simulator.num_inputs - len(gate_ids)))
    pos = 0
    for i in range(simulator.num_inputs):
        if i not in gate_ids:
            P[i, pos] = 1
            pos += 1
    return simulator.slice(P, v, proxy)


def axis_align_transitions(
    simulator, target_state, transitions, compensation_gates, proxy=True
):
    """Transform the simulators coordinate system such that transitions are aligned with coordinate axes

    Takes a set of transitions from a target state and a set of gate indices of same length.
    Computes a linear transformation such, that the normal of the ith transition is parallel to the ith gate axis supplied as
    argument.

    For example, to align the transition from state [1,1,1] to [1,1,2] with the first plunger gate, we set
    target_state=[1,1,1], transitions=[[1,0,0]] and compensation_gates=[0]

    Parameters
    ----------
    simulator: AbstractPolytopeSimulator
        The simulator object which is to be transformed
    target_state: list of int
        The state from which the transitions are extracted
    transitions: NxD np.array of int
        The set of N transitions (given as D-dimensional difference vectors state-target_state) to align
    compensation_gates: list of int
        The indices of the plunger gates that should be transformed to align with the transition normals
    proxy: bool
        whether or not the returned simulator should be a proxy, i.e., share cache if possible
    """
    compensation_gates = np.array(compensation_gates, dtype=int)
    # get the polytope of the target state
    polytope = simulator.boundaries(target_state)

    # find the transitions inside the polytope
    transition_idxs = []
    for transition in transitions:
        idx = find_label(polytope.labels, transition)[0]
        transition_idxs.append(idx)

    # get normals of the transitions
    normals = -polytope.A[transition_idxs, :]
    # normalize to ensure that we do not change the spacing of transitions
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    # compute compensation matrix
    B = normals[:, compensation_gates]
    compensation = -B.T @ np.linalg.inv(B @ B.T)

    # compute coordinate transform
    P = np.eye(simulator.num_inputs)

    # get the indizes of the elements in the submatrix of the compensation parameters
    P_sub_ids = (
        simulator.num_inputs * compensation_gates[:, None] + compensation_gates[None, :]
    )
    np.put(P, P_sub_ids, compensation.flatten())

    return simulator.slice(P, np.zeros(simulator.num_inputs), proxy=proxy)

def compute_sensor_compensation_transform(
    simulator,
    target_state,
    compensation_gates,
    sensor_ids,
    sensor_detunings,
    sensor_slope_detuning=0.0,
):
    """Transforms the simulation to compensate the sensors against all other gates.

    This function allows for perfect or imperfect sensor compensation as well as the exact position on the sensor peak.
    This is done by finding the compensation values of the sensor plunger gates to compensate for the linear cross-talk of all other
    plungers. This compensation is computed for a given target state as the compensation parameters might depend on the capacitances
    in the state if they are variable.

    The position on the sensor peak is given by sensor_detunings which move the position as a direct modification of the sensor potential.

    Parameters
    ----------
    simulator: AbstractPolytopeSimulator
        The simulator object which is to be transformed
    target_state: list of int
        The state from which the transitions are extracted
    compensation_gates: list of int
        The gates to be used for compensation of the sensor. Typically the sensor plunger gates in the device
    sensor_ids: list of int
        The indices of the sensor ids.
    sensor_detunings: np.array of float
        detuning parameter for each sensor which allows to move the sensor on a pre-specified point of the peak.
    sensor_slope_detuning: float
        (Experimental) scaling factor that moves the compensation linearly from perfect compensation (0) to no compensation (1).

    Returns
    -------
    P: the compensation matrix
    compute_bias(v): computes the offset b so that P@v+b is a point with the given detuning
    v: a point with detuning = 0, the point the compensation is computed about
    """
    if len(sensor_ids) != len(compensation_gates):
        raise ValueError(
            "Number of gates for compensation must equal number of sensors"
        )

    if len(sensor_ids) != len(sensor_detunings):
        raise ValueError(
            "Number of gates for compensation must equal number of sensors"
        )

    for sensor in sensor_ids:
        if target_state[sensor] <= 0:
            raise ValueError(
                "Target state must have at least one electron on each sensor dot"
            )

    compensation_gates = np.array(compensation_gates, dtype=int)
    other_gates = np.delete(np.arange(simulator.num_inputs), compensation_gates)
    sensor_detunings = np.array(sensor_detunings)

    # by default we assume that for the sensor dots,
    # we compute the transition between dots K and K+1
    # where K is the electron occupation on the target state,
    # which means that we take the polytope at the target state
    # and compute the transition for the electron K->K+1 on the sensor dot.
    # However, we will change the computed polytope based
    # on the sensor detuning. if it is positive, we will instead compute
    # the polytope for the K+1 electron and then search for the transition K+1->K

    target_state = target_state.copy()
    transitions = []
    detunings = []
    for detuning, sens_id in zip(sensor_detunings, sensor_ids):
        if detuning > 0:
            transitions.append(-np.eye(1, simulator.num_dots, sens_id))
            detunings.append(-detuning)
        else:
            target_state[sens_id] -= 1
            transitions.append(np.eye(1, simulator.num_dots, sens_id))
            detunings.append(detuning)

    # get geometry of the target state to compensate for
    polytope = simulator.boundaries(target_state)

    # find the sensor transitions inside the polytope
    transition_idxs = []
    for transition in transitions:
        idx = find_label(polytope.labels, transition)[0]
        transition_idxs.append(idx)

    # get normals of sensor transitions
    normals = polytope.A[transition_idxs, :]
    # compute point on the intersection of the transition
    v = find_point_on_transitions(polytope, transition_idxs)
    

    # compute compensation matrix
    #normals /= np.linalg.norm(normals, axis=1)[:, None]
    A1 = normals[:, compensation_gates]
    A2 = normals[:, other_gates]
    compensation = -np.linalg.inv(A1) @ A2
    # now create the P-matrix
    P = np.eye(simulator.num_inputs)
    # get the indizes of the elements in the submatrix of the compensation parameters
    P_sub_ids = (
        simulator.num_inputs * compensation_gates[:, None] + other_gates[None, :]
    )
    np.put(P, P_sub_ids, compensation.flatten())

    # add errors to P-matrix
    P_error = (1 - sensor_slope_detuning) * P + sensor_slope_detuning * np.eye(
        simulator.num_inputs
    )
    
    #offset function. 
    
    # apply sensor detunings. First compute virtual gate matrix P_S for the sensor voltages
    P_S = A1.T @ np.linalg.inv(A1 @ A1.T)
    v_detuned = v.copy()
    v_detuned[compensation_gates] += P_S@detunings
    offset_v = v_detuned - P[:, other_gates] @ v[other_gates]
    
    
    def compute_offset(v):
        #using only the non-sensor gates, compute the sensor point with given detuning
        v_comp = P[:, other_gates] @ v[other_gates] + offset_v
        #b is the offset so that P_error@v+b=v_comp
        b = v_comp-P_error@v
        return b
    
    return P_error,  compute_offset, v_detuned

def compensate_simulator_sensors(
    simulator,
    target_state,
    compensation_gates,
    sensor_ids,
    sensor_detunings,
    sensor_slope_detuning=0.0,
):
    """Transforms the simulation to compensate the sensors against all other gates.

    This function allows for perfect or imperfect sensor compensation as well as the exact position on the sensor peak.
    This is done by finding the compensation values of the sensor plunger gates to compensate for the linear cross-talk of all other
    plungers. This compensation is computed for a given target state as the compensation parameters might depend on the capacitances
    in the state if they are variable.

    The position on the sensor peak is given by sensor_detunings which move the position as a direct modification of the sensor potential.

    Parameters
    ----------
    simulator: AbstractPolytopeSimulator
        The simulator object which is to be transformed
    target_state: list of int
        The state from which the transitions are extracted
    compensation_gates: list of int
        The gates to be used for compensation of the sensor. Typically the sensor plunger gates in the device
    sensor_ids: list of int
        The indices of the sensor ids.
    sensor_detunings: np.array of float
        detuning parameter for each sensor which allows to move the sensor on a pre-specified point of the peak.
    sensor_slope_detuning: float
        (Experimental) scaling factor that moves the compensation linearly from perfect compensation (0) to no compensation (1).

    Returns
    -------
    sliced_sim: the sliced simulation that is created from the computed compensation parameters
    tuning_point: a vector of gate voltages that indicates the exact compensation point of the simulation
    """
    
    P_error, compute_bias, v = compute_sensor_compensation_transform(
        simulator,
        target_state,
        compensation_gates,
        sensor_ids,
        sensor_detunings,
        sensor_slope_detuning
    )
    offset = compute_bias(v)
    
    return simulator.slice(P_error, offset, True), v