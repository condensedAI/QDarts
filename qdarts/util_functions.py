import cvxpy as cp
import numpy as np

def is_invertible_matrix(A,max_cond=1.e8):
    """ Returns true if A is an invertible matrix.
    
    Parameters
    ----------
    max_cond: float
        conditioning of A (fraction of maximum and minimum absolute eigenvalue) above which
        it is assumed that A is numerically rank-deficient. 
    """
    if A.shape[0] != A.shape[1]:
        return False
    else:
        return np.linalg.cond(A)<max_cond
def solve_linear_problem(prob):
    """Internal helper function to solve supplied linear cvxpy problems"""
    try:
       prob.solve(verbose=False, solver=cp.CLARABEL, max_iter=100000)
    except cp.SolverError:
        prob.solve(solver=cp.GLPK)

def compute_polytope_slacks(A, b, maximum_slack):
    """Computes the slacks of each candidate transition of a ground state polytope.
    
    The polytope is given given by all points x, such that Ax+b<0. There might be boundaries
    such that for no x holds that A_ix+b_i = 0. In this case the definition of the polytope is
    the same when it is removed. However, we can relax this by allowing a positive slack and 
    accept transitions to still be relevant for the polytope when we find an x, such that
    
    A_ix+b_i <= slack
    
    if this inequality holds exactly with slack=0, we say that the ith transition touches
    the polytope and the larger slack is, the more distant is the polytope
    
    The function computes the slack for all transitions in A_i and b_i. default slack
    is needed if for some reason it is not possible to compute the slack due to numerical
    difficulties.
    
    Parameters
    ----------
    A: NxK np.array of floats
        The linear coefficients of the N affine linear equations
    b: N np.array of floats
        The constant offsets of the N affine linear equations
    maximum_slack: float
        Value for the maximum acceptable slack for transitions to be considered near the polytop.
        TODO: review whether this parameter is needed.
    """
    
    #to find whether an equation a^Tx+b<= 0 touches the polytope, we need to find
    #a point that is on the intersection between polytope and the line where
    #a^Tx+b=0. For this we systematiclly go throuh the full list of 
    #equations and for each solve an LP with this goal. if an equation 
    # is found not to touch the polytope it is removed from the list
    # of candidates for all future solves.
    
    #we slightly generalize this problem by weakening the equation via
    # a slack variable 
    #a^Tx+b = eps
    #eps > 0
    #and minimize for eps.
    #if eps = 0, then the equation touches the polytope. If not, then
    #it can be removed. this allows us to introduce a maximum slack value 
    #which allows us to keep close matches for later stages in the simulation, e.g., to keep
    # neighbours that are possibly relevant for tunneling equations. Note: for eps>0 this formulation
    # produces the x value in the polytope closest to the equation in function value a^Tx+b.
    
    #we return the vector of eps values for all equations so that the user can filter transitions afterwards
    
    #first two special cases:
    if len(b) == 1:
        return np.zeros(1)
    
    #now we know there is a polyope and we can compute its sides
    N = len(A)
    touching = np.ones(N, dtype=bool) #equations with eps~=0. At the beginning we assume all are touching
    slacks = (maximum_slack + 1)*np.ones(N) #slack value (updated when equation is computed)    
    for k in range(N):
        # take all previous tested and verified touching eqs and all untested eqs, except the current    
        touching[k] = False
        Ak = A[touching,:] 
        bk = b[touching]
        
        
        #the current equation to test
        A_eq = A[k] 
        b_eq = b[k]
        
        #setup optimisation problem
        x = cp.Variable(A.shape[1])
        eps = cp.Variable()
        prob = cp.Problem(cp.Minimize(eps),
             [A_eq @ x + b_eq + eps == 0, Ak@ x + bk <= 0, eps >= 0])
        solve_linear_problem(prob)
        if prob.status not in ["infeasible", "infeasible_inaccurate"]:
            slacks[k] = eps.value
            if eps.value < 1.e-6:
                touching[k] = True
    return slacks

def compute_maximum_inscribed_circle(A, b, bounds_A, bounds_b):
    """Computes the maximum inscribed circle in a polytope intersected with a set of linear inequalities.
    
    The maximum inscribed circle is a crude measure for position and size of a polytope.
    It computes the circle with maximum radius r and midpoint m, such that all its points 
    lie inside the polytope. The function returns (m,r).
    
    Since the polytope given by linear equations A,b might be unbounded, the function takes another
    set of linear equations for establishing lower and upper bounds. In essence, this is the same as
    adding the additional equalities to A and b and computing the maximum inscribed circle for that polytope.
    
    
    
    Parameters
    ----------
    A: NxK np.array of floats
        The linear coefficients of the N affine linear equations of the polytope
    b: N np.array of floats
        The constant offsets of the N affine linear equations of the polytope
    bounds_A: MxK np.array of floats
        The linear coefficients of the M added linear inequality constraints
    bounds_b: M np.array of floats
        The constant offsets of the M added linear inequality constraints
    """
    if len(A)==0:
        return None, 0.0

    norm_A = np.linalg.norm(A, axis=1)
    norm_bounds = np.linalg.norm(bounds_A, axis=1)
    
    r = cp.Variable()
    v = cp.Variable(A.shape[1])
    constraints = [
        A @ v + b + r * norm_A <= 0, #linear boundaries are only allowed to intersect the sphere once
        bounds_A @ v + bounds_b + norm_bounds*r <= 0, #also stay away from bound constraints
        r >=0 # Radius is strictly positive
    ]
    prob = cp.Problem(cp.Maximize(r), constraints)
    solve_linear_problem(prob)
    return v.value, r.value
    
def find_label(labels, label):
    """helper function that finds the position of a state in a matrix of states"""
    dist = np.sum(np.abs(labels-np.array(label)),axis=1)
    return np.where(dist<1.e-5)[0]

def find_point_on_transitions(polytope, indizes):
    """Finds a point on a facet (or intersection point of multiple facets) of a polytope
    
    Given a precomputed polytope with facets (A,b) and their slacks,
    computes a point where a set of facet inequalities are exactly equal. 
    Among all points that fulfill this, we pick the mid point defined
    by the maximum inscribed circle on the facet (or the subfacet
    created by the intersection of facets).
    
    Parameters
    ----------
    
    polytope: A polytope object
        The polytope for which the intersections are computed
    indices: list of int
        The subset of facets for which a common point is to be found.
        note that it is quietly assumed that the facets are touching the polytope.
    """
    slacks = np.delete(polytope.slacks, indizes)
    A = np.delete(polytope.A, indizes,axis=0)
    b = np.delete(polytope.b, indizes)
    A = A[slacks<1.e-8]
    b = b[slacks<1.e-8]
    A_eq = polytope.A[indizes,:]
    
    b_eq = polytope.b[indizes]
    
    norms = np.linalg.norm(A,axis=1)
    
    eps = cp.Variable()
    x = cp.Variable(A.shape[1])
    prob = cp.Problem(cp.Maximize(eps),
             [A @ x + b+norms*eps <=0, A_eq@x+b_eq == 0,eps >= 0])
    solve_linear_problem(prob)
    return x.value

def fix_gates(simulator, gate_ids, gate_values, proxy=False):
    """Fixes a number of gate voltages in the simulator object
    
    Returns a new simulation where the values of the given gates are fixed to a constant. 
    This is done by computing the apropriate parameters for slice, and therefore the
    operation can not be undone in the returned simulation.
    
    Please keep in mind that by doing this, all indices of gate voltages at entries
    after the deleted entries change.
    
    Parameters
    ----------
    simulator: BasePolytopeSimulator
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
    
    P=np.zeros((simulator.num_inputs,simulator.num_inputs-len(gate_ids)))
    pos = 0
    for i in range(simulator.num_inputs):
        if i not in gate_ids:
            P[i,pos] = 1
            pos += 1
    return simulator.slice(P, v, proxy)
    
def axis_align_transitions(simulator, target_state, transitions, compensation_gates,proxy=True):
    """Transform the simulators coordinate system such that transitions are aligned with coordinate axes
    
    Takes a set of transitions from a target state and a set of gate indices of same length. 
    Computes a linear transformation such, that the ith transition is parallel to the ith gate axis supplied as
    argument.
    
    Parameters
    ----------
    simulator: BasePolytopeSimulator
        The simulator object which is to be transformed
    target_state: list of int
        The state from which the transitions are extracted
    transitions: NxK np.array of int
        The set of N transitions (given as difference vectors state-target_state) to align
    compensation_gates: list of int
        The indices of the plunger gates that should be transformed to align with the transition normals
    proxy: bool
        whether or not the returned simulator should be a proxy, i.e., share cache if possible
    """
    compensation_gates = np.array(compensation_gates, dtype=int)
    #get the polytope of the target state
    polytope  = simulator.boundaries(target_state)
    
    #find the transitions inside the polytope
    transition_idxs = []
    for transition in transitions:
        idx = find_label(polytope.labels, transition)[0]
        transition_idxs.append(idx)
    
    #get normals of the transitions
    normals = -polytope.A[transition_idxs,:]
    #normalize to ensure that we do not change the spacing of transitions
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    
    #compute compensation matrix
    B = normals[:,compensation_gates]
    compensation = -B.T@np.linalg.inv(B@B.T)
    
    #compute coordinate transform
    P=np.eye(simulator.num_inputs)
    
    #get the indizes of the elements in the submatrix of the compensation parameters
    P_sub_ids = simulator.num_inputs * compensation_gates[:,None] + compensation_gates[None,:]
    np.put(P,P_sub_ids, compensation.flatten())
    
    return simulator.slice(P,np.zeros(simulator.num_inputs), proxy=proxy)
    
def compensate_simulator_sensors(simulator, target_state, compensation_gates, sensor_ids, sensor_detunings, sensor_slope_detuning=0.0):
    """Transforms the simulation to compensate the sensors against all other gates.
    
    This function allows for perfect or imperfect sensor compensation as well as the exact position on the sensor peak. 
    This is done by finding the compensation values of the sensor plunger gates to compensate for the linear cross-talk of all other 
    plungers. This compensation is computed for a given target state as the compensation parameters might depend on the capacitances
    in the state if they are variable. 
    
    The position on the sensor peak is given by sensor_detunings which move the position as a direct modification of the sensor potential.
    
    Parameters
    ----------
    simulator: BasePolytopeSimulator
        The simulator object which is to be transformed
    target_state: list of int
        The state from which the transitions are extracted
    compensation_gates: list of int
        The gates to be used for compensation of the sensor. Typiclaly the sensor plunger gates in the device
    sensor_ids: list of int
        The indices of the sensor ids.
    sensor_detunings: np.array of float
        detuning parameter for each sensor which allows to move the sensor on a pre-specified point of the peak.
    sensor_slope_detuning: float 
        (Experimental) scaling factor that moves the compensation linearly from perfect compensation (0) to no compensation (1).
    """
    if len(sensor_ids) != len(compensation_gates):
        raise ValueError('Number of gates for compensation must equal number of sensors')
        
    if len(sensor_ids) != len(sensor_detunings):
        raise ValueError('Number of gates for compensation must equal number of sensors')
    
    for sensor in sensor_ids:
        if target_state[sensor] <= 0:
            raise ValueError('Target state must have at least one electron on each sensor dot')
        
    compensation_gates = np.array(compensation_gates,dtype=int)
    other_gates = np.delete(np.arange(simulator.num_inputs),compensation_gates)
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
        if  detuning > 0:
            
            transitions.append(-np.eye(1,simulator.num_dots,sens_id))
            detunings.append(-detuning)
        else:
            target_state[sens_id] -= 1
            transitions.append(np.eye(1,simulator.num_dots,sens_id))
            detunings.append(detuning)
    
    #get geometry of the target state to compensate for
    polytope  = simulator.boundaries(target_state)
    
    
    #find the sensor transitions inside the polytope
    transition_idxs = []
    for transition in transitions:
        idx = find_label(polytope.labels, transition)[0]
        transition_idxs.append(idx)
    
    #get normals of sensor transitions
    normals = polytope.A[transition_idxs,:]
    #compute point on the intersection of the transition
    v = find_point_on_transitions(polytope, transition_idxs)
    #apply sensor detunings. First compute virtual gates for the
    #two sensor voltages
    comp_det = normals.T@ np.linalg.inv(normals @ normals.T)
    #now use the compensation to define sensor detunings
    v_detuning = comp_det @ sensor_detunings
    v -= v_detuning #use detuning to move the point away from the transition
    
    
    
    #compute compensation matrix
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    A1 = normals[:,compensation_gates]
    A2 = normals[:,other_gates]
    compensation = -np.linalg.inv(A1)@A2
    
    
    #now create the P-matrix
    P=np.eye(simulator.num_inputs)
    #get the indizes of the elements in the submatrix of the compensation parameters
    P_sub_ids = simulator.num_inputs * compensation_gates[:,None] + other_gates[None,:]
    np.put(P,P_sub_ids, compensation.flatten())
    
    #add errors to P-matrix 
    P = (1-sensor_slope_detuning)*P+sensor_slope_detuning*np.eye(simulator.num_inputs)
    
    #If we compensate now with v as central point, our gates would compute
    #relative voltages to this (arbitrary) point. Tis would make it impossible
    #to plot the same region with different compensation points 
    #instead, we will now take v and move it such, that the other gates are 0.
    v_zero = v - P[:,other_gates]@v[other_gates]
    
    return simulator.slice(P, v_zero,True)
