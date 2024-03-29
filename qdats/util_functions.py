import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm

def is_invertible_matrix(A,max_cond=1.e8):
    if A.shape[0] != A.shape[1]:
        return False
    else:
        return True
        return np.linalg.cond(A)<max_cond
def solve_linear_problem(prob):
    try:
       prob.solve(verbose=False, solver=cp.CLARABEL, max_iter=100000)
    except cp.SolverError:
        prob.solve(solver=cp.GLPK)

def compute_polytope_slacks(A, b, maximum_slack):
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
        
    #check if there is any feasible point in the polytope
    x = cp.Variable(A.shape[1])
    eps = cp.Variable(1)
    prob = cp.Problem(cp.Minimize(0), 
            [A@ x + b <= 0])
    solve_linear_problem(prob)
    if prob.status in ["infeasible", "infeasible_inaccurate"]:
        return (maximum_slack+1)*np.ones(len(b))
    
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
        prob = cp.Problem(cp.Minimize(eps),
             [A_eq @ x + b_eq + eps == 0, Ak@ x + bk <= 0, eps >= 0])
        solve_linear_problem(prob)
        if prob.status not in ["infeasible", "infeasible_inaccurate"]:
            slacks[k] = eps.value
            if eps.value < 1.e-6:
                touching[k] = True
    return slacks

def compute_maximum_inscribed_circle(A, b, bounds_normals, bounds_limits):
    if len(A)==0:
        return None, 0.0

    norm_A = np.linalg.norm(A, axis=1)
    norm_bounds = np.linalg.norm(bounds_normals, axis=1)
    
    r = cp.Variable(1)
    v = cp.Variable(A.shape[1])
    constraints = [
        A @ v + b + r * norm_A <= 0, #linear boundaries are only allowed to intersect the sphere once
        bounds_normals @ v + bounds_limits + norm_bounds*r <= 0, #also stay away from bound constraints
        r >=0 # Radius is strictly positive
    ]
    prob = cp.Problem(cp.Maximize(r), constraints)
    solve_linear_problem(prob)
    return v.value, r.value
    
def find_label(labels, label):
    dist = np.sum(np.abs(labels-np.array(label)),axis=1)
    return np.where(dist<1.e-5)[0]

def find_point_on_transitions(polytope, indizes):
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
    """
    Returns a new simulation where the values of the given gates are fixed to a constant. 
    This is done by computing the apropriate parameters for slice, and therefore the
    operation can not be undone in the returned simulation.
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
    
def axis_align_transitions(simulator, target_state, transitions, compensation_gates):
    """
    Takes a set of transitions from a target state and a set of gate indices of same length. 
    Computes a linear transformation such, that the ith transition is parallel to the ith gate axis supplied as
    argument.
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
    
    return simulator.slice(P,np.zeros(simulator.num_inputs))
    
def compensated_simulator(simulator, target_state, compensation_gates, sensor_ids, sensor_detunings, sensor_slope_detuning=0.0):
    
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
    
    return simulator.slice(P,v_zero)