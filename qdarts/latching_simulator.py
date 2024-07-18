import numpy as np
from util_functions import find_label
from tunneling_simulator import ApproximateTunnelingSimulator, LocalSystem
import scipy
from tqdm import tqdm
import cvxpy as cp

def sigmoid(x):
    x=np.maximum(x,-100*np.ones(x.shape))
    return 1/(1+np.exp(-x))
        
class LatchingSimulator:
    def __init__ (self, tunneling_sim, reservoir_coupling,scan_dt, transition_quantile=None):
    
        self.tunneling_sim = tunneling_sim
        self.poly_sim = self.tunneling_sim.poly_sim
        self.transition_quantile = transition_quantile if transition_quantile is not None else 0.999
        self.reservoir_coupling = reservoir_coupling
        self.scan_dt = scan_dt
        
        self.num_inputs = self.poly_sim.num_inputs
        self.num_dots = self.poly_sim.num_dots
            
    def slice(self, P, m, proxy=False):
        sliced_tunneling_sim = self.tunneling_sim.slice(P,m, proxy)
        sliced_latching_simulator = LatchingSimulator(
            sliced_tunneling_sim,
            self.reservoir_coupling,
            self.scan_dt,
            self.transition_quantile
        )
        return sliced_latching_simulator
    def boundaries(self, state):
        return self.tunneling_sim._get_polytope(state)
        
    def _compute_gamma(self, local_system):
        #we only compute gamma for the core index set
        core_indices = local_system.core_basis_indices
        basis = local_system.basis_labels[core_indices]
        H = local_system.H[core_indices,:][:,core_indices]
    
        #compute gamma matrix for the full hamiltonian
        
        # compute the coupling matrix for all state pairs in the basis.
        
        # Step 1: compute tc
        # hack: we query t_c indirectly via H, as this is
        # how the off-diagonal elements are currently set.
        
        
        tc_inner = -(H-np.diag(np.diag(H)))

        # now compute which transition pairs require coupling to the reservoir
        # compute which combinations of state differ by a total of 1 charge
        diffs = np.abs(np.sum(basis.reshape(-1,1,basis.shape[1])-basis.reshape(1,-1,basis.shape[1]),axis=2))
        #then multiply by the reservoir coupling
        tc_reservoir = (diffs==1) * self.reservoir_coupling

        #final tunnel matrix, note that all values are in eV
        tunnel_matrix = tc_inner + tc_reservoir
        
        # Step 2: compute transition-rate matrix gamma

        #this is a bit hacky since i did not find the total model.
        #It appears that on a double dot with zero detuning, we have
        #that tunnel rate is tc/h and then it is kinda known/assumed that 
        #as detuning increases in any direction, we have ~exponential decay.
        #i assume this comes from the observation that tunnel rate is measured
        #as a back<->forth, which is based on the probability to be in a certain configuration
        #times the probability to switch, while for gamma we are (for now) ONLY interested in the probability 
        #to switch. Let i,j be the source and target state so we have i->j. I am modeling the exponential decay part
        #as p(j)/(p(i)+p(j)), meaning that if j is much less likely than i, then the transition rate will be small
        #and vice versa.
        eV = 1.602e-19
        h_in_ev = 1.055e-34*2*np.pi/eV
        

        dE = np.diag(H).reshape(1,-1) - np.diag(H).reshape(-1,1)
        gamma = 1000*tunnel_matrix/h_in_ev*2*sigmoid(-local_system.beta*dE)
        gamma = gamma - np.diag(np.sum(gamma,axis=1))
        return gamma,basis
    
    def filter_basis(self, gamma, basis, start_state_idx, dt):
        #we filter gamma to only include basis elements that have a realistic probability of being reached from
        #the start state. For this we do a flood fill search where we include a state s_j into a list of states s_i0..s_ik
        #if there exists an s_io such that the probability to transition from s_i0 to s_j  
        #within time dt is bigger than the threshold.
        #computing this is easy as this results in the lower bound for gamma_ii0:
        gamma_min = -np.log(self.transition_quantile)/dt
        selected=[start_state_idx]
        candidate_pos = 0
        while candidate_pos < len(selected):
            candidate = selected[candidate_pos]
            for j in range(len(gamma)):
                if j in selected: continue
                if gamma[candidate,j]>=gamma_min:
                    #second level check: we are jumping to this state but some other state
                    #is much much more often sampled in the same time.
                    selected.append(j)
            candidate_pos += 1
          
        #select subset of gamma matrix and renormalize diagonal elements
        gamma_sel = gamma[selected,:][:,selected]
        gamma_sel = gamma_sel - np.diag(np.sum(gamma_sel,axis=1))
        return gamma_sel, basis[selected], selected
        
        
        
    def _compute_markov_chain(self, local_system, dt):
        gamma,basis = self._compute_gamma(local_system)
        p_stable = np.maximum(np.diag(local_system.mixed_state)[local_system.core_basis_indices],np.zeros(len(basis)))
        #print(gamma.shape,local_system.H.shape, local_system.basis_labels.shape)
        #print(basis, local_system.state)
        #the resulting gamma matrix might be very big. So find a suitable subbasis 
        start_state_idx = find_label(basis, local_system.state)[0]
        gamma, basis, filter_basis_idx = self.filter_basis(gamma, basis, start_state_idx,dt)
        subset_basis_indices = local_system.core_basis_indices[filter_basis_idx]
        
        #compute stable distribution on the subset of selected states
        #we need to renormalize in order to get around numerical issues
        p_stable = np.diag(local_system.compute_mixed_state_of_subset(subset_basis_indices))
        p_stable = np.maximum(p_stable,np.zeros(len(p_stable)))
        p_stable /=np.sum(p_stable) #rare edge case where the only possible state has prob zero.
        
        if False and len(filter_basis_idx) > 1:
            
            
            #ensure that p_stable is stable distribution of gamma
            x = cp.Variable(p_stable.shape)
            D = cp.diag(x)
            gamma_norm = gamma/np.max(np.abs(gamma))
            gamma_off=gamma_norm-np.diag(gamma_norm)
            diagDg = cp.sum(gamma_off@D,axis=1)
            gD = gamma_off@D - cp.diag(diagDg)
            broadcast_p_stable = np.outer(p_stable, np.ones(p_stable.shape))
            pG =  cp.multiply(broadcast_p_stable,gD)
            detailed_balance = pG-pG.T
            
            f = cp.sum_squares(detailed_balance)
            prob = cp.Problem(cp.Minimize(f), 
                    [x >=0, cp.sum(x) == len(p_stable)])
            prob.solve(verbose=False, max_iter=100000)
            gamma = gamma@np.diag(x.value)
            #renormalize
            gamma = gamma - np.diag(np.sum(gamma,axis=1))
        
        return gamma, p_stable, basis
    
    
    #difference to public version is that this returns the last state
    #this is needed for the proper latching simulation in the 2D scan
    def _sensor_scan(self, v_start, v_end, resolution, state, dt = None, find_ground_state = True, use_proxy=True):
    
        self.boundaries([4,6,2,4,5,5])
        #prepare start state
        #if find_ground_state is set, we will use as initial state the ground state
        #otherwise we take the provided state as is as initial state
        if find_ground_state:
            state = self.poly_sim.find_state_of_voltage(v_start, state_hint = v_start_state_hint)
        #if the user has not supplied a scan time dt, we use the one in the model
        if dt is None:
            dt = self.scan_dt
            
        P=(v_end - v_start).reshape(-1,1)
        if use_proxy:
            sim_slice = self.slice(P, v_start, proxy=use_proxy)
        else:
            sim_slice = self
        
        
        values = np.zeros((resolution, self.tunneling_sim.sensor_sim.num_sensors))
        basis=None
        prev_core_basis = None
        for i_point,v0 in enumerate(np.linspace([0.0],[1.0], resolution)):
            if use_proxy:
                v = v0
            else:
                v = v_start + P@v0
            
            #compute local system
            local_system = sim_slice.tunneling_sim.compute_local_system(v, state, hint_is_exact = True)
            prev_core_basis = self.poly_sim.boundaries(state).labels
            #compute markov chain at the current local system
            gamma, p_stable, basis = self._compute_markov_chain(local_system, dt)
            #find current state position in basis
            state_idx = find_label(basis, state)[0]
            
            
            #sample new state states until time is up
            cumulative_step_time = 0.0
            cur_state_idx = state_idx
            step_state_idxs=[]
            step_times=[]
            max_steps = 10
            #special case: if we only have one state in gamma, we are done
            if len(gamma) == 1:
                cumulative_step_time = dt
                step_state_idxs.append(cur_state_idx)
                step_times.append(dt)
            #case that gamma has more than one entry
            for i in range(max_steps):
                #sample time to transition to another state
                rate = np.maximum(-dt*gamma[cur_state_idx,cur_state_idx],1.e-6)
                step_time = np.random.exponential(1/rate) #numpy uses inverse rate parameter.
                
                #record the time we stayed in the current state before jumping
                step_state_idxs.append(cur_state_idx)
                step_times.append(step_time)
                cumulative_step_time += step_time
                
                #if the jump falls within the interval, draw the next state
                if cumulative_step_time < dt:
                    #now compute where this jump would take us, provided it falls within time dt
                    #compute probability for each other state to jump into it
                    p_target = -gamma[cur_state_idx]/gamma[cur_state_idx,cur_state_idx]
                    p_target[cur_state_idx] = 0
                    p_target = np.maximum(p_target, np.zeros(p_target.shape))
                    p_target /=np.sum(p_target)
                    cur_state_idx = np.random.choice(range(len(p_target)), p = p_target)
                else: 
                    break #we are done
                
            #if we exceeded the steps, we assume that overall step
            #size is small and just compute the sensor average
            #and draw a random next state
            if (cumulative_step_time < dt):
                values[i_point] = local_system.sample_sensor_equilibrium()
                #to prevent numerical difficulties in expm, we rate-limit gamma
                #for this, we scale the matrix such that the maximum diagonal element
                #doe snot exceed a certain rate. this keep the stable distribution invariant
                max_elem = np.max(np.abs(gamma))
                gamma_factor = dt*np.minimum(max_elem, 1e2/dt)/max_elem
                P_transition = scipy.linalg.expm(gamma_factor*gamma)
                p = P_transition[cur_state_idx]
                p = np.maximum(p,np.zeros(len(p)))
                p /= np.sum(p)
                cur_state_idx = np.random.choice(range(len(p)), p = p)
            else: #compute the time integrated average
                sampled_values = np.array([local_system.sample_sensor_configuration(basis[s]) for s in step_state_idxs])
                step_times = np.array(step_times)
                step_times[-1] = dt - np.sum(step_times[:-1]) #take upper limit of integration into account
                step_times /= dt #normalize for average
                values[i_point] = step_times@sampled_values
            #transform to state label
            state = basis[cur_state_idx]
        return values, state
    def sensor_scan(self, v_start, v_end, resolution, state, dt = None, find_ground_state = True, use_proxy=True):
        return _sensor_scan(v_start, v_end, resolution, state, dt, find_ground_state, use_proxy)[0]
        
    def sensor_scan_2D(self, v_offset, P, minV, maxV, resolution, state, dt = None, find_ground_state = True, use_proxy=True):
        if P.shape[1] != 2:
            raise ValueError("P must have two columns")
        if isinstance(resolution, int):
            resolution = [resolution, resolution]
        
        #obtain initial guess for state
        if find_ground_state:
            state = self.poly_sim.find_state_of_voltage(v_offset+P@minV, state_hint = state)
        
        #now slice down to 2D for efficiency
        sim_slice = self.slice(P, v_offset, proxy=True)
            
        values=np.zeros((resolution[0],resolution[1], self.tunneling_sim.sensor_sim.num_sensors))
        for i,v2 in tqdm(enumerate(np.linspace(minV[1],maxV[1],resolution[1]))):
            v_start = np.array([minV[0],v2])
            v_end = np.array([maxV[0],v2])
            values[i], state = sim_slice._sensor_scan(v_start, v_end, resolution[0], state, dt=dt, find_ground_state = False, use_proxy=False)
        return values
