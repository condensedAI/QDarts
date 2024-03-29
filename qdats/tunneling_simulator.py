import numpy as np
from functools import partial
from scipy  import sparse as sp
import scipy.stats as stats

def softmax(v,axis=None):
    max_v = np.max(v)
    y = np.exp(v-max_v)
    return y/np.sum(y,axis)

# def sigmoid(x):
#     return 1.0/(1.0+np.exp(np.minimum(-x,200)))
    
# def logsumexp(v,axis=None):
#     max_v = np.max(v,axis)
#     y = np.exp(v-max_v)
#     return np.log(np.sum(y,axis)) + max_v
    
class BaseSensorSim:
    def __init__(self, num_sensors):
        self.num_sensors = num_sensors
    def precompute_sensor_state(self, state, A, b, labels):
        return None
    def eval_sensor(self, H, rho, sensor_state):
        raise NotImplementedError('eval_sensor is not implemented')
    
class NoisySensorDot(BaseSensorSim):
    def __init__(self, sensor_dot_ids):
        super().__init__(len(sensor_dot_ids))
        self.sensor_dot_ids = sensor_dot_ids
        self.g_max = 1
        self.fast_noise_var = 0.0
        self.n_virtual_samples = 1
        self.peak_width_multiplier = 1
        self.slow_noise_gen=None

    def config_noise(self, sigma, n_virtual_samples, slow_noise_gen = None):
        self.fast_noise_var = sigma**2
        self.n_virtual_samples = n_virtual_samples
        self.slow_noise_gen = slow_noise_gen
        
    def config_peak(self, g_max, peak_width_multiplier):
        self.g_max = g_max
        self.peak_width_multiplier = peak_width_multiplier
    
    def precompute_sensor_state(self, state, A, b, labels):
        sensor_state ={}
        for i, sensor_id in enumerate(self.sensor_dot_ids):
            labels_nosens = np.delete(labels,sensor_id, axis=1)
            labels_unique, inverse_index = np.unique(labels_nosens, return_inverse=True,axis=0)

            labels_sens = labels[:,sensor_id]
            sorted_ind = np.lexsort((labels_sens,inverse_index))

            relevant_label_indices = []
            prev = []
            next = []
            cur = -1
            last = None
            last_2 = None
            for ind in sorted_ind:
                l = labels_nosens[ind]
                if np.any(l != cur):
                    cur = l
                    last  = None
                    last_2 = None
                else:
                    if not last_2 is None:
                        relevant_label_indices.append(last)
                        prev.append(last_2)
                        next.append(ind)
                    last_2 = last
                    last = ind
            terms = np.array(relevant_label_indices,dtype=int)
            prev = np.array(prev,dtype=int)
            next = np.array(next,dtype=int)
            sensor_state[sensor_id] = (terms,prev,next)
        return sensor_state
    
    def eval_sensor(self, H, mixed_state, sensor_state, beta):
        results = np.zeros(len(self.sensor_dot_ids))
        for i, sensor_id in enumerate(self.sensor_dot_ids):
            terms,neighbour_prev, neighbour_next = sensor_state[sensor_id]
            
            #get probability of each state of the array
            p = np.diag(mixed_state)[terms]
            #compute sensor detuning between every state and their neighbour
            eps_prev = np.abs(np.diag(H)[terms]-np.diag(H)[neighbour_prev])
            eps_next = np.abs(np.diag(H)[terms]-np.diag(H)[neighbour_next])
            eps = np.minimum(eps_prev,eps_next)
            #add noise
            if self.slow_noise_gen is None:
                eps = eps[:,None]
            else:
                eps = (eps[:,None]+self.slow_noise_gen()[None,:])
            if self.fast_noise_var > 0:
                fast_noise = np.random.randn(*eps.shape)*np.sqrt(self.fast_noise_var)
                eps += fast_noise
            eps *=beta 
            #simulate the fast noise around the peak. for this we approximate the logistic shape of the peak by a normal
            #peak and compute the exact noise and variance assuming that the fast noise is uncorrelated normal distributed
            var_peak = 0
            #var_peak = self.fast_noise_var*beta**2   #NOTE: We give fast noise in ueV
            var_logistic = (1/0.631*self.peak_width_multiplier)**2
            # compute first and second moment of the noise. by law of large numbers, as n_virtual_samples->large
            # the avg will become normally distributed.
            norm_pdf = lambda x, mu,var: 1/np.sqrt(2*np.pi*var)*np.exp(-(x-mu)**2/(2*var))
            mean_g = 4*norm_pdf(0,eps, var_peak + var_logistic)
            second_moment_g =  16*norm_pdf(0,eps, (self.fast_noise_var + var_logistic/2))* 0.5/np.sqrt(np.pi*var_logistic)
            std_g = np.sqrt(np.abs(second_moment_g - mean_g**2)/self.n_virtual_samples)
            

            #now sample from the approximation
            g = beta*self.g_max*(mean_g + std_g*np.random.randn(*eps.shape))
            #average over the probability of each sample
            results[i] = np.sum(p*np.mean(g,axis=1))/np.sum(p)
        return results
        
class ApproximateTunnelingSimulator:
    def __init__ (self, polytope_sim, tunnel_matrix, T, sensor_sim):
        self.poly_sim = polytope_sim
        self.tunnel_matrix = tunnel_matrix
        eV = 1.602e-19
        kB = 1.380649e-23/eV
        self.beta=1.0/(kB*T)
        self.T = T
        self.sensor_sim = sensor_sim
        
        #clean up potentially stored conflicting data
        for poly in self.poly_sim.cached_polytopes():
            poly.additional_info.pop("features_out_info",None)
            
        self.num_additional_neighbours=np.zeros(self.poly_sim.num_dots, dtype=int)
            
    def slice(self, P, m, proxy=False):
        sliced_poly_sim = self.poly_sim.slice(P,m, proxy)
        sliced_features_out_sim = ApproximateTunnelingSimulator(sliced_poly_sim, self.tunnel_matrix, self.T, self.sensor_sim)
        sliced_features_out_sim.num_additional_neighbours = self.num_additional_neighbours.copy()
        return sliced_features_out_sim
        
    def _compute_tunneling_op(self, state_list):
        
        N = state_list.shape[0]
        n_dots = state_list.shape[1]
        TOp = np.zeros((N,N),dtype=int)*n_dots**2 #by definition of self.tunnel_matrix the first element of the array is 0
        
        sums = np.sum(state_list,axis=1)
        for i,s1 in enumerate(state_list):
            for j,s2 in zip(range(i+1, len(state_list)),state_list[i+1:]):
                if sums[i] != sums[j]:
                    continue
                
                if np.sum(np.abs(s1-s2)) == 0:
                    continue
                abs_diff = np.abs(s1-s2)
                if np.sum(abs_diff) != 2:
                    continue
                
                #compute lookup indices in features_outing strength matrix
                #if only one index is there this means we have a transition of a dot that is connected to a reservoir.
                idxs = np.where(abs_diff>0)[0]
                if len(idxs) == 1:
                    ind = idxs[0]*n_dots + idxs[0]
                else:
                    ind = idxs[0]*n_dots + idxs[1]
                TOp[i,j] = ind
                TOp[j,i] = ind
        return TOp
        
    def _create_state_list(self, state, direct_neighbours):
        state_list = np.vstack([direct_neighbours, [np.zeros(len(state),dtype=int)]])
        
        additional_states = []
        for i in range(self.poly_sim.num_dots):
            e_i = np.eye(1, self.poly_sim.num_dots, i,dtype=int)
            for k in range(1,1+self.num_additional_neighbours[i]):
                additional_states.append(state_list+k*e_i)
                additional_states.append(state_list-k*e_i)

        if len(additional_states) > 0:
            for add in additional_states:
                state_list = np.vstack([state_list,add])
            state_list = np.unique(state_list, axis=0)
        state_list += state[None,:]
        state_list = state_list[np.all(state_list>=0,axis=1)]
        return state_list
    def _get_polytope(self, state):
        state = np.asarray(state)
        polytope = self.poly_sim.boundaries(state)
        #cache features_out info in polytope structure
        if not "extended_polytope" in polytope.additional_info.keys():
            #create a list of all neighbour states of interest for use in the Hamiltonian
            state_list = self._create_state_list(state, polytope.labels)
            
            
            #create full set of transition equations
            A,b = self.poly_sim.compute_transition_equations(state_list, state)
            
            TOp = self._compute_tunneling_op(state_list)
            extended_polytope = status=type('',(object,),{})()
            extended_polytope.A = A
            extended_polytope.b = b
            extended_polytope.TOp = TOp
            extended_polytope.labels = state_list
            polytope.additional_info["extended_polytope"] = extended_polytope
            
            #also compute the sensor info
            polytope.additional_info['sensor_state'] = self.sensor_sim.precompute_sensor_state(state, A, b, state_list)
        return polytope
        
    
    def _compute_mixed_state(self, H):
        diffs = np.diag(H)-np.min(np.diag(H))
        sel = np.where(diffs<2*self.poly_sim.get_maximum_polytope_slack())[0]
        H_sel = H[:,sel][sel,:]
        
        eigs, U = np.linalg.eigh(H_sel)
        ps = softmax(-eigs * self.beta)
        rho_sel = U @ np.diag(ps) @ U.T
        
        rho = np.zeros(H.shape)
        indizes = H.shape[0]*sel[:,None]+sel[None,:]
        np.put(rho,indizes.flatten(), rho_sel.flatten())
        return rho
        
    def _create_hamiltonian(self, v, A, b, tunnel_matrix, TOp):
        N = A.shape[0]
        energy_diff = -(A@v+b)
        diags = np.sort(energy_diff)
        if tunnel_matrix is None:
            return np.diag(energy_diff)
        else:
            t_term = ((tunnel_matrix.reshape(-1)[TOp.reshape(-1)]).reshape(N,N))
            return np.diag(energy_diff)-t_term
    
    def sensor_scan(self, v_start, v_end, resolution, v_start_state_hint, use_proxy=True):
        #prepare start state
        state = self.poly_sim.find_state_of_voltage(v_start, state_hint = v_start_state_hint)
        
        P=(v_end - v_start).reshape(-1,1)
        if use_proxy:
            sim_slice = self.slice(P, v_start, proxy=use_proxy)
        else:
            sim_slice = self
        
        
        polytope = sim_slice._get_polytope(state)
        values = np.zeros((resolution, self.sensor_sim.num_sensors))
        for i,v0 in enumerate(np.linspace([0.0],[1.0], resolution)):
            if use_proxy:
                v = v0
            else:
                v = v_start + P@v0
            if not sim_slice.poly_sim.inside_state(v, state):
                state = sim_slice.poly_sim.find_state_of_voltage(v,state_hint=state)
                polytope = sim_slice._get_polytope(state)
            
            #compute mixed state of the current voltage configuration
            extended_polytope = polytope.additional_info['extended_polytope']
            H = sim_slice._create_hamiltonian(v, extended_polytope.A, extended_polytope.b, self.tunnel_matrix, extended_polytope.TOp)
            mixed_state = sim_slice._compute_mixed_state(H)
            
            #compute sensor response for the mixed state
            sensor_state = polytope.additional_info['sensor_state']
            values[i] = sim_slice.sensor_sim.eval_sensor(H, mixed_state, sensor_state, self.beta)
        return values
        
    def sensor_scan_2D(self, v_offset, P, minV, maxV, resolution, state_hint_lower_left):
        if P.shape[1] != 2:
            raise ValueError("P must have two columns")
        if isinstance(resolution, int):
            resolution = [resolution, resolution]
        
        #obtain initial guess for state
        line_start = self.poly_sim.find_state_of_voltage(v_offset+P@minV, state_hint = state_hint_lower_left)
        
        #now slice down to 2D for efficiency
        sim_slice = self.slice(P, v_offset, proxy=True)
            
        values=np.zeros((resolution[0],resolution[1], self.sensor_sim.num_sensors))
        for i,v2 in enumerate(np.linspace(minV[1],maxV[1],resolution[1])):
            v_start = np.array([minV[0],v2])
            v_end = np.array([maxV[0],v2])
            line_start = sim_slice.poly_sim.find_state_of_voltage(v_start, state_hint = line_start)
            values[i] = sim_slice.sensor_scan(v_start, v_end, resolution[0], line_start, use_proxy=False)
        return values