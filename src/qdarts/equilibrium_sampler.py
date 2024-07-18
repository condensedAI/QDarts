import cvxpy as cp
import numpy as np
import scipy

class EquilibriumSystemSampler:
    def __init__(self, sim, reservoir_coupling, p_thresh=1.e-5):
        self.sim = sim
        self.reservoir_coupling = reservoir_coupling
        self.p_thresh = p_thresh

    def transition_prob(self, v, dt, state_hint):
        local_system = self.sim.compute_local_system(v, state_hint)
        Gamma, p, basis = self._compute_markov_chain(local_system)
        P = scipy.linalg.expm(dt*Gamma)
        return (P*p.reshape(-1,1)), basis
    def _compute_markov_chain(self, local_system):
        #compute relevant 
        p = np.diag(local_system.mixed_state)
        pos_p = p>self.p_thresh
        p = p[pos_p]/np.sum(p[pos_p])
        basis = local_system.basis_labels[pos_p]
        if len(p)==1:
            return np.zeros((1,1)),p, basis
        
        H = local_system.H[pos_p,:][:,pos_p]
        # compute the coupling matrix for all state pairs in the basis.
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
        #compute transition matrix

        #this is a bit hacky since i did not find the total model.
        #It appears that on a double dot with zero detuning, we have
        #that tunnel rate is tc/h and then it is kinda known/assumed that 
        #as detuning increases in any direction, we have ~exponential decay.
        #i assume this comes from the observation that tunnel rate is measured
        #as a back<->forth, which is based on the probability to be in a certain configuration
        #times the probability to switch, while for Gamma we are (for now) ONLY interested in the probability 
        #to switch. Let i,j be the source and target state so we have i->j. I am modeling the exponential decay part
        #as p(j)/(p(i)+p(j)), meaning that if j is much less likely than i, then the transition rate will be small
        #and vice versa.

        eV = 1.602e-19
        h_in_ev = 1.055e-34*2*np.pi/eV
        Gamma = tunnel_matrix/h_in_ev*2*p.reshape(1,-1)/(p.reshape(1,-1)+p.reshape(-1,1))
        Gamma = Gamma - np.diag(np.sum(Gamma,axis=1))
        # The values above are hacky and wrong, especially since 
        # the original simulation does not even have tunnel rates to the reservoirs
        # so currently we do not fulfill that p^T expm(Gamma)=p^T
        # or more simply p^T Gamma = 0
        # Thus, sampling from the markov chain and averaging the sensor signal would not lead to the same values
        # as the sampled equilibrium sensor response. To remedy that, we find a diagonal matrix D such, that
        # p^T D Gamma = 0
        # This solution is not unique, since we have that for any fitting D, (t*D) is a solution as well.
        # So we also add the constraint that D_ii>=0 and sum_i D_ii =1
        # by setting f = ||p^T D Gamma||^2
        # we arrive at a convex quadratic optimization problem
        # Note: for some strange reason the solution very often is 1, i.e., no corrections are needed. curious.
        x = cp.Variable(p.shape)
        D = cp.diag(x)
        f = cp.sum_squares(p@D@(Gamma/np.max(Gamma)))#note we rescale here to prevent numerical difficulties due to the extremely large values
        eps = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(f), 
                [x >=0, cp.sum(x) == len(p)])
        prob.solve(verbose=False, max_iter=100000)
        Gamma = np.diag(x.value)@Gamma
        return Gamma, p, basis
    def get_rate_matrix(self, v, state_hint):
        local_system = self.sim.compute_local_system(v, state_hint)
        gamma, p, basis = self._compute_markov_chain(local_system)
        return gamma, basis
    def sample_eqilibrium_trace(self, v, dt, N, state_hint):
        local_system = self.sim.compute_local_system(v, state_hint)
        Gamma, p, basis = self._compute_markov_chain(local_system)
        P = scipy.linalg.expm(dt*Gamma)
        
        state = np.random.choice(range(len(p)), p = p)
        states = []
        observations=[]
        for i in range(N):
            state = np.random.choice(range(len(p)), p = P[state])
            obs = local_system.sample_sensor_configuration(basis[state])
            observations.append(obs)
            states.append(basis[state].copy())
        return np.array(observations), np.array(states,dtype=int)
