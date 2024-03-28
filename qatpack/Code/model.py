import numpy as np
from numpy.distutils.misc_util import is_sequence
from util_functions import compute_maximum_inscribed_circle, compute_polytope_slacks
from polytope import Polytope

#internal unit conversion of capacitances from attoFarrad to Farrad/eV
eV = 1.602e-19
to_Farrad_per_eV = 1e-18/eV
    
class BaseCapacitanceModel:
    def __init__(self, num_dots, num_inputs, bounds_normals, bounds_limits):
        self.num_dots = num_dots
        self.num_inputs = num_inputs
        
        if not is_sequence(bounds_limits):
            bounds_limits = bounds_limits * np.ones(num_inputs)

        if bounds_normals is None:
            if num_inputs != len(bounds_limits):
                raise ValueError("if bounds_normals is not given, bounds_limits must be either a scalar or a sequence of length same as number of gates")
            bounds_normals = -np.eye(self.num_inputs)
        
        
        self.bounds_normals = np.asarray(bounds_normals)
        self.bounds_limits = np.asarray(bounds_limits)
        
    def compute_transition_equations(self, state_list, state):
        raise NotImplementedError("Implementation error: transition equations not implemented")
    
    def slice(self, P, m):
        raise NotImplementedError("Implementation error: slice not implemented")
        
    def enumerate_neighbours(self, state):
        d = state.shape[0]
        # Compute list of possible state transitions for the provided state
        # For simplicity, we restrict to only single electron additions/subtractions per dot
        # This leaves 3^d-1 states
        state_list=np.zeros((1,d),dtype=int)
        for i in range(d):
            state_list1 = state_list.copy()
            state_listm1 = state_list.copy()
            state_listm1[:,i] = -1
            state_list1[:,i] = 1
            state_list = np.vstack([state_list,state_list1])
            if state[i] >= 1:    
                state_list = np.vstack([state_list,state_listm1])

        # First element is all-zeros, we don't want it
        state_list=state_list[1:]

        return [state_list+state]
        
    def compute_polytope_for_state(self, state, maximum_slack):
        #get the potentially bacthed list of states
        state_lists = self.enumerate_neighbours(state)
        As = []
        bs = []
        transition_slacks = []
        states = []
        # Now for each of those, get the list of transitions...
        for idx,state_list in enumerate(state_lists):
            A,b = self.compute_transition_equations(state_list, state)
            
            #check, whether there are superfluous transitions
            #TODO: Oswin: i don't remember what the significance of this was.
            zero_const = np.all(np.abs(A)<1.e-8,axis=1)
            if np.any(zero_const):
                A = A[~zero_const]
                b = b[~zero_const]
                state_list = state_list[~zero_const]
            # ... and check for this batch whether we can filter out non-touching ones
            slacks = compute_polytope_slacks(A,b, maximum_slack)
            keep = slacks <= maximum_slack+1.e-8
            
            #if we have kept nothing, this means there is a set of equations that is not fullfillable
            #this happens often when slicing, e.g, a polytope is not within the sliced subspace.
            if not np.any(keep):
                return Polytope(state)
                
            As.append(A[keep])
            bs.append(b[keep])
            transition_slacks.append(slacks[keep])
            states.append(state_list[keep])
            
        
        # Keep iterating over the list, merging facets, until they are all merged
        while len(As)>1:
            # Take the next set of As, bs and states, and merge them into a new set
            A = np.vstack(As[:2])
            b = np.concatenate(bs[:2])
            max_slack = np.concatenate(max_slacks[:2])
            state = np.vstack(states[:2])
            # Update the lists; we've now taken care of another set of two
            As=As[2:]
            bs=bs[2:]
            states = states[2:]
            max_slacks = max_slacks[2:]
            transition_slacks = transition_slacks[2:]
            
            # Handle possible duplicate transitions
            state, indxs = np.unique(state, axis=0, return_index = True)
            A = A[indxs]
            b = b[indxs]
            
            # Find transitions in the merged sets
            slacks= self._check_transition_existence(A,b, max_slack)
            keep = slacks <= maximum_slack+1.e-8
            
            #if we have kept nothing, this means there is a set of equations that is not fullfillable
            #this happens often when slicing, e.g, a polytope is not within the sliced subspace.
            if not np.any(keep):
                return Polytope(state)

            # Add the merged ones back to the list
            As.append(A[keep])
            bs.append(b[keep])
            max_slacks.append(max_slack[keep])
            transition_slacks.append(slacks[keep])
            states.append(state[keep])
        
        #create final polytope
        poly = Polytope(state)
        touching = transition_slacks[0]<1.e-8
        point_inside, _ = compute_maximum_inscribed_circle(As[0][touching], bs[0][touching], self.bounds_normals, self.bounds_limits)
        poly.set_polytope(states[0] - state, As[0], bs[0], transition_slacks[0], point_inside)
        return poly
        
    def verify_polytope(self, polytope, maximum_slack):
        if not polytope.must_verify:
            return polytope
        slacks = compute_polytope_slacks(polytope.A, polytope.b, maximum_slack)
        keep = slacks <= maximum_slack + 1.e-8
        touching = slacks <= 1.e-6
        point_inside, _ = compute_maximum_inscribed_circle(polytope.A[touching], polytope.b[touching], self.bounds_normals, self.bounds_limits)
        
        verified_polytope = Polytope(polytope.state)
        verified_polytope.set_polytope(
            polytope.labels[keep],
            polytope.A[keep], polytope.b[keep],
            slacks[keep],
            point_inside
        )
        return verified_polytope

class Model(BaseCapacitanceModel):
    def __init__(self, config):
        self.__init__(config.C_g, config.C_D, config.ks, config.bounds_limits, config.bounds_normals, config.transform_C_g, config.offset)

    def __init__(self, C_g, C_D, bounds_limits,  bounds_normals=None, transform_C_g = None, offset = None, ks = None):
        
        # Set the transformation matrix to the identity if not provided
        if transform_C_g is None:
            transform_C_g = np.eye(C_g.shape[1])
        
        super().__init__(C_D.shape[0], transform_C_g.shape[1], bounds_normals, bounds_limits)
        
        # Set instance properties
        self.C_g_atto = np.asarray(C_g)
        self.C_D_atto = np.asarray(C_D)
        self.transform_C_g = np.array(transform_C_g)

        # Check that an offset is provided for every dot
        self.offset = np.zeros(self.num_dots)
        if offset is not None:
            if len(offset) != self.num_dots:
                raise ValueError("The offset you provided does not have an offset for every dot.")
            self.offset = np.array(offset)

        # Convert units from attoFarrad to Farrad per eV
        self.C_g = self.C_g_atto * to_Farrad_per_eV
        self.C_D = self.C_D_atto * to_Farrad_per_eV
                    
        # Check if value for non-constant capacitance is provided
        # TODO: ks now assumed the same for all dots
        self.ks = ks
        if ks is not None:
            self.ks = np.array(ks)
            #if np.any(ks<1):
            #    raise ValueError("The ks values must be larger than 1")
         
            # TODO: What are S values?
            # Cache S values for up to 1000 total dots 
            self.S_cache=np.zeros((1000,self.num_dots))
            self.S_cache[0,:] = 1
            self.S_cache[1,:] = 1
            
            r = 2.6
            alphas = 1-0.137*(1+r)/(ks+r)
            for n in range(2,self.S_cache.shape[0]):
                Sprev = self.S_cache[n-2]
                self.S_cache[n] = n/(2*alphas*(ks+2)/(n+ks)+(n-2)/Sprev)
        
    def _compute_capacitances(self, state):
        N = len(state)
        
        S = np.eye(N)
        if self.ks != None:
            S_values = self.S_cache[state,range(N)]
            S = np.diag(S_values)

        sum_C_g = np.sum(self.C_g,axis=1)

        # General transform by changing dot capacitances
        Cn_g = S @ self.C_g
        Cn_D = S @ self.C_D @ S
        Csum = S @ S @ sum_C_g + np.sum(Cn_D,axis=1)+np.diag(Cn_D)

        Cn_inv = np.linalg.inv(np.diag(Csum) - Cn_D)
        return Cn_inv, Cn_g

    def compute_transition_equations(self, state_list, state_from):
        """
        Computes the normals and offsets for facets that separate 
        the state `state_from` and the states in `state_list`.
        """
        # Get number of targets
        N = state_list.shape[0]

        # Compute normal and offset for the from_state
        C0_inv, C0_g = self._compute_capacitances(state_from)
        q0 = state_from @ C0_inv
        A0 = q0 @ C0_g
        b0 = q0 @ state_from -2*q0 @ C0_g @ self.offset   
        
        # Now compute the normals and offsets for the target states
        A = np.zeros((N, self.num_inputs))
        b = np.zeros(N)
        for i,n in enumerate(state_list):
            # Compute the capacitances for the target state
            Cn_inv, Cn_g = self._compute_capacitances(n)
            qn = n @ Cn_inv
            An = qn @ Cn_g

            # Compute the normal
            A[i] = (An - A0)@self.transform_C_g
            # Compute the offset
            b[i] = (b0 - qn @ n)/2 + qn @ Cn_g @ self.offset
            
        return A,b
    
    def slice(self, P, m):
        new_offset = self.offset+self.transform_C_g@m
        new_transform = self.transform_C_g@P
        
        new_boundsA = self.bounds_normals@P
        new_boundsb = self.bounds_limits + self.bounds_normals@m
        
        #throw out almost orthogonal bounds
        sel = np.linalg.norm(new_boundsA,axis=1)>1.e-7*np.linalg.norm(self.bounds_normals,axis=1)
        new_boundsA = new_boundsA[sel]
        new_boundsb = new_boundsb[sel]
        
        return  Model(self.C_g_atto, self.C_D_atto, new_boundsb, new_boundsA, new_transform, new_offset, ks= self.ks)
