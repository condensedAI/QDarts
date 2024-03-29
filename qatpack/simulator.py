import numpy as np
from scipy.linalg import sqrtm
from numpy.distutils.misc_util import is_sequence
from util_functions import is_invertible_matrix, find_label, find_point_on_transitions, compute_polytope_slacks, compute_maximum_inscribed_circle, fix_gates, axis_align_transitions
from polytope import Polytope
from model import Model
import time

class BaseCapacitiveDeviceSimulator:
    def __init__(self, capacitance_model):
        self.capacitance_model = capacitance_model
        self.cache = {}
        
        # Make available the number of dots and gates
        self.num_inputs = capacitance_model.num_inputs
        self.num_dots = capacitance_model.num_dots
    
    def slice(self, P, m, proxy=False):
        """
        Computes the slice through the device by setting v=m+Pv', where v is the plunger gate voltages of the 
        original device and v' is the new coordinate system. Must be implemented by derived classes
        """
        raise NotImplementedError('slice: Derived class must implement this method.')
    
    def compute_polytope(self, state):
        """
        Computes the polytope for a given state. Is implemented by the derived class and claled when the 
        polytope for a state is not found in cache.
        """
        raise NotImplementedError('ceate_polytope: Derived class must implement this method.')
        
    def compute_transition_equations(self, state_list, state_from):
        """
        For a given state and list of neighbour states, computes the linear equations that compute the energy differences
        Between the target state_from and the other states.
        """
        return self.capacitance_model.compute_transition_equations(state_list,state_from)
        
    def get_maximum_polytope_slack(self):
        """
        Returns the maximum distance the closest point of a transition can have to the polytope
        before it is discarded. Setting to 0 means that only transitions that actually touch the polytope
        are kept.
        """
        raise NotImplementedError('get_maximum_polytope_slack: Derived class must implement this method.')
    
    def set_maximum_polytope_slack(self, maximum_slack):
        """
        Sets the maximum distance the closest point of a transition can have to the polytope
        before it is discarded. Setting to 0 means that only transitions that actually touch the polytope
        are kept.
        """
        raise NotImplementedError('get_maximum_polytope_slack: Derived class must implement this method.')
    
    def cached_polytopes(self):
        """
        Returns a sequence including all computed and cached polytopes for inspection and modification.
        """
        return self.cache.values()
        
    def boundaries(self, state):
        """
        Returns the polytope of a given state with all its boundaries, labels and meta information.
        """
        # Convert to array to be sure
        state = np.asarray(state).astype(int)
        
        # lookup key of this state
        dict_key = tuple(state.tolist())
        # See if we already have this key in our prepared list
        if not dict_key in self.cache.keys():
            self.cache[dict_key] = self.compute_polytope(state)
        
        #obtain polyope from dict
        polytope = self.cache[dict_key]
        
        #slice is allowed to be lazy but then we need to verify the polytope now.
        if polytope.must_verify:
           polytope = self.capacitance_model.verify_polytope(polytope,self.get_maximum_polytope_slack())
           self.cache[dict_key] = polytope
        
        return polytope
    
    
    def inside_state(self, v, state):
        """
        Returns true if a point v is fully within the currently active polytope. 
        Excluding the boundary.
        """
        polytope = self.boundaries(state)
        if len(polytope.labels) == 0:
            return False
        f = polytope.A@v + polytope.b
        return np.all(f < 1.e-8)
    
    def find_boundary_intersection(self, old_v, new_v, state, epsilon=1.e-6, deep_search=True):
        """
        For a given state and a voltage old_v within this state and a point new_v outside this state,
        computes the intersection with the boundary of the polytope on the line between old_v and new_v. 
        the intersection point and new target state is computed. Epsilon computes the precision of the computed voltage.
        Should be a small positive value to pevent numerical problems. Deep_search: whether an iterative search is performed for
        the new point in case non eof the direct neighbours of the polytope match. If false, will throw an exception in that case.
        An exception is also raised when the deep search failed.
        """
        if not self.inside_state(old_v,state):
            raise ValueError("old_v must be in the provided state.")
        
        polytope = self.boundaries(state)
        
        direction = new_v - old_v
        direction /= np.linalg.norm(direction)
                
        A_line = polytope.A @ direction
        b_line = polytope.b + polytope.A @ old_v
        positive = np.where(A_line > 0)[0]
        ts = -b_line[positive]/A_line[positive]
        transition_idx = np.argmin(ts)
        
        #construct point of cosest hit
        transition_state = state + polytope.labels[positive[transition_idx]]
        v_intersect = old_v + (1+epsilon)*ts[transition_idx]*direction
        if self.inside_state(v_intersect, transition_state):
            return transition_state, v_intersect
        
        #the new point might have went through a corner, so we check all states whose transitions are now violated

        rel_energy = polytope.A@v_intersect+polytope.b
        idx_order = np.argsort(rel_energy)
        for idx in idx_order:
            #pass 1: ignore transitions that don't touch the polytope.
            if polytope.slacks[idx]>1e-6:
                continue
            if rel_energy[idx] < -1.e-8:
                continue
            transition_state = state + polytope.labels[idx]
            if self.inside_state(v_intersect, transition_state):
                return transition_state, v_intersect
                
        for idx in idx_order:
            #pass 2: now try the near-hits
            if polytope.slacks[idx]<1e-6:
                continue
            if rel_energy[idx] < -1.e-8:
                continue
            transition_state = state + polytope.labels[idx]
            if self.inside_state(v_intersect, transition_state):
                return transition_state, v_intersect
        if not self.inside_state(v_intersect, transition_state):
            if deep_search == False:
                print(old_v, new_v, state)
                raise LookupError()
            
            transition_state = self.find_state_of_voltage(new_v, state, deep_search= False)
            
        return transition_state, v_intersect
    
    def find_state_of_voltage(self,v,state_hint, deep_search=True):
        """
        For a given state voltage, computes the state for which is within the polytope of the state.
        state_hint: a likely candidate for the state. The polytope of the state must not be empty (can only happen when slicing)
        """
        state = state_hint
        polytope = self.boundaries(state)
        if len(polytope.labels) == 0:
            raise ValueError("polytope of state_hint does not intersect with plane")
       
        # Check if hint was correct
        # If not we have to search.
        # We hope that the solution is close enough and find the transitions
        v_inside = polytope.point_inside.copy()
        while not self.inside_state(v,state):
            state,v_inside = self.find_boundary_intersection(v_inside, v, state, deep_search = deep_search)
        
        return state
    
class CapacitiveDeviceSimulator(BaseCapacitiveDeviceSimulator):
    """
    This class simulates a quantum dot device based on a capacitance model.
    
    The simulator interally keeps track of the Coulomb diamonds (polytopes) and their transitions (facets),
    and takes care of keeping track of which transitions are feasible, with what precision, etc.
    This allows one to ask questions such as: "which transition does this facet correspond to?" and 
    "what is the orthogonal axis in voltage space (i.e. virtual gate) that tunes across it?". 
    The simulator will return, for each transition, a point on the transition line and the virtual gate.
    
    It also has the ability to take 2D slices through high dimensional voltage spaces to construct 2D 
    projections of charge stability diagrams.
    """
    
    def __init__(self, capacitance_model):
        super().__init__(capacitance_model)
        self.maximum_slack = 0.0
    
    def compute_polytope(self, state):
        return self.capacitance_model.compute_polytope_for_state(state,self.maximum_slack)
        
    def get_maximum_polytope_slack(self):
        """
        Returns the maximum distance the closest point of a transition can have to the polytope
        before it is discarded. Setting to 0 means that only transitions that actually touch the polytope
        are kept.
        """
        return self.maximum_slack
    
    def set_maximum_polytope_slack(self, maximum_slack):
        """
        Sets the maximum distance the closest point of a transition can have to the polytope
        before it is discarded. Setting to 0 means that only transitions that actually touch the polytope
        are kept.
        """
        self.maximum_slack = maximum_slack
        self.cache={}
        
    def slice(self, P, m, proxy=None):
        """
        Computes a simulator that is given when one exchanges the gate voltage sv by v=Pv'+m. P can have less columns than the number of gates
        in which case the polytopes of the returned simulation are slices through the original polytopes.
        """
        
        #if proxy is not set, we check whether P is invertible
        #if it is invertible, then reusing the cache is the most efficient
        #in the general case where we don't know whether the original simulator
        #will be used still.
        
        #checking invertibility also allows us to quickly transform the cache
        is_invertible = is_invertible_matrix(P)
        if proxy is None:
            proxy = is_invertible
        
        if proxy == True:
            sliced_proxy =  CapacitiveDeviceSimulatorProxy(self, P, m)
            return sliced_proxy
        else:
            sliced_simulator = CapacitiveDeviceSimulator(self.capacitance_model.slice(P,m))
            sliced_simulator.maximum_slack = self.maximum_slack
            #slice all precomputed polytopes in a lazy manner.
            for key, polytope in self.cache.items():
                if is_invertible:
                    sliced_simulator.cache[key] = polytope.invertible_transform(P, m)
                else:
                    sliced_simulator.cache[key] = polytope.lazy_slice(P, m)
            
            return sliced_simulator

class CapacitiveDeviceSimulatorProxy(BaseCapacitiveDeviceSimulator):

    """
    This class is a slice proxy for the CapacitiveDeviceSimulator class. It gets returned by
    any slice operation, when a "soft" slice is needed. This is unlikely to be used by the user 
    directly and mostly used during plotting. The advantage of a soft slice is that it can make better use of 
    caching at the expense of higher computation cost: all queries for polytopes are computed by the original simulator
    and thus if several different slices of the same simulator are needed, they can share computed polytopes.
    """
    def __init__(self, simulator, P, m):
        super().__init__(simulator.capacitance_model.slice(P,m))
        self.simulator = simulator
        self.P = P
        self.m = m
        
    def get_maximum_polytope_slack(self):
        """
        Returns the maximum distance the closest point of a transition can have to the polytope
        before it is discarded. Setting to 0 means that only transitions that actually touch the polytope
        are kept.
        """
        return self.simulator.get_maximum_polytope_slack()
    
    def set_maximum_polytope_slack(self, maximum_slack):
        """
        Sets the maximum distance the closest point of a transition can have to the polytope
        before it is discarded. Setting to 0 means that only transitions that actually touch the polytope
        are kept.
        """
        self.simulator.set_maximum_polytope_slack(maximum_slack)
        self.cache={}
    
    def compute_polytope(self, state):
        #query or compute original polytope
        polytope = self.simulator.boundaries(state)
        
        #transform lazyly
        
        polytope_sliced = polytope.lazy_slice(self.P, self.m)
        t0 = time.time()
        polytope_sliced = self.capacitance_model.verify_polytope(polytope_sliced,self.get_maximum_polytope_slack())
        return polytope_sliced
       
    def slice(self, P, m, proxy=None):
        if proxy is None:
            proxy = True
       
        if proxy == True:
            return CapacitiveDeviceSimulatorProxy(self,P, m)
        else:
            new_P = self.P@P
            new_m = self.m + self.P@m
            return self.simulator.slice(new_P, new_m, False)
  


