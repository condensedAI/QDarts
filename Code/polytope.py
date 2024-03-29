import numpy as np
from util_functions import is_invertible_matrix, solve_linear_problem, compute_polytope_slacks, compute_maximum_inscribed_circle, find_label, find_point_on_transitions

class Polytope:
    """ Represents an arbitrary polytope in the form of a set of linear inequalities, 
        and provides several convenience methods for working with them.
    """
    def __init__(self, state):
        #empty polytope
        self.state = state
        self.labels = np.array([])
        self.A = np.array([])
        self.b = np.array([])
        self.slacks = np.array([])
        self.point_inside = np.array([])
        self.must_verify = False
        self.additional_info={}
        
    def set_polytope(self, labels, A, b, slacks, point_inside, must_verify = False):
        self.labels = labels
        self.A = A
        self.b = b
        self.slacks = slacks
        self.point_inside = point_inside
        self.must_verify = must_verify
    
    def lazy_slice(self, P, m):
        sliced = Polytope(self.state)
        if self.A.shape[0] == 0:
            sliced.set_polytope(self.labels, self.A, self.b, np.array([]), None, False)
        else:
            sliced.set_polytope(
                self.labels,
                self.A@P, #we know the line equations
                self.b+self.A@m, #and their offsets
                None, None, #but nothing else
                True #user must verify this later.
            )
        return sliced
    
    def invertible_transform(self, P, m):
        if self.must_verify:
            return self.lazy_slice(P, m)
    
        transformed = Polytope(self.state)
        
        transformed_point_inside = self.point_inside@P+m
        transformed.set_polytope(
            self.labels,
            self.A@P, #we know the line equations
            self.b+self.A@m, #and their offsets
            self.slacks, #slacks are in units of energy, that is constant under invertible transforms.
            transformed_point_inside
        )
        return transformed