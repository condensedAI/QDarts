import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from matplotlib import pyplot as plt
from numpy.distutils.misc_util import is_sequence
#all code here is required for plotting in the provided notebook

#function that finds a point inside a given polytope
def find_feasible_point(halfspaces):
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds = (None,None))
    return res.x[:-1]


def plot_2D_polytope(ax,A,b,color,lower_bounds, label=None, linestyle='-',linewidth=1):
    eqs=np.hstack([A,b.reshape(-1,1)])
    #add constraints to polytope
    eqs = np.vstack([eqs,lower_bounds])
    
    # Get the corners
    
    feasible_point = find_feasible_point(eqs)
    if not np.all(eqs[:,:-1]@feasible_point+eqs[:,-1] < 1.e-5):
        return None, None
    corners = HalfspaceIntersection(eqs, feasible_point).intersections

    # Easiest way to get proper line-segments of the boundaries in 2D is via the convex hull
    # Form a convex hull based on the corners
    hull = ConvexHull(corners)
    # Plot lines between the points forming the hull (they are assumed to come in counterclockwise order)
    for i,simplex in enumerate(hull.simplices):
        label_trace = label
        if i != 0:
            label_trace = None
        ax.plot(corners[simplex, 0], corners[simplex, 1], linestyle, c=color, lw=linewidth,label=label_trace)
    return feasible_point, (np.min(corners,axis=0),np.max(corners,axis=0))


def get_2D_polytope(A,b,color,lower_bounds, label=None, linestyle='-',linewidth=1):
    eqs=np.hstack([A,b.reshape(-1,1)])
    #add constraints to polytope
    eqs = np.vstack([eqs,lower_bounds])
    
    # Get the corners
    
    feasible_point = find_feasible_point(eqs)
    if not np.all(eqs[:,:-1]@feasible_point+eqs[:,-1] < 1.e-5):
        return None, None
    corners = HalfspaceIntersection(eqs, feasible_point).intersections

    # Easiest way to get proper line-segments of the boundaries in 2D is via the convex hull
    # Form a convex hull based on the corners
    return corners
    


def raster_CSD_states(simulation, v_0, P, minV, maxV, resolution, state_hint_lower_right):
    if not is_sequence(resolution):
        resolution = [resolution,resolution]
    states=np.zeros((resolution[0],resolution[1], simulation.num_dots),dtype=int)
    line_start = simulation.find_state_of_voltage(v_0+P@np.array([minV[0],minV[1]]), state_hint_lower_right)
    for i,v1 in enumerate(np.linspace(minV[0],maxV[0],resolution[0])):
        state = line_start
        for j,v2 in enumerate(np.linspace(minV[1],maxV[1],resolution[1])):
            v = v_0+P@np.array([v1,v2])
            state = simulation.find_state_of_voltage(v,state)
            if not simulation.inside_state(v,state):
                print("error")
            if j == 0:
                line_start = state
            states[i,j] = state
    return states



def get_CSD_data(simulation, v_0, P, lower_left, upper_right, resolution, state_hint_lower_left):
    """
    Function that computes a Charge Stability Diagram from a simulation of a device.
    The function plots the states at voltages v=v_0+P*x where x is a vector with elements 
    in a box defined by its lower-left and upper-right corners given by lower_left and upper_right and 
    number of values given by the resolution. if upper_right=-lower_left then v_0 is the center
    pixel of the rastered plot.
    
    By default, a background image is plotted 
    which is based on the charge onfiguration at a position. This is overlayed with a line plot indicating the
    exact transition points between states. Optionally each region is labelled using the exact electron state.

    simulation: the device simulation to raster
    ax: matplotlib axis object to plot into
    v_0: the origin of the coordinate system to plot.
    P: coordinate system. A nx2 axis where n is the number of gates in simulation. 
    lower_left: minimum value in x for both axes
    upper_right: maximum value of x in both axes
    resolution: number of sampled points in each direction of x. either single number or one per axis.
    state_hint_lower_left: starting point to guess the initial state of the lower left corner. must not be empty within sim.
    draw_labels: whether to draw the label of the state in a region
    draw_background: whether to draw a color map of the states of the CSD.
    """
    minV = np.array(lower_left)
    maxV = np.array(upper_right)
    
    #find the true corner state before slicing the simulation. otherwise we might have trouble 
    #finding it if the state hint does not touch the projection

    lower_left_corner = v_0 + P@minV
    corner_state = simulation.find_state_of_voltage(lower_left_corner, state_hint_lower_left)
    simulation_slice = simulation.slice(P, v_0, proxy=True)
    
    #compute CSD
    states = raster_CSD_states(simulation_slice, np.zeros(2), np.eye(2), minV, maxV, resolution, corner_state)
    color_weights = np.linspace(1,2.7,simulation_slice.num_dots)
    CSD_data = 1+np.sum(color_weights.reshape(1,1,-1)*states,axis=2)
    return simulation_slice, CSD_data, states


def get_polytopes(states, simulation_slice, minV, maxV, V_offset):
    #iterate over the list of different states and plot their sliced polytope
    
    states = [tuple(s) for s in states.reshape(-1,simulation_slice.num_dots).tolist()]
    state_list = set(states)
    polytope_list = {}
    for state in state_list:
        #get the polytope
        polytope=simulation_slice.boundaries(state)
        A=polytope.A
        b=polytope.b
        
        #check if polytope is empty and continue otherwise (should never trigger)
        if A.shape[0] == 0:
            continue
        lower_bounds_graph = np.hstack([-np.eye(2), (minV-0.05*abs(minV))[:,None]])
        upper_bounds_graph = np.hstack([np.eye(2), -(maxV+0.05*abs(maxV))[:,None]])
        lower_bounds = np.vstack([lower_bounds_graph, upper_bounds_graph])
        corners = get_2D_polytope(A,b,"white",lower_bounds)
        polytope_list[str(state)] = np.array(corners+V_offset)
    return polytope_list

def plot_polytopes(ax, polytopes, fontsize = 10, color = "w", 
                   axes_rescale = 1, only_labels = False, only_edges = False,
                   skip_dots = []
                   ):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for polytope in polytopes:
        corners = polytopes[polytope]
        hull = ConvexHull(polytopes[polytope])
        if not only_labels:
            for i,simplex in enumerate(hull.simplices):
                ax.plot(corners[simplex, 0]* axes_rescale , corners[simplex, 1]* axes_rescale , "-", c="w", lw=1)
        if not only_edges:
            box_mid  = (np.array(np.min(corners, axis=0)) + np.array(np.max(corners, axis=0)))/2*axes_rescale
            # skip skip_dots entries to polytope
            res = np.array([int(ele) for ele in str(polytope[1:-1]).split(",")])
            inds = np.array(list(set(np.arange(len(res))) - set(skip_dots)), dtype=int)

            
            if( box_mid[0] > xlim[0] and  box_mid[0] < xlim[1] and box_mid[1] > ylim[0] and box_mid[1] < ylim[1] ):
                ax.text(box_mid[0], box_mid[1], str(list(res[inds])), c=color,ha='center', va='center', fontsize=fontsize)


'''
def get_polytopes(states, simulation_slice, minV, maxV):
    #iterate over the list of different states and plot their sliced polytope
    draw_labels = True
    states = [tuple(s) for s in states.reshape(-1,simulation_slice.num_dots).tolist()]
    state_list = set(states)
    polytope_list = []
    for state in state_list:
        polytope_list.append(simulation_slice.boundaries(state))
        print(polytope_list[-1])
        #get the polytope
        polytope=simulation_slice.boundaries(state)
        A=polytope.A
        b=polytope.b
        
        #check if polytope is empty and continue otherwise (should never trigger)
        if A.shape[0] == 0:
            continue
        lower_bounds_graph = np.hstack([-np.eye(2), (minV-0.05*abs(minV))[:,None]])
        upper_bounds_graph = np.hstack([np.eye(2), -(maxV+0.05*abs(maxV))[:,None]])
        lower_bounds = np.vstack([lower_bounds_graph, upper_bounds_graph])
        point_inside,box = plot_2D_polytope(ax,A,b,"white",lower_bounds)
        print(box)
        if draw_labels and not box is None:
            box_mid = (box[0]+box[1])/2
            # Add charge state text 
            if( box_mid[0] > minV[0] and  box_mid[0] < maxV[0] and box_mid[1] > minV[1] and box_mid[1] < maxV[1] ):
                ax.text(box_mid[0], box_mid[1], str(state), c="white",ha='center', va='center')
'''
    