import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from qdarts.util_functions import is_sequence
from tqdm import tqdm


def find_feasible_point(halfspaces):
    """Computes a feasible point by a polytope defined in halfspace format. internal."""
    norm_vector = np.reshape(
        np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1)
    )
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    return res.x[:-1]


def plot_2D_polytope(
    ax, A, b, color, lower_bounds, label=None, linestyle="-", linewidth=1
):
    """Plots a single 2D polytope. internal."""
    eqs = np.hstack([A, b.reshape(-1, 1)])
    # add constraints to polytope
    eqs = np.vstack([eqs, lower_bounds])

    # Get the corners

    feasible_point = find_feasible_point(eqs)
    if not np.all(eqs[:, :-1] @ feasible_point + eqs[:, -1] < 1.0e-5):
        return None, None
    corners = HalfspaceIntersection(eqs, feasible_point).intersections

    # Easiest way to get proper line-segments of the boundaries in 2D is via the convex hull
    # Form a convex hull based on the corners
    hull = ConvexHull(corners)
    # Plot lines between the points forming the hull (they are assumed to come in counterclockwise order)
    for i, simplex in enumerate(hull.simplices):
        label_trace = label
        if i != 0:
            label_trace = None
        ax.plot(
            corners[simplex, 0],
            corners[simplex, 1],
            linestyle,
            c=color,
            lw=linewidth,
            label=label_trace,
        )
    return feasible_point, (np.min(corners, axis=0), np.max(corners, axis=0))


def get_2D_polytope(A, b, color, lower_bounds, label=None, linestyle="-", linewidth=1):
    """computes the corners of a 2D polytope from the provided polytope. internal"""
    eqs = np.hstack([A, b.reshape(-1, 1)])
    # add constraints to polytope
    eqs = np.vstack([eqs, lower_bounds])

    # Get the corners

    feasible_point = find_feasible_point(eqs)
    if not np.all(eqs[:, :-1] @ feasible_point + eqs[:, -1] < 1.0e-5):
        return None, None
    corners = HalfspaceIntersection(eqs, feasible_point).intersections

    # Easiest way to get proper line-segments of the boundaries in 2D is via the convex hull
    # Form a convex hull based on the corners
    return corners


def raster_CSD_states(
    simulation, P, v_0, minV, maxV, resolution, state_hint_lower_right
):
    """Creates a grid of points in 2D space and computes for each point the ground state. Internal."""
    if not is_sequence(resolution):
        resolution = [resolution, resolution]
    states = np.zeros((resolution[0], resolution[1], simulation.num_dots), dtype=int)
    line_start = simulation.find_state_of_voltage(
        v_0 + P @ np.array([minV[0], minV[1]]), state_hint_lower_right
    )
    pbar = tqdm(total=np.prod(resolution), desc="Rastering CSD")
    # TODO: this is terrible in terms of performance
    for i, v1 in enumerate(np.linspace(minV[0], maxV[0], resolution[0])):
        state = line_start
        for j, v2 in enumerate(np.linspace(minV[1], maxV[1], resolution[1])):
            v = v_0 + P @ np.array([v1, v2])
            state = simulation.find_state_of_voltage(v, state)
            if not simulation.inside_state(v, state):
                print("error")
            if j == 0:
                line_start = state
            states[i, j] = state
            pbar.update(1)
    return states


def get_CSD_data(
    simulation, P, v_0, lower_left, upper_right, resolution, state_hint_lower_left,proxy=False
):
    """
    Function that computes a Charge Stability Diagram from a simulation of a device.
    The function plots the states at voltages v=v_0+P*x where x is a vector with elements
    in a box defined by its lower-left and upper-right corners given by lower_left and upper_right and
    number of values given by the resolution. if upper_right=-lower_left then v_0 is the center
    pixel of the rastered plot.

    By default, a background image is plotted
    which is based on the charge configuration at a position. This is overlaid with a line plot indicating the
    exact transition points between states. Optionally each region is labelled using the exact electron state.

    Parameters
    ----------
        simulation:
            the device simulation to raster
        ax:
            matplotlib axis object to plot into
        v_0:
            the origin of the coordinate system to plot.
        P:
            coordinate system. A nx2 axis where n is the number of gates in simulation.
        lower_left:
            minimum value in x for both axes
        upper_right:
            maximum value of x in both axes
        resolution:
            number of sampled points in each direction of x. either single number or one per axis.
        state_hint_lower_left:
            starting point to guess the initial state of the lower left corner. must not be empty within sim.
        proxy:
            Whether to cache the computations of the full polytopes. False is faster in most cases.
    """
    minV = np.array(lower_left)
    maxV = np.array(upper_right)

    # find the true corner state before slicing the simulation. otherwise we might have trouble
    # finding it if the state hint does not touch the projection

    lower_left_corner = v_0 + P @ minV
    corner_state = simulation.find_state_of_voltage(
        lower_left_corner, state_hint_lower_left
    )
    simulation_slice = simulation.slice(P, v_0, proxy=proxy)

    # compute CSD
    states = raster_CSD_states(
        simulation_slice, np.eye(2), np.zeros(2), minV, maxV, resolution, corner_state
    )
    color_weights = np.linspace(1, 2.7, simulation_slice.num_dots)
    CSD_data = 1 + np.sum(color_weights.reshape(1, 1, -1) * states, axis=2)
    return simulation_slice, CSD_data, states


def get_polytopes(states, simulation_slice, minV, maxV):
    """For each unique state in the provided state list, computes the corners of the polytope. of the 2D sliced simulation.

    This function is used for plotting of the exact state lines of the underlying capacitive model.
    """
    # iterate over the list of different states and plot their sliced polytope

    states = [tuple(s) for s in states.reshape(-1, simulation_slice.num_dots).tolist()]
    state_list = set(states)
    polytope_list = {}
    for state in state_list:
        # get the polytope
        polytope = simulation_slice.boundaries(state)
        A = polytope.A
        b = polytope.b

        # check if polytope is empty and continue otherwise (should never trigger)
        if A.shape[0] == 0:
            continue
        lower_bounds_graph = np.hstack([-np.eye(2), (minV - 0.05 * abs(minV))[:, None]])
        upper_bounds_graph = np.hstack([np.eye(2), -(maxV + 0.05 * abs(maxV))[:, None]])
        lower_bounds = np.vstack([lower_bounds_graph, upper_bounds_graph])
        corners = get_2D_polytope(A, b, "white", lower_bounds)
        polytope_list[str(state)] = np.array(corners)
    return polytope_list


def plot_polytopes(
    ax,
    polytopes,
    fontsize=10,
    color="w",
    axes_rescale=1,
    only_labels=False,
    only_edges=False,
    skip_dots=[],
    alpha=1,
    lw=1,
):
    """Plot the polytopes computes by get_polytopes"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for polytope in polytopes:
        corners = polytopes[polytope]
        hull = ConvexHull(polytopes[polytope])
        if not only_labels:
            for i, simplex in enumerate(hull.simplices):
                ax.plot(
                    corners[simplex, 0] * axes_rescale,
                    corners[simplex, 1] * axes_rescale,
                    "-",
                    c=color,
                    lw=lw,
                    alpha=alpha,
                )
        if not only_edges:
            box_mid = (
                (np.array(np.min(corners, axis=0)) + np.array(np.max(corners, axis=0)))
                / 2
                * axes_rescale
            )
            # skip skip_dots entries to polytope
            res = np.array([int(ele) for ele in str(polytope[1:-1]).split(",")])
            inds = np.array(list(set(np.arange(len(res))) - set(skip_dots)), dtype=int)

            if (
                box_mid[0] > xlim[0]
                and box_mid[0] < xlim[1]
                and box_mid[1] > ylim[0]
                and box_mid[1] < ylim[1]
            ):
                ax.text(
                    box_mid[0],
                    box_mid[1],
                    str(res[inds].tolist()),
                    c=color,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
