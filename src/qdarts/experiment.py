from qdarts.simulator import CapacitiveDeviceSimulator
from qdarts.tunneling_simulator import (
    TunnelBarrierModel,
    ApproximateTunnelingSimulator,
    NoisySensorDot,
)
from qdarts.noise_processes import OU_process, Cosine_Mean_Function
from qdarts.capacitance_model import CapacitanceModel
from qdarts.util_functions import (
    find_point_on_transitions,
    fix_gates,
    find_label,
    compensate_simulator_sensors,
    axis_align_transitions,
)

# for the algorithm
import numpy as np

# for plotting
from matplotlib import pyplot as plt
from qdarts.plotting import get_CSD_data, get_polytopes

# for measuring runtime

plt.rcParams.update({"font.size": 18})

# SIMULATION CONSTANTS
SLOW_NOISE = {"tc": 50}


class Experiment:  # TODO: change name to the simulator name
    """Experiment class that helps setup the different parts of the simulator into an easy to ue interface.
    The usage of this class is best described by the tutorials in the example folder.
    """

    def __init__(
        self,
        capacitance_config,
        tunneling_config=None,
        sensor_config=None,
        print_logs=True,
    ):
        """
        capacitance_config: dictionary containing the capacitance model parameters
        tunneling_config: dictionary containing the tunneling model parameters
        sensor_config: dictionary containing the sensor model parameters
        print_logs: bool, whether to print logs
        """
        # Store configurations
        self.capacitance_config = capacitance_config
        self.tunneling_config = tunneling_config
        self.sensor_config = sensor_config
        self.print_logs = print_logs
        
        if print_logs:
            print("EXPERIMENT INITIALIZED")
            print("-----------------------")

        # Deploy simulators
        self.capacitance_sim = self.deploy_capacitance_sim(capacitance_config)
        self.has_sensors = False

        # Check requirements for sensor and tunneling configurations
        if tunneling_config is not None and sensor_config is None:
            raise ValueError(
                "Specifying a tunneling configuration also requires a sensor configuration."
            )
        if tunneling_config is None and sensor_config is not None:
            raise ValueError(
                "Specifying a sensor configuration also requires a tunneling configuration."
            )
        if tunneling_config is not None:
            self.sensor_model = self.deploy_sensor_model(sensor_config)
            self.tunneling_sim = self.deploy_tunneling_sim(
                self.capacitance_sim, tunneling_config
            )
            self.has_sensors = True

    # DEPLOYMENT FUNCTIONS
    # --------------------------
    def deploy_capacitance_sim(self, config):
        """
        Function that deploys a capacitance simulator.

        Arguments
        ---------
        config: dictionary containing the capacitance model parameters

        Returns
        -------
        sim: CapacitiveDeviceSimulator object
        """
        capacitance_model = CapacitanceModel(
            config["C_Dg"], config["C_DD"], -1, ks=config["ks"]
        )
        sim = CapacitiveDeviceSimulator(capacitance_model)

        # Save the parameters
        self.N = len(config["C_Dg"])  # number of dots
        self.inner_dots = list(
            np.arange(self.N)
        )  # indices of the dots. NOTE: no sensor at this point

        if self.print_logs:
            if config["ks"] is not None:
                # Print log of capacitance parameters
                log = """
                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                {}
                Dot-gate capacitances: 
                {}
                Size of Coulomb peaks V[n] scale as {}/(n + {})
                """.format(
                    config["C_DD"],
                    config["C_Dg"],
                    np.round(
                        (1 - 0.137 * 3.6 / (config["ks"] + 2.6) * config["ks"]), 3
                    ),
                    2 + config["ks"]
                )
                print(log)
            else:
                # Print log of capacitance parameters
                log = """
                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                {}
                Dot-gate capacitances: 
                {}
                Size of Coulomb peaks V[n] is constant
                """.format(config["C_DD"], config["C_Dg"])
                print(log)

        return sim

    def deploy_tunneling_sim(self, capacitance_sim, tunneling_config):
        """
        Function that deploys a tunneling simulator.

        Arguments
        ---------
        capacitance_sim: CapacitiveDeviceSimulator object
        tunneling_config: dictionary containing the tunneling model parameters

        Returns
        -------
        tunneling_sim: ApproximateTunnelingSimulator object
        """
        tunneling_matrix = tunneling_config["tunnel_couplings"]
        tunnel_barrier_offsets = np.log(tunneling_matrix + 1e-20)
        barrier_levers = (
            tunneling_config["barrier_levers"]
            if "barrier_levers" in tunneling_config.keys()
            else None
        )
        barrier_sim = TunnelBarrierModel(tunnel_barrier_offsets, barrier_levers)
        tunneling_sim = ApproximateTunnelingSimulator(
            capacitance_sim,
            barrier_sim,
            tunneling_config["temperature"],
            self.sensor_model,
        )

        # explore the neighboring polytopes
        slack = tunneling_config["energy_range_factor"] * np.maximum(
            1.0 / tunneling_sim.beta, np.max(tunneling_matrix)
        )
        tunneling_sim.poly_sim.set_maximum_polytope_slack(slack)
        tunneling_sim.num_additional_neighbours[self.sensor_model.sensor_dot_ids] = 2

        if self.print_logs:
            # Print log of tunneling parameters
            log = """
            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            {}
            Temperature: {} K
            Energy range factor: {}
            """.format(
                tunneling_matrix,
                tunneling_config["temperature"],
                tunneling_config["energy_range_factor"],
            )
            print(log)

        return tunneling_sim

    def deploy_sensor_model(self, sensor_config):
        """
        Function that deploys a sensor model.

        Arguments
        ---------
        config: dictionary containing the sensor model parameters

        Returns
        -------
        sensor_sim: NoisySensorDot object
        """
        if "noise_amplitude" not in sensor_config.keys():
            sensor_config["noise_amplitude"] = {}
        if "fast_noise" not in sensor_config["noise_amplitude"]:
            sensor_config["noise_amplitude"]["fast_noise"] = 0.0
        if "slow_noise" not in sensor_config["noise_amplitude"]:
            sensor_config["noise_amplitude"]["slow_noise"] = 0.0
        if "signal_noise" not in sensor_config["noise_amplitude"]:
            sensor_config["noise_amplitude"]["signal_noise"] = 0.0
        # Separate between inner and sensor dots
        self.inner_dots = list(
            set(self.inner_dots) - set(sensor_config["sensor_dot_indices"])
        )

        # Define slow-noise generator
        slow_noise_gen = OU_process(  # TODO: LATER: implement 1/f noise
            sigma=sensor_config["noise_amplitude"]["slow_noise"],
            tc=SLOW_NOISE["tc"],
            dt=1,  # the unit of time is the single measurment
            num_elements=len(sensor_config["sensor_dot_indices"]),
        )

        if "mean_field" in sensor_config.keys():
            mean_field_config = sensor_config["mean_field"]
            if mean_field_config["type"] == "Cosine_Mean_Function":
                b = mean_field_config["b"] if "b" in mean_field_config.keys() else None
                slow_noise_gen = Cosine_Mean_Function(
                    slow_noise_gen, mean_field_config["a"], mean_field_config["W"], b
                )

        # Deploy sensor model
        sensor_sim = NoisySensorDot(sensor_config["sensor_dot_indices"])

        # Configure sensor model
        sensor_sim.config_noise(
            sigma=sensor_config["noise_amplitude"]["fast_noise"],
            signal_noise_scale=sensor_config["noise_amplitude"]["signal_noise"],
            slow_noise_gen=slow_noise_gen,
        )
        sensor_sim.config_peak(
            g_max=1.0, peak_width_multiplier=sensor_config["peak_width_multiplier"]
        )

        # Pring log of sensor parameters
        if self.print_logs:
            log = """
            Sensor model deployed with the following parameters:   
            Sensor dot indices: {}
            Sensor detunings: {} meV
            Coulomb peak width: {} meV
            Slow noise amplitude: {} ueV
            Fast noise amplitude: {} ueV
            Signal noise scale: {}
            """.format(
                sensor_config["sensor_dot_indices"],
                np.array(sensor_config["sensor_detunings"]) * 1e3,
                np.round(
                    2
                    * sensor_config["peak_width_multiplier"]
                    * self.tunneling_config["temperature"]
                    * 86
                    * 1e-3
                    / 0.631,
                    2,
                ),
                np.round(sensor_config["noise_amplitude"]["slow_noise"] * 1e6, 4),
                np.round(sensor_config["noise_amplitude"]["fast_noise"] * 1e6, 4),
                np.round(sensor_config["noise_amplitude"]["signal_noise"], 5),
            )
            print(log)
        return sensor_sim

    def center_transition(
        self,
        simulator,
        target_state,
        target_transition,
        plane_axes,
        use_virtual_gates=False,
        compensate_sensors=False,
    ):
        """
        Function that center the CSD at a given facet (transition) of the polytope (occupation state).

        Arguments
        ---------
        simulator: any simulator object
        target_state: int, the state at which the transition happens, e.g. [2,2]
        target_transition: list of integers, the transition point e.g. [1,-1] would be the transition from [2,2] to [1,1]
        plane_axes: 2xN array, the axes of the transition which span the plane
        use_virtual_gates: bool, whether to use virtual gates
        compensate_sensors: bool, whether to compensate the sensors

        Returns
        -------
        plane_axes: 2xN array, the axes spanning the cut through voltage plane
        transition_sim: CapacitanceSimulator object, the transition simulator
        """
        if compensate_sensors:
            # fix the sensor gate if we compensate
            simulator = fix_gates(
                simulator,
                self.sensor_config["sensor_dot_indices"],
                np.zeros(len(self.sensor_config["sensor_dot_indices"])),
            )
            # reduce dimension of the voltage space
            complement_gates = [
                i
                for i in range(plane_axes.shape[1])
                if i not in self.sensor_config["sensor_dot_indices"]
            ]
            plane_axes = plane_axes[:, complement_gates]
        # find boundaries of the selected polytope
        poly = simulator.boundaries(target_state)
        # get_labels of the transitions
        inner_labels = poly.labels[:, self.inner_dots]

        # Find the index of the transition point
        transition = np.array(target_transition)[self.inner_dots].T
        try:
            idx_multidot_transition = [find_label(inner_labels, transition)[0]]
        except ValueError:
            raise "The transition point is not in the polytope. Please choose a different transition"

        # Find the offset of V
        v_transition = find_point_on_transitions(poly, idx_multidot_transition)
        print("v_offset found:", v_transition)

        if use_virtual_gates:
            # Compute the normals
            pair_transitions = np.array(
                [np.array(transition).T for transition in plane_axes], dtype=int
            )
            idxs = [
                find_label(inner_labels, t[self.inner_dots])[0]
                for t in pair_transitions
            ]
            normals = -poly.A[idxs]
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            P_transition = normals.T
            transition_sim = axis_align_transitions(
                simulator.slice(P_transition, v_transition),
                target_state,
                poly.labels[idxs],
                [0, 1],
            )
        else:
            transition_sim = simulator.slice(plane_axes.T, v_transition)

        return np.eye(2), transition_sim

    # GETTERS
    # --------------------------

    def get_virtualised_sim(self, simulator, target_state):
        """
        Function that takes a simulator and virtualises the gates specified by inner_dots.

        Arguments
        ---------
        simulator: AbstractPolytopeSimulator object
        target_state: int, the initial corner state guess

        Returns
        -------
        simulator: AbstractPolytopeSimulator object, the virtualised simulator
        """
        gate_transitions = np.eye(self.N, dtype=int)[self.inner_dots]
        # TODO: Default target state is lower left corner state (initial guess). In future user could specify!
        self.target_state = target_state
        simulator = axis_align_transitions(
            simulator, self.target_state, gate_transitions, self.inner_dots
        )
        return simulator

    def get_compensated_sim(self, simulator, target_state):
        """
        Function that takes a capacitance simulator and compensates the sensors.

        Arguments
        ---------
        simulator: AbstractPolytopeSimulator object
        target_stater: int, the state at which sensor compensation happens

        Returns
        -------
        simulator: AbstractPolytopeSimulator object, the compensated simulator
        """
        if not self.has_sensors:
            raise ValueError("Compensating sensors requires a sensor model.")

        sim,v = compensate_simulator_sensors(
            simulator,
            target_state=target_state,
            compensation_gates=self.sensor_config["sensor_dot_indices"],
            sensor_ids=self.sensor_config["sensor_dot_indices"],
            sensor_detunings=self.sensor_config["sensor_detunings"]
        )
        
        #translate sensor gates to 0.
        offset = np.zeros(len(v))
        offset[self.sensor_config["sensor_dot_indices"]] = v[self.sensor_config["sensor_dot_indices"]]
        return sim.slice(np.eye(len(v)),offset)
        

    def get_plot_args(self, x_voltages, y_voltages, plane_axes, v_offset=None):
        """
        Function that returns the arguments for plotting the CSD.

        Arguments
        ---------
        x_voltages: list of floats, the x-axis voltages
        y_voltages: list of floats, the y-axis voltages
        plane_axes: 2xN array, the axes of the plane in which the CSD is to be rendered
        v_offset: Nx1 array, the offset voltage of all of the gates, which defines the origin of the plot

        Returns
        -------
        v_offset: Nx1 array, the offset voltage of all of the gates
        minV: 2x1 array, the minimum voltage of selected axes
        maxV: 2x1 array, the maximum voltage of selected axes
        resolution: list of integers, the resolution of the plot
        """

        if v_offset is None:
            v_offset = np.zeros(plane_axes.shape[1], dtype=float)
        else:
            v_offset = np.array(v_offset, dtype=float)

        xout = x_voltages + np.dot(v_offset, plane_axes[0])
        yout = y_voltages + np.dot(v_offset, plane_axes[1])

        minV = np.array([x_voltages[0], y_voltages[0]])
        maxV = np.array([x_voltages[-1], y_voltages[-1]])

        resolution = [len(x_voltages), len(y_voltages)]

        return v_offset, minV, maxV, resolution, xout, yout

    # RENDER FUNCTIONS
    # --------------------------

    def generate_CSD(
        self,
        x_voltages,
        y_voltages,
        plane_axes,
        target_state=None,
        target_transition=None,
        use_virtual_gates=False,
        compensate_sensors=False,
        compute_polytopes=False,
        use_sensor_signal=False,
        v_offset=None,
        insitu_axis=None,
    ):
        """
        Function that renders the capacitance CSD for a given set of voltages and axes.

        Arguments
        ---------
        x_voltages: list of floats, the x-axis voltages
        y_voltages: list of floats, the y-axis voltages
        plane_axes: 2xN array, the axes of the plane in which the CSD is to be rendered
        target_state: int, the guess state or the state at which the transition happens
        target_transition: list of integers, the transition point e.g. [1,-1] would be the transition from [2,2] to [1,1]
        use_virtual_gates: bool, whether to use virtual gates
        compensate_sensors: bool, whether to compensate the sensors
        compute_polytopes: bool, whether to compute the polytopes
        use_sensor_signal: bool, whether to use the sensor signal
        v_offset: Nx1 array, the offset voltage of all of the gates, which defines the origin of the plot

        Returns
        -------
        xout, yout: list of floats, the x and y voltages
        CSD_data: 2D array, the CSD data
        polytopes: dictionary, the polytopes of the CSD. None if compute_polytopes is False
        sensor_values: 3D array, the sensor signal [size(xout),size(yout),num_sensors]. None if use_sensor_signal is False
        v_offset: Nx1 array, the offset voltage of all of the gates
        """
        # check required parameters
        if target_state is None:
            # if not target state use [0,0,0,0,0,0]
            target_state = [0] * self.N

        plane_axes = np.array(plane_axes)
        # prepare plot
        v_offset, minV, maxV, resolution, xout, yout = self.get_plot_args(
            x_voltages, y_voltages, plane_axes, v_offset
        )

        # pick the simulator
        if use_sensor_signal:
            simulator = self.tunneling_sim
        else:
            simulator = self.capacitance_sim
        if compensate_sensors:
            simulator = self.get_compensated_sim(simulator, target_state=target_state)
        if target_transition is not None:
            plane_axes, simulator = self.center_transition(
                simulator,
                target_state,
                target_transition,
                plane_axes,
                use_virtual_gates,
                compensate_sensors,
            )
            v_offset = np.zeros(2)  # TODO: how to do it nicer?

        elif use_virtual_gates:
            simulator = self.get_virtualised_sim(simulator, target_state)

        if use_sensor_signal:
            csimulator = simulator.poly_sim
        else:
            csimulator = simulator

        # Part for the electrostatic CSD:
        CSD_data = None
        polytopes = None
        if not use_sensor_signal or compute_polytopes:
            backend, CSD_data, states = get_CSD_data(
                csimulator,
                np.array(plane_axes).T,
                v_offset,
                minV,
                maxV,
                resolution,
                target_state,
            )
        if compute_polytopes:
            polytopes = get_polytopes(states, backend, minV, maxV)

        # Part for the sensor signal:
        sensor_values = None
        if use_sensor_signal:
            # TODO: How does this work ? The long chain of condition makes it hard
            # to understand what's going on ... Arguments like plane_axes share the
            # same name but end up having a completely different shape depending on the flags
            sensor_values = simulator.sensor_scan_2D(
                plane_axes,
                v_offset,
                minV,
                maxV,
                resolution,
                target_state,
                insitu_axis=insitu_axis,
            )

        return xout, yout, CSD_data, polytopes, sensor_values, v_offset
