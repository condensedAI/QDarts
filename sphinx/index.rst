API
===
.. currentmodule:: qdarts

In the following, we will describe the core components of our API

Experiment Class
----------------
For a quick start for uing the simulator, there is the Experiment class that is also used
in the example notebook. We refer to the notebook for a detailled usage example.

.. autosummary::
    :toctree: interfaces
    :recursive:
    
    experiment.Experiment
    
    
Simulation Components
---------------------
The main components to build your own simulations are shown below.
We first need to create a Capacitance model that generates a 
capacitive energy function :math:`E(v,n)`. With this, we can start
to generate a purely capacitive simulation.

.. autosummary::
    :toctree: interfaces
    :recursive:
    
    capacitance_model.CapacitanceModel
    simulator.CapacitiveDeviceSimulator

This simulation is rather basic and can only compute areas of voltages that lead to the same ground
state, so called coulomb diamonds. This can already be used to generate
Charge Stability diagrams by computing 2D slices through the set of polytopes,
but those will not look very realistic, nor will they include tunneling.

To move beyond this simulation, we need to drastically improve the modeling.
This next step is based on the full tunneling simulator that requires a simulation
of the sensor dot and a simulation of tunnel barriers - the latter can be simplified by
just providing a matrix of constant tunnel couplings.

.. autosummary::
    :toctree: interfaces
    :recursive:
    
    tunneling_simulator.ApproximateTunnelingSimulator
    tunneling_simulator.TunnelBarrierModel
    tunneling_simulator.NoisySensorDot


The simulation of the sensor dot can optionally make use of a noise model in order
to generate dependent noise. For this, we currently offer the following two classes

.. autosummary::
    :toctree: interfaces
    :recursive:
    
    noise_processes.OU_process
    noise_processes.Cosine_Mean_Function



Data Classes
-------------
Finally, both levels of simulations have their own data classes. The polytope class is returned by the
boundaries method by any simulator and the local system is returned by the tunneling simulator. Both describe
the local state of the simulator in some region.

.. autosummary::
    :toctree: interfaces
    :recursive:
    
    tunneling_simulator.LocalSystem
    polytope.Polytope

Interfaces
----------
QDarts offers a set of interfaces and base classes that can serve as a primer to how to extend the simulator by 
replacing existing components. In most cases, only a few specialized functions need to be implemented as the base
class implements most of the important logic. Please note that QDarts is under active development, all of these 
interfaces are subject to change.

.. autosummary::
    :toctree: interfaces
    :recursive:
    
    capacitance_model.AbstractCapacitanceModel
    simulator.AbstractPolytopeSimulator
    simulator.AbstractCapacitiveDeviceSimulator
    noise_processes.AbstractNoiseProcess
    tunneling_simulator.AbstractSensorSim   
    
    
