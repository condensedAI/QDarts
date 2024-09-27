# QDarts
Efficient **Q**uantum **D**ot **ar**ray **t**ransition **s**imulator. 

## Description
We provide an efficient simulation package, QDarts, generating realistic charge conductance signals from medium, more than 10 quantum dot arrays. By levering the polytope finding algorithm from [O. Krause, A. Chatterjee, F. Kuemmeth and E. van Nieuwenburg, Learning coulomb diamonds in large quantum dot arrays, SciPost Physics 13(4), 084 (2022)](https://scipost.org/SciPostPhys.13.4.084), the QDarts allows for:
- Transition finding in high-dimensional voltage space, 
- Selection of arbitrary cuts in the voltage space,
- Simulating effects of finite tunnel couplings,
- Including non-constant charging energies,
- Simulation of multiple sensor dot,
- Tunable noise parameters,
- User-friendly interface.

## Installation
The package supports Python 3.6 and later. To install the package, run the following command:

    pip install qdarts

## Manuscript
The package is based on the manuscript by [Krzywda et al., QDarts: A Quantum Dot Array Transition Simulator for finding charge transitions in the presence of finite tunnel couplings, non-constant charging energies and sensor dots](). The manuscript has been submitted to the SciPost Physics Codebases.

## Examples
The package provides a simple example to demonstrate the usage of the package. The example is available in the examples qatpack/examples folder. The example demonstrates the simulation of a quantum dot array with sensor dots, tunnel couplings, and non-constant charging energy. 

As a proof of principle, in the example we reconstruct the figure from the paper [Neyens et al.](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.12.064049z), which shows the measured charge conductance signal from two sensor dots, which detect simultanous four-dot transition in the quantum dot array. The figure, visible below, has been computed in about a minute on a standard laptop.

<p align="center">
  <img src="https://github.com/condensedAI/QDarts/blob/main/examples/figures/neyens.png" />
<p/>
    
## Files in this repository
    qdarts
        |-- qdarts
            |-- model.py
            |-- noise_processes.py
            |-- experiment.py
            |-- plotting.py
            |-- polytope.py 
            |-- simulator.py
            |-- tunneling_simulator.py
            |-- util_functions.py
        |-- examples
            |-- examples_scipost.ipynb # notebook to reproduce figures from paper
        |-- README.md
        |-- LICENCE.md
        |-- CITATION.cff


## Roadmap
The package is under active development. The future plans include:
- [ ]   Adding barrier gates,
- [ ]   Including realistic noise processes, including 1/f noise,
- [ ]   Adding more examples,
- [ ]   Adding a method for generating capacitance matrices from:
    - [ ] QD array layout,
    - [ ] Experimental data,
    - [ ] Finite element method simulations,
- [ ]   Scaling up to larger quantum dot arrays N>10,
