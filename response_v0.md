# Referee 1
*The work by Krzywda et al. presents a python package to simulate charge stability diagrams of capacitively- and tunnel-coupled quantum dot systems. The user can specify the parameters of the device (capacitance network parameters, noise characteristics of the sensing dot, etc.), and the code outputs the charge stability diagram.*

I know that these types of simulations are routinely done in many experimental and theoretical research groups worldwide (as well as the few companies building quantum hardware based on such devices) using in-house-built software solutions. Therefore I see the present package as a welcome attempt to provide a general tool for this community, which is likely to make research and engineering in the field more efficient.*

*The package has a number of functionalities that go beyond the minimal model of electrostatics with the constant-interaction model: e.g., gate virtualisation, charge-dependent capacitances, noise on the sensor dots, etc. Admittedly, there are also important functionalities that are often needed but missing from the present implementation: e.g., spin degree of freedom, orbital level spacing, role of the barrier gates, etc.*

*The authors benchmark their package against a selection of state-of-the-art experiments, which I find mostly satisfactory.*


### Issues to be adressed
1. *There is one point, however, where I am uncertain if the model used here is appropriate. This has to do with the simulation of the finite tunnel coupling effect, shown, e.g., in Fig. 8. My understanding is that in reality (more precisely, on the level of the Hubbard model), there is an even-odd effect for the tunnel-coupling-induced features of the charge stability diagram. For example, the tunnel coupling between the (1,0) and (0,1) charge configurations is weaker than the tunnel coupling between the (1,1) and (0,2) charge configurations by a factor of sqrt(2). In turn, this difference should be visible in any charge stability diagram that includes both transitions. However, the tunneling model specified by the authors, see Eq. (1), does not include this effect. My feeling is that this effect could be easily incorporated in the model, the only price to be payed is a slightly more elaborate formula instead of Eq. (1). I request the authors to implement this change, or if they decide not to do so, then at least comment on this issue in the revised version of the manuscript.
 (I do understand that this even-odd effect is related to the spin degree of freedom, which is admittedly missing from the current version of the package, but I also feel that this effect is qualitatively different from other spin effects, and also straightforward to implement in the existing framework. )*

**Response** 
We appreciate the suggestion. We have implemented this observation into the simulator and included it also in the manuscript. To be preise, when the number of electrons on the pair of dots affected by a transition is even, we increase the tunnel coupling by sqrt(2).

> In reality, the value of effective tunnel coupling depends on the parity of the number of electrons in the dots, with an enhancement by a factor of √2 expected for an even number of electrons. However, this effect originates from the spin degree of freedom and is a direct consequence of the symmetry of the lowest energy states—the singlets of double and single occupation. In the current version of the code, we do not include the spin degree of freedom; therefore, the even-odd effect is not present. We plan to include the spin degree of freedom in future versions of the code, which will allow us to capture this effect.

TODO: PUT IT IN THE TEXT

1. *In the Conclusion, the authors list QDsim and SimCATS as software tools that are similar to QDarts. From the viewpoint of a potential user who wants to pick one of these potential solutions, I find it desirable to have a more detailed comparison of these tools, describing similarities and differences. Ideally, this could come in the form of a table comparing functionalities and performance, but if that is too much to ask for, then anything going beyond the current description would be welcome.*

**Response**
We agree with the reviewer that transparency in comparison between different packages is highly beneficial for the community and can possibly lead to their convergence into one fully-functional and widely used simulation package.

To allow for comparison, in the reviewed version, we have decided to include a table that compares the functionalities of the three packages and highlights the differences between them.

TODO TABLE
| Feature | QDarts | QDsim | SimCATS |
| --- | --- | --- | --- |
| Constant Interaction Model | Yes | Yes | Yes |
| Charge-dependent capacitances | Yes | ? | ? |
| Noise | Physical model | Additive | ? |
| Finite temperature | Yes | ? | ? |
| Gate virtualization | Yes | Yes | Yes |
| Sensor compensation | Yes | No | No |
| Finite tunnel coupling | Yes | No |? |
| Transition finding | Yes | No | No |
| Role of the barrier gates | No* | ? |? |
| Spin degree of freedom | No | No | No |
| Number of dots | <20 | <100 | ? |


TODO: TABLE (EVERT), IS THERE ANYTHING THEY CAN DO AND WE CANNOT?


(iii) *My final request is to add color scales and physical units to the conductance plots (Fig. 1, Fig. 7, Fig. 8). Perhaps one could trace these back from the example python notebooks; still I feel this is important, to make the paper self-contained and complying to the principles of accuracy and reproducibility.*
**Response** 
Appriciating the feedback, we have implemented the changes in both the manuscript and the example file.

TODO: COLOR SCALES (JAN)

# Referee 2

*Krzydwa et al. present a physical model of the electrostatic of quantum dots which allows to extract the stable charge configuration of an array of quantum dot and simulate the response of the sensor to charge transition.
This work is very timely as we can see with few papers submitted at the same time on ArXiv which try to adress this need in the community with different approaches .
Moreover, the model is refined to account for the noise and the tunnel coupling and is tested against experimental data which are quantitatively reproduced.*

### Issues to be adressed
1. *The model is implemented in a python code which is accessible and properly documented. As experimentalists working on this problem, we found the code easy to use (while a bit difficult to install) and quite efficient in term of speed.
We have bench-marked this code against our own which has a more brute force approach and found the present code to be much more efficient in particular toward a large number of quantum dots and charge number. We would therefore encourage experimentalists to use this code to simulate their charge stability diagram of an open array.*

**Response** 
We appriciate the effort made by the reviewer to test the code and provide feedback. We are glad that the code was found to be efficient and easy to use. At the same time, we are grateful for the suggestion of improving the installation process.  

To improve user experience, we have developed transparent documentation and installation instructions, which is available at https://condensedai.github.io/QDarts/. We have also recorded a video tutorial that will be available on the project's website. 

TODO: VIDEO (JAN)

1. *There are still some functionalities that are missing and which could definitely benefit to the community to simulate more complex array. These are recommendations, and a bit beyond our expertise to estimate the technical feasibility.
First of all, the ability to work with a finite number of charge in the array would be very helpful to simulate isolated arrays which is a widely used approach in experiments (see Flentje, et al. Nat Commun. 8, 501 (2017) or Meyer et al. Nano Lett. 24, 11593 (2023) or Yang, et al. Nature 580, 350–354 (2020).)*

**Response** 

This is a great suggestion, unfortunately it is at least partially difficult to implement. If the goal is to simulate a real array, then often only some dots are decoupled from the reservoir, while sensor dots need to be connected to the reservoir to work. Your suggestion is easy to implement without a sensor dot, in this case, this can be achieved by writing a wrapper around the CapacitanceModel class to limit the neighbour enumeration of a state. (e.g., call CapacitanceModel.enumerate_neighbours and then remove all returned states that do not leave the total number of electrons constant). However, the full implementation with partial reservoir connection requires serious implementation effort and we would like not having to support a partial implementation that is reasonably easy to implement by the user.
However, we believe that restricting transitions is not the way to go. Instead our goal is to implement transition dynamics into the array, based on barrier gates and tunnel coupling strength. While this allows to implement effects like latching, setting the transition speed for certain reservoir transitions to 0 can be used to efficiently implement this effect. We aim to implement this in a follow-up publication.

3. *Second, instead of using charge detection, probing the quantum capacitance through gate-based reflectometry would be a nice functionality. For instance, reproducing RF signal on stability diagrams for different parameters such as tunnel coupling, lever arm, frequency etc… would be extremely useful. This readout method is believed to be scalable approach to control and read spins in large array (Crippa, et al. Nat Commun 10, 2776 (2019), Veldhorst, et al. Nat Commun 8, 1766 (2017).)*

**Response** 
 
 We are grateful for this suggestion. In order to make the code more versatile and useful for a wider range of applications, we implemented the vannila version of in-situ reflectometry and provided its description in the new subsection of the manuscript. We also included a new example file that demonstrates the use of this feature.

 TODO: IN-SITU REFLECTOMETRY (JAN)




