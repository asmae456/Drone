# optimal_quad_control_SL
End-to-end optimal quadcopter control through Supervised Learning

**Notebooks**
1) **Data Generation:** used to generate the datasets of optimal trajectories
3) **Network Training:** uses the generated datasets to train a G&CNet to learn the optimal state feedback
5) **Simulation:** simulates the quadcopter controlled by a trained G&CNet
7) **Generate C code:** used to convert the G&CNet from pytorch to c code
9) **Minimum Snap trajectories:** used to compute the minimum snap polynomial trajectories and convert it into C code

Note that in order to run the Dataset Generation notebook, AMPL (A Mathematical Programming Language) needs to be installed as well as the NLP solver SNOPT.
# Drone
