#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2016 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.
#

# This is the main path of your do-mpc installation relative to the execution folder
path_do_mpc = '../../'
# Add do-mpc path to the current directory
import sys
sys.path.insert(0,path_do_mpc+'code')
# Do not write bytecode to maintain clean directories
sys.dont_write_bytecode = True
# Import keras to load trained models
import keras
from keras.models import model_from_json
# Start CasADi
from casadi import *
# Import do-mpc core functionalities
import core_do_mpc
# Import do-mpc plotting and data managament functions
import data_do_mpc

import numpy as NP

import pdb

"""
-----------------------------------------------
do-mpc: Definition of the do-mpc configuration
-----------------------------------------------
"""

# Import the user defined modules
import template_model
import template_optimizer
import template_EKF
import template_simulator

# Create the objects for each module
model_1 = template_model.model()
# Create an optimizer object based on the template and a model
optimizer_1 = template_optimizer.optimizer(model_1)
# Create an observer object based on the template and a model
observer_1 = template_EKF.observer(model_1)
# Create a simulator object based on the template and a model
simulator_1 = template_simulator.simulator(model_1)
# Create a configuration
configuration_1 = core_do_mpc.configuration(model_1, optimizer_1, observer_1, simulator_1)

# Set up the solvers
configuration_1.observer.observed_states = configuration_1.model.ocp.x0
configuration_1.setup_solver()
configuration_1.make_step_optimizer()

"""
-----------------------------------------------
Load neural network
-----------------------------------------------
"""

filename = 'controller_1'
json_file = open(filename+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(filename+".h5")
print("Loaded model from disk")

"""
----------------------------
do-mpc: MPC loop
----------------------------
"""
while (configuration_1.simulator.t0_sim + configuration_1.simulator.t_step_simulator < configuration_1.optimizer.t_end):
    t0_sim = configuration_1.simulator.t0_sim
    v_0_real = 10
    v_real = v_0_real + 5*sin(2*pi*0.1*t0_sim)
    configuration_1.simulator.p_real_batch = NP.array([v_real])
    
    """
    ----------------------------
    do-mpc: Optimizer
    ----------------------------
    """
    # Make one optimizer step (solve the NLP)
    # configuration_1.make_step_optimizer()
    u_lb = NP.array([-10.0])
    u_ub = NP.array([10.0])
    x_lb = NP.array([0.2, -1.1,-3.0])
    x_ub = NP.array([0.8, 1.1, 3.0])
    x_in_scaled = NP.atleast_2d((configuration_1.observer.observed_states - x_lb) / (x_ub - x_lb))
    u_opt_scaled = NP.squeeze(loaded_model.predict(x_in_scaled))
    u_opt = u_opt_scaled * (u_ub - u_lb) + u_lb
    u_opt_lim = NP.maximum(NP.minimum(u_opt,u_ub),u_lb)
    configuration_1.optimizer.u_mpc = u_opt

    """
    ----------------------------
    do-mpc: Simulator
    ----------------------------
    """
    # Simulate the system one step using the solution obtained in the optimization
    # configuration_1.make_step_simulator() # simulation step in observer included

    """
    ----------------------------
    do-mpc: Observer
    ----------------------------
    """
    # Make one observer step
    configuration_1.make_step_observer()

    """
    ------------------------------------------------------
    do-mpc: Prepare next iteration and store information
    ------------------------------------------------------
    """
    # Store the information
    configuration_1.store_mpc_data()

    # Set initial condition constraint for the next iteration
    configuration_1.prepare_next_iter()

    """
    ------------------------------------------------------
    do-mpc: Plot MPC animation if chosen by the user
    ------------------------------------------------------
    """
    # Plot animation if chosen in by the user
    data_do_mpc.plot_animation(configuration_1)

"""
------------------------------------------------------
do-mpc: Plot the closed-loop results
------------------------------------------------------
"""

data_do_mpc.plot_mpc(configuration_1)

# Export to matlab if wanted
data_do_mpc.export_to_matlab(configuration_1)


input("Press Enter to exit do-mpc...")
