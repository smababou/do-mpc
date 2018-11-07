# Run different batches with different initial conditions to get reasonable training data

# This is the main path of your do-mpc installation relative to the execution folder
path_do_mpc = '../../'
# Add do-mpc path to the current directory
import sys
sys.path.insert(0,path_do_mpc+'code')
sys.path.insert(0,'..')
# Do not write bytecode to maintain clean directories
sys.dont_write_bytecode = True

# Import keras to load trained models
import keras
from keras.models import model_from_json

# Import numpy
import numpy as NP
# Start CasADi
from casadi import *
# Import do-mpc core functionalities
import core_do_mpc
# Import do-mpc plotting and data managament functions
import data_do_mpc
# Import function to make projection
import pdb

# number of batches to generate data from
n_batches = 100
offset = 0
controller_number = 0
neural_network = True # NOTE: if false MPC instead of NN
# initialize the problem (first lines of do_mpc.py)

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

"""
-----------------------------------------------
Load neural network
-----------------------------------------------
"""
if neural_network:
    filename = 'controller_du' + str(controller_number)
    json_file = open(filename+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename+".h5")
    print("Loaded model from disk")

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
# if neural_network:
#     configuration_1.make_step_optimizer()
configuration_1.simulator.p_real_batch = NP.zeros([2])

# Store the inputs from real MPC and NN_mpc in an array
nu = configuration_1.model.u.size(1)
u_mpc_all = NP.resize([], (nu*2 + 1,n_batches + offset))

for i in range(offset, offset + n_batches):

    # The initial states for the batches are:
    theta_0 = 0.29359907+0.05
    phi_0 = 0.52791537
    psi_0 = 0.0

    theta_0 = NP.random.uniform(0.2,1.6)
    phi_0 = NP.random.uniform(-1.4,1.4)
    psi_0 = NP.random.uniform(-3.3,3.3)

    initial_state_batch = NP.array([theta_0, phi_0, psi_0])

    # choose the real parameters
    E_batch = NP.random.uniform(4.0,6.0)
    c_batch = NP.random.uniform(0.005,0.04)

    w_mean = 8.0 + NP.random.uniform() * 4.0
    var_t = 0.05 + NP.random.uniform() * 0.1
    w_lb = NP.array([7.0])
    w_ub = NP.array([13.0])
    w_amp_max = NP.minimum(NP.abs(w_ub-w_mean),NP.abs(w_lb-w_mean))
    w_amp = NP.random.uniform() * w_amp_max
    w_shift = NP.random.uniform() * 2.0 * pi
    w_init = w_mean + w_amp * sin(w_shift)

    configuration_1.simulator.p_real_batch[0] = E_batch
    # configuration_1.simulator.p_real_batch[1] = c_batch
    configuration_1.simulator.p_real_batch[-1] = w_init

    configuration_1.observer.ekf.x_hat[3] = E_batch + NP.random.normal(0,0.05)
    configuration_1.observer.ekf.x_hat[-1] = w_init + NP.random.normal(0,0.1)

    # Update initial condition for this batch
    x_scaling = configuration_1.model.ocp.x_scaling
    X_offset = configuration_1.optimizer.nlp_dict_out['X_offset']
    nx = len(configuration_1.model.ocp.x0)
    configuration_1.optimizer.arg['lbx'][X_offset[0,0]:X_offset[0,0]+nx] = initial_state_batch / x_scaling
    configuration_1.optimizer.arg['ubx'][X_offset[0,0]:X_offset[0,0]+nx] = initial_state_batch / x_scaling
    configuration_1.simulator.x0_sim = DM(initial_state_batch) / x_scaling
    configuration_1.model.ocp.x0 = initial_state_batch
    # Restart iteration counter
    configuration_1.mpc_iteration = 1
    configuration_1.simulator.t0_sim = 0
    configuration_1.simulator.tf_sim = 0 + configuration_1.simulator.t_step_simulator
    configuration_1.mpc_data = data_do_mpc.mpc_data(configuration_1)

    configuration_1.mpc_data.mpc_parameters[0,:] = configuration_1.simulator.p_real_batch
    configuration_1.mpc_data.mpc_parameters_est[0,:] = configuration_1.observer.ekf.x_hat[3:]

    # Make only 1 step of the control to test randomly generated initial conditions
    # while (configuration_1.simulator.t0_sim + configuration_1.simulator.t_step_simulator < configuration_1.optimizer.t_end):

    # Update wind
    t0_sim = configuration_1.simulator.t0_sim
    configuration_1.simulator.p_real_batch[-1] = w_mean + w_amp * sin(2*pi*var_t*t0_sim+w_shift)

    # Solve the real MPC problem
    configuration_1.make_step_optimizer()
    # Solve the problem with the neural network
    u_lb = NP.array([-10.0])
    u_ub = NP.array([10.0])
    x_lb = NP.array([0.2, -1.4,-3.3])
    x_ub = NP.array([1.6, 1.4, 3.3])
    x_in_scaled = NP.atleast_2d(((configuration_1.observer.observed_states) - x_lb) / (x_ub - x_lb))
    x_in_scaled = NP.atleast_2d(((initial_state_batch) - x_lb) / (x_ub - x_lb))
    u_opt_scaled = NP.squeeze(loaded_model.predict(x_in_scaled))
    u_opt = u_opt_scaled * (u_ub - u_lb) + u_lb
    u_opt_lim = NP.maximum(NP.minimum(u_opt,u_ub),u_lb)

    # # Simulate the system one step using the solution obtained in the optimization
    # configuration_1.make_step_simulator() # NOTE: included in step_observer
    # # projection when constraint violated or will be violated
    # # make_projection(configuration_1)
    #
    # # Make one observer step
    # configuration_1.make_step_observer()
    #
    # # Store the information
    # configuration_1.store_mpc_data()
    #
    # # Set initial condition constraint for the next iteration
    # configuration_1.prepare_next_iter()

    # Store both solutions and flag about the feasibility of the problem
    stats = configuration_1.optimizer.solver.stats()
    success = stats['success']
    if success == True:
        flag_feas = [1]
    else:
        flag_feas = [0]
    u_mpc_all[:,i:i+1] = NP.array([configuration_1.optimizer.u_mpc, u_opt_lim, flag_feas])
    print("--- Batch number " + str(i) + " --- T = " + str(t0_sim) + "s ---")
    # Export data
    # data_do_mpc.plot_mpc(configuration_1)
    # data_do_mpc.export_for_learning(configuration_1, "data/2_uncertainties_NN/data_batch_" + str(i))
    print(u_mpc_all[:,i])
    NP.save("empirical_risk", u_mpc_all)
# Remove infeasible solutions
u_clean = u_mpc_all.T
remove_rows = []
for i in range(u_clean.shape[0]):
    if u_clean[i,2] == 0:
        # remove the data with unfeasible solutions
        remove_rows.append(i)
u_clean = NP.delete(u_clean,remove_rows,0)
