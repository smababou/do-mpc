# Run different batches with different initial conditions to get reasonable training data

# This is the main path of your do-mpc installation relative to the execution folder
path_do_mpc = '../../'
# Add do-mpc path to the current directory
import sys
sys.path.insert(0,path_do_mpc+'code')
# Do not write bytecode to maintain clean directories
sys.dont_write_bytecode = True

# Import numpy
import numpy as NP
# Start CasADi
from casadi import *
# Import do-mpc core functionalities
import core_do_mpc
# Import do-mpc plotting and data managament functions
import data_do_mpc
import pdb

# number of batches to generate data from
n_batches = 1
offset = 0
# initialize the problem (first lines of do_mpc.py)

"""
-----------------------------------------------
do-mpc: Definition of the do-mpc configuration
-----------------------------------------------
"""

# Import the user defined modules
import template_model
import template_model_observer
import template_optimizer
import template_observer
import template_simulator

# Create the objects for each module
model_1 = template_model.model()
model_observer = template_model_observer.model()
# Create an optimizer object based on the template and a model
optimizer_1 = template_optimizer.optimizer(model_1)
# Create an observer object based on the template and a model
observer_1 = template_observer.observer(model_observer)
# Create a simulator object based on the template and a model
simulator_1 = template_simulator.simulator(model_1)
# Create a configuration
configuration_1 = core_do_mpc.configuration(model_1, optimizer_1, observer_1, simulator_1)

# Set up the solvers
configuration_1.setup_solver()



for i in range(offset, offset + n_batches):
    # The initial states for the batches are:
    # m_W_0 = (0.1*NP.random.randn(1) + 1 ) * 10000.0
    # m_A_0 = (0.1*NP.random.randn(1) + 1 ) * 853.0*1.0  #3700.0
    # m_P_0 = (0.1*NP.random.randn(1) + 1 ) * 26.5
    # T_R_0  = 90 + 273.15 + NP.random.uniform(-1.0,1.0)
    # T_S_0  = 90 + 273.15 + NP.random.uniform(-1.0,1.0)
    # Tout_M_0  = 90 + 273.15 + NP.random.uniform(-1.0,1.0)
    # T_EK_0 = 35 + 273.15 + NP.random.uniform(-1.0,1.0) * 5
    # Tout_AWT_0= 35 + 273.15 + NP.random.uniform(-1.0,1.0) * 5
    # accum_momom_0   = (0.1*NP.random.randn(1) + 1 ) *  300.0
    # # choose the real parameters
    # configuration_1.simulator.p_real_batch = NP.array([950.0, 7.0]) * NP.random.uniform(0.7, 1.3, 2)
    # delH_R_real = configuration_1.simulator.p_real_batch[0]
    # c_pR        = 5.0
    # T_adiab_0		= m_A_0*delH_R_real/((m_W_0+m_A_0+m_P_0)*c_pR)+T_R_0
    # initial_state_batch = NP.array([m_W_0, m_A_0, m_P_0, T_R_0, T_S_0, Tout_M_0, T_EK_0, Tout_AWT_0, accum_momom_0,T_adiab_0])
    initial_state_batch = NP.load("init_values/init_" + str(i) + ".npy")
    m_W_0 = initial_state_batch[0]
    m_A_0 = initial_state_batch[1]
    m_P_0 = initial_state_batch[2]
    T_R_0  = initial_state_batch[3]
    T_S_0  = initial_state_batch[4]
    Tout_M_0  = initial_state_batch[5]
    T_EK_0 = initial_state_batch[6]
    Tout_AWT_0= initial_state_batch[7]
    accum_momom_0   = initial_state_batch[8]
    T_adiab_0 = initial_state_batch[9]
    # choose the real parameters
    # configuration_1.simulator.p_real_batch = NP.array([950.0, 7.0]) * NP.random.uniform(0.7, 1.3, 2)
    configuration_1.simulator.p_real_batch = initial_state_batch[10:12]
    configuration_1.simulator.p_real_batch = NP.array([950.0*1.2, 7.0*1.2])
    # delH_R_real = configuration_1.simulator.p_real_batch[0]
    # c_pR        = 5.0
    # T_adiab_0		= m_A_0*delH_R_real/((m_W_0+m_A_0+m_P_0)*c_pR)+T_R_0
    # Update initial condition for this batch
    x_scaling = configuration_1.model.ocp.x_scaling
    X_offset = configuration_1.optimizer.nlp_dict_out['X_offset']
    nx = len(configuration_1.model.ocp.x0)
    configuration_1.optimizer.arg['lbx'][X_offset[0,0]:X_offset[0,0]+nx] = initial_state_batch[:10] / x_scaling
    configuration_1.optimizer.arg['ubx'][X_offset[0,0]:X_offset[0,0]+nx] = initial_state_batch[:10] / x_scaling
    configuration_1.simulator.x0_sim = DM(initial_state_batch[:10]) / x_scaling
    configuration_1.model.ocp.x0 = initial_state_batch[:10]
    # Restart iteration counter
    configuration_1.mpc_iteration = 1
    configuration_1.simulator.t0_sim = 0
    configuration_1.simulator.tf_sim = 0 + configuration_1.simulator.t_step_simulator
    configuration_1.mpc_data = data_do_mpc.mpc_data(configuration_1)
    # Do not stop until a predefined amount of polymer has been produced
    while (configuration_1.simulator.x0_sim[2] * configuration_1.model.ocp.x_scaling[2] < 20681-m_P_0-accum_momom_0-200):#(configuration_1.simulator.x0_sim[2] * configuration_1.model.ocp.x_scaling[2] < 20681-m_P_0-accum_momom_0):
        print(configuration_1.simulator.x0_sim[2] * configuration_1.model.ocp.x_scaling[2])
        print(20681-m_P_0-accum_momom_0-200)
        print(configuration_1.optimizer.u_mpc)
        # Make one optimizer step (solve the NLP)
        configuration_1.make_step_optimizer()
        #print configuration_1.simulator.x0_sim[2] * configuration_1.model.ocp.x_scaling[2]
        #print 20681-m_P_0-accum_momom_0
        # Simulate the system one step using the solution obtained in the optimization
        configuration_1.make_step_simulator()

        # Make one observer step
        configuration_1.make_step_observer()

        # Store the information
        configuration_1.store_mpc_data()

        # Set initial condition constraint for the next iteration
        configuration_1.prepare_next_iter()
        data_do_mpc.plot_animation(configuration_1)
        print("--- Batch number ---" + str(i))
    # Export data
    # data_do_mpc.plot_mpc(configuration_1)
    exp_name = "sim_data/MPC_EKF_"+str(i)
    configuration_1.simulator.export_name = exp_name
    data_do_mpc.export_to_matlab(configuration_1)
    data_do_mpc.plot_mpc(configuration_1)
    pdb.set_trace()
    # data_do_mpc.export_for_learning(configuration_1, "training_data/data_batch_" + str(i))
