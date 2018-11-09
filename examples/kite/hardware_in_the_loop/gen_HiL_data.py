# Run different batches with different initial conditions to get reasonable training data

# This is the main path of your do-mpc installation relative to the execution folder
path_do_mpc = '../../../'
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
import observer_do_mpc
import pdb

# For communication to arduino
import serial
from time import sleep

# number of batches to generate data from
offset = 3
controller_number = 2
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
Establish connection to arduino
-----------------------------------------------
"""
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1.0)

"""
-----------------------------------------------
Load neural network
-----------------------------------------------
"""
if neural_network:
    filename = '../controller_' + str(controller_number)
    json_file = open(filename+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename+".h5")
    print("Loaded model from disk")

for i in range(offset, offset + 1):

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
    if neural_network:
        configuration_1.make_step_optimizer()
    configuration_1.simulator.p_real_batch = NP.zeros([2])

    # The initial states for the batches are:
    theta_0 = 0.29359907+0.05
    phi_0 = 0.52791537
    psi_0 = 0.0

    initial_state_batch = NP.array([theta_0, phi_0, psi_0])

    # choose the real parameters
    E_batch = 5.0 #NP.random.uniform(4.0,6.0)
    # c_batch = NP.random.uniform(0.005,0.04)

    w_mean = 10.0 #8.0 + NP.random.uniform() * 4.0
    var_t = 0.05 + NP.random.uniform() * 0.1
    w_lb = NP.array([7.0])
    w_ub = NP.array([13.0])
    w_amp_max = NP.minimum(NP.abs(w_ub-w_mean),NP.abs(w_lb-w_mean))
    w_amp = NP.random.uniform() * w_amp_max
    w_shift = 0.0 #NP.random.uniform() * 2.0 * pi
    w_init = w_mean + w_amp * sin(w_shift)

    configuration_1.simulator.p_real_batch[0] = E_batch
    # configuration_1.simulator.p_real_batch[1] = c_batch
    configuration_1.simulator.p_real_batch[-1] = w_init

    configuration_1.observer.ekf.x_hat[3] = E_batch #+ NP.random.normal(0,0.05)
    configuration_1.observer.ekf.x_hat[-1] = w_init #+ NP.random.normal(0,0.1)

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

    # Do not stop until a predefined amount of polymer has been produced
    while (configuration_1.simulator.t0_sim + configuration_1.simulator.t_step_simulator < configuration_1.optimizer.t_end):

        # Update wind
        t0_sim = configuration_1.simulator.t0_sim
        configuration_1.simulator.p_real_batch[-1] = w_mean + w_amp * sin(2*pi*var_t*t0_sim+w_shift)

        # get optimal input
        while (arduino.in_waiting < 1):
            sleep(0.05)
        u_opt_byte = arduino.readline()
        u_opt = float(u_opt_byte.decode("utf8"))
        u_opt_lim = NP.maximum(NP.minimum(u_opt,10.0),-10.0)
        configuration_1.optimizer.u_mpc = u_opt_lim

        for j in range(3):

            # simulate one step
            configuration_1.make_step_simulator()

            # obtain measurements and send to micro
            observer_do_mpc.make_measurement(configuration_1)
            arduino.write(bytes(str(configuration_1.observer.measurement[0]),"utf8"))
            sleep(0.05)
            arduino.write(bytes(str(configuration_1.observer.measurement[1]),"utf8"))

            while (arduino.in_waiting < 1):
                sleep(0.05)

            is_cont_byte = arduino.readline();
            is_cont = is_cont_byte.decode("utf8")
            if not (is_cont == "continue\r\n"):
                raise NameError('Did not get command to continue!')

        while (arduino.in_waiting < 1):
            sleep(0.05)
        #  obtain estimated states
        theta_est_byte = arduino.readline()
        theta_est = float(theta_est_byte.decode("utf8"))
        phi_est_byte = arduino.readline()
        phi_est = float(phi_est_byte.decode("utf8"))
        psi_est_byte = arduino.readline()
        psi_est = float(psi_est_byte.decode("utf8"))

        # obtain estimated parameters
        E0_est_byte = arduino.readline()
        E0_est = float(E0_est_byte.decode("utf8"))
        v0_est_byte = arduino.readline()
        v0_est = float(v0_est_byte.decode("utf8"))

        # save ekf results
        configuration_1.observer.ekf.x_hat = NP.array([theta_est,phi_est,psi_est,E0_est,v0_est])
        print(NP.array([theta_est,phi_est,psi_est,E0_est,v0_est]))

        # Store the information
        configuration_1.store_mpc_data()

    # Export data
    data_do_mpc.plot_mpc(configuration_1)
    data_do_mpc.export_for_learning(configuration_1, "results/data_batch_" + str(i))
