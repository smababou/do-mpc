# Run different batches with different initial conditions to get reasonable training data

# This is the main path of your do-mpc installation relative to the execution folder
path_do_mpc = '../../../'
# Add do-mpc path to the current directory
import sys
sys.path.insert(0,path_do_mpc+'code')
sys.path.insert(0,'..')
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

# parameters
n_split = 5
L = 400.0
counter_lim = 1000

theta_array = NP.linspace(0.0,0.5*pi, num=n_split, endpoint=True)
phi_array = NP.linspace(-0.5*pi,0.5*pi, num=n_split, endpoint=True)
psi_array = NP.linspace(-1.0*pi,1.0*pi, num=n_split, endpoint=True)

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
--------------------------------------------------------------------------
Load configuration
--------------------------------------------------------------------------
"""

# Create the objects for each module
model_1 = template_model.model()
# Create an optimizer object based on the template and a model
optimizer_1 = template_optimizer.optimizer(model_1)
# Create an observer object based on the template and a model
observer_1 = template_EKF.observer(model_1)
# Create a simulator object based on the template and a model
simulator_1 = template_simulator.simulator(model_1)
# Create a configuration
conf = core_do_mpc.configuration(model_1, optimizer_1, observer_1, simulator_1)
conf.setup_solver()

"""
--------------------------------------------------------------------------
Gen data
--------------------------------------------------------------------------
"""

X_offset = conf.optimizer.nlp_dict_out['X_offset']
nx = conf.model.x.shape[0]
nu = conf.model.u.shape[0]
current_batch = NP.resize(NP.array([]),(0,nx+nu))
counter = 0
new_map = True

for theta in theta_array:
    for phi in phi_array:
        for psi in psi_array:
            if (L*sin(theta)*cos(phi)>=100.0):
                conf.optimizer.arg['lbx'][X_offset[0,0]:X_offset[0,0]+nx] = NP.squeeze(NP.array([theta,phi,psi]))
                conf.optimizer.arg['ubx'][X_offset[0,0]:X_offset[0,0]+nx] = NP.squeeze(NP.array([theta,phi,psi]))
                conf.make_step_optimizer()
                stats = conf.optimizer.solver.stats()
                success = stats['success']
                if success:
                    u_opt = NP.squeeze(conf.optimizer.u_mpc)
                    new_data_point = NP.reshape(NP.array([theta,phi,psi,u_opt]),(1,-1))
                    current_batch = NP.append(current_batch,new_data_point,axis=0)
                    counter += 1

                    if counter == counter_lim:
                        if new_map:
                            NP.save('map_'+str(n_split),current_batch)
                        else:
                            old_batches = NP.load('map_'+str(n_split)+'.npy')
                            map_to_save = NP.vstack((old_batches,current_batch))
                            NP.save('map_'+str(n_split),map_to_save)
                            del old_batches, map_to_save
                        current_batch = NP.resize(NP.array([]),(0,nx+nu))
                        counter = 0
                        new_map = False

if new_map:
    NP.save('map_'+str(n_split),current_batch)
else:
    old_batches = NP.load('map_'+str(n_split)+'.npy')
    map_to_save = NP.vstack((old_batches,current_batch))
    NP.save('map_'+str(n_split),map_to_save)
