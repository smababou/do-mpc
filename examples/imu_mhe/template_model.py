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

from casadi import *
import numpy as NP
import core_do_mpc
import pdb
import scipy.io as sio
from quaternion_aux import *
def model():

    """
    --------------------------------------------------------------------------
    template_model: define the non-uncertain parameters
    --------------------------------------------------------------------------
    """

    data = sio.loadmat("simulation_data_without_disturbance.mat", squeeze_me=True, struct_as_record=False)
    # True measurements
    rate = float(data["meta"].rate)*1.0
    acc1_ = data["imu"].imu1.acc_ideal[0:,:]
    gyr1_ = data["imu"].imu1.gyr_ideal[0:,:]
    acc2_ = data["imu"].imu2.acc_ideal[0:,:]
    gyr2_ = data["imu"].imu2.gyr_ideal[0:,:]

    # True outputs (unknown information)
    quat1_ref = data["ref"].imu1.ori_imu.quat[0:,:]
    quat2_ref = data["ref"].imu2.ori_imu.quat[0:,:]
    pos1_ref = data["ref"].imu1.pos[0:,:]
    pos2_ref = data["ref"].imu2.pos[0:,:]
    vel1_ref = NP.insert(NP.diff(pos1_ref, axis = 0)*rate, 0,0, axis = 0)
    vel2_ref = NP.insert(NP.diff(pos2_ref, axis = 0)*rate, 0,0, axis = 0)
    # Joint center positions
    o1 = NP.array([0.01, -0.02, 0])
    o2 = NP.array([0.01,  0.015, 0])
    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    alpha   = SX.sym("alpha")
    beta    = SX.sym("beta")
    # Biases for the sensors
    acc1_bias = SX.sym("acc1_bias", 3)
    gyr1_bias = SX.sym("gyr1_bias", 3)
    acc2_bias = SX.sym("acc2_bias", 3)
    gyr2_bias = SX.sym("gyr2_bias", 3)
    # Define the differential states as CasADi symbols
    quat1    = SX.sym("quat1",4) # Concentration A
    quat2    = SX.sym("quat2",4) # Concentration B
    vel1    = SX.sym("vel1",3) # Reactor Temprature
    vel2    = SX.sym("vel2",3) # Jacket Temprature
    pos1    = SX.sym("pos1",3) # Reactor Temprature
    pos2    = SX.sym("pos2",3) # Jacket Temprature
    # Define the outputs as symbols
    acc1 = SX.sym("acc1",3)
    gyr1 = SX.sym("gyr1",3)
    acc2 = SX.sym("acc2",3)
    gyr2 = SX.sym("gyr2",3)
    # Define the algebraic states as CasADi symbols
    acc1_real = acc1 + acc1_bias
    gyr1_real = gyr1 + gyr1_bias
    acc2_real = acc2 + acc2_bias
    gyr2_real = gyr2 + gyr2_bias
    # Define the control inputs as CasADi symbols

    u      = SX.sym("u") # Vdot/V_R [h^-1]


    # Define time-varying parameters that can change at each step of the prediction and at each sampling time of the MPC controller. For example, future weather predictions

    tv_param_1 = SX.sym("tv_param_1")
    tv_param_2 = SX.sym("tv_param_2")
    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    # Define the algebraic equations

    # Define the differential equations

    dquat1 = quaternionMultiply(quat1, quaternionFromGyr(gyr1_real, rate))
    dquat2 = quaternionMultiply(quat2, quaternionFromGyr(gyr2_real, rate))

    dvel1 = vel1 + (quaternionRotate(dquat1, acc1_real) - NP.array([0.0, 0.0, 9.81]))/rate
    dvel2 = vel2 + (quaternionRotate(dquat2, acc2_real) - NP.array([0.0, 0.0, 9.81]))/rate

    dpos1 = pos1 + (dvel1)/rate
    dpos2 = pos2 + (dvel2)/rate

    # Center positions for the Constraints
    c1 = mtimes(quaternionRotate(dquat1, NP.array([0.0,0.0,1.0])).T, quaternionRotate(dquat2, NP.array([0.0,0.0,1.0]))) - 1
    center1 = dpos1 + quaternionRotate(dquat1, o1)
    center2 = dpos2 + quaternionRotate(dquat2, o2)
    c2 = norm_2(center1 - center2)

    # Make it differential equations
    #
    # dquat1 =  rate * (-quat1 +  quaternionMultiply(quat1, quaternionFromGyr(gyr1, rate)))
    # dquat2 =  rate * (-quat2 +  quaternionMultiply(quat2, quaternionFromGyr(gyr2, rate)))
    #
    # dvel1 = (quaternionRotate(quat1, acc1) - NP.array([0.0, 0.0, 9.81]))
    # dvel2 = (quaternionRotate(quat2, acc2) - NP.array([0.0, 0.0, 9.81]))
    #
    # dpos1 = vel1
    # dpos2 = vel2
    #
    # # Center positions for the Constraints
    # c1 = mtimes(quaternionRotate(quat1, NP.array([0.0,0.0,1.0])).T, quaternionRotate(quat2, NP.array([0.0,0.0,1.0]))) - 1
    # center1 = pos1 + quaternionRotate(quat1, o1)
    # center2 = pos2 + quaternionRotate(quat2, o2)
    # c2 = norm_2(center1 - center2)


    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = vertcat(quat1, quat2, vel1, vel2, pos1, pos2)

    _y = vertcat(c1, c2)

    _z = vertcat([])

    _u = vertcat(acc1, gyr1, acc2, gyr2)

    _xdot = vertcat(dquat1, dquat2, dvel1, dvel2, dpos1, dpos2)

    _p = vertcat(acc1_bias, gyr1_bias, acc2_bias, gyr2_bias)

    _tv_p = vertcat(tv_param_1, tv_param_2)



    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states

    quat1_0 = quat1_ref[0,:]
    quat2_0 = quat2_ref[0,:]
    vel1_0 = (quaternionRotate(quat1_0, acc1_[0,:]) -[0,0,9.81])/rate
    vel2_0 = (quaternionRotate(quat2_0, acc2_[0,:]) -[0,0,9.81])/rate
    pos1_0 = pos1_ref[0,:]
    pos2_0 = pos2_ref[0,:]

    # Wrong initial conditions
    quat1_0 = NP.array([0.0,0.0,1.0,0.0])
    quat2_0 = NP.array([0.0,1/NP.sqrt(2),1/NP.sqrt(2),0.0])
    # quat1_0 = NP.array([1.0,0.0,0.0,0.0])
    # quat2_0 = NP.array([1.0,0.0,0.0,0.0])
    vel1_0 = (quaternionRotate(quat1_0, acc1_[0,:]) -[0,0,9.81])/rate
    vel2_0 = (quaternionRotate(quat2_0, acc2_[0,:]) -[0,0,9.81])/rate
    pos1_0 = (1 + 0.1*NP.random.randn(3))*pos1_ref[0,:]
    pos2_0 = (1 + 0.1*NP.random.randn(3))*pos2_ref[0,:]
    x0 = NP.squeeze(vertcat(quat1_0, quat2_0, vel1_0,vel2_0, pos1_0, pos2_0))
    # No algebraic states
    z0 = NP.array([])

    # Bounds on the states. Use "inf" for unconstrained states

    x_lb = -1.08 * NP.ones(_x.shape[0])
    x_ub =  1.08 * NP.ones(_x.shape[0])

    x_lb = -500 * NP.ones(_x.shape[0])
    x_ub =  500 * NP.ones(_x.shape[0])

    x_lb[8:11]  = -2 * NP.ones(3)
    x_lb[11:14] = -2 * NP.ones(3)
    x_lb[14:17] = -10* NP.ones(3)
    x_lb[17:20] = -10*NP.ones(3)

    x_ub[8:11]  = 2 * NP.ones(3)
    x_ub[11:14] = 2 * NP.ones(3)
    x_ub[14:17] = 10* NP.ones(3)
    x_ub[17:20] = 10*NP.ones(3)
    # No algebraic states
    z_lb = NP.array([])
    z_ub = NP.array([])

    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    u_lb = -60 * NP.ones(_u.shape[0])
    u_ub =  60 * NP.ones(_u.shape[0])
    u0 = NP.squeeze(vertcat(acc1_[1,:], gyr1_[1,:], acc2_[1,:], gyr2_[1,:]))

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = 1.0*NP.squeeze(NP.ones(_x.shape))
    z_scaling = NP.array([])
    u_scaling = 1.0*NP.squeeze(NP.ones(_u.shape))
    y_scaling = 1.0*NP.squeeze(NP.ones(_y.shape))
    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    # cons = vertcat([norm_2(quat1), -norm_2(quat1), norm_2(quat2), -norm_2(quat2)])
    cons = vertcat([])
    cons_ub = NP.array([])
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    # cons_ub = NP.array([1,-1,1,-1])

    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 0
    # Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    penalty_term_cons = NP.array([])
    # Maximum violation for the constraints
    maximum_violation = NP.array([0])

    # Define the terminal constraint (leave it empty if not necessary)
    cons_terminal = vertcat()
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    cons_terminal_lb = NP.array([])
    cons_terminal_ub = NP.array([])


    """
    --------------------------------------------------------------------------
    template_model: cost function
    --------------------------------------------------------------------------
    """
    # Define the cost function
    # Lagrange term
    lterm =  tv_param_1

    # Mayer term
    mterm =  tv_param_1

    # Penalty term for the control movements
    rterm = NP.array([0.0])

    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'y': _y, 'rhs':_xdot,
                  'p': _p, 'z':_z,'x0': x0, 'x_lb': x_lb,'x_ub': x_ub,
                  'u0':u0, 'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling,
                  'u_scaling':u_scaling, 'y_scaling':y_scaling, 'cons':cons,
                  "cons_ub": cons_ub, 'cons_terminal':cons_terminal,
                  'cons_terminal_lb': cons_terminal_lb,'tv_p':_tv_p,
                  'cons_terminal_ub':cons_terminal_ub,
                  'soft_constraint': soft_constraint,
                  'penalty_term_cons': penalty_term_cons,
                  'maximum_violation': maximum_violation,
                  'mterm': mterm,'lterm':lterm,
                  'rterm':rterm}

    model = core_do_mpc.model(model_dict)

    return model
