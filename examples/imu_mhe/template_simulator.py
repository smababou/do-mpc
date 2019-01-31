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
import scipy.io as sio
from quaternion_aux import *

def simulator(model):

    """
    --------------------------------------------------------------------------
    template_simulator: integration options
    --------------------------------------------------------------------------
    """
    # Choose the simulator time step
    t_step_simulator = 0.01
    # Choose options for the integrator
    opts = {"abstol":1e-10,"reltol":1e-10, 'tf':t_step_simulator}
    # Choose integrator: for example 'cvodes' for ODEs or 'idas' for DAEs
    integration_tool = 'cvodes'

    # Choose the real value of the uncertain parameters that will be used
    # to perform the simulation of the system. They can be constant or time-varying
    def p_real_now(current_time):
        p_real =  NP.zeros(12)
        return p_real

    # Choose the real value of the time-varing parameters
    data = sio.loadmat("simulation_data_without_disturbance.mat", squeeze_me=True, struct_as_record=False)
    # True measurements
    rate = float(data["meta"].rate)
    # pdb.set_trace()
    acc1_ = data["imu"].imu1.acc_ideal[1:,:]
    gyr1_ = data["imu"].imu1.gyr_ideal[1:,:]
    acc2_ = data["imu"].imu2.acc_ideal[1:,:]
    gyr2_ = data["imu"].imu2.gyr_ideal[1:,:]
    N = acc1_.shape[0]
    # if noise and variables
    # pdb.set_trace()
    acc1_dist = 0.1 * (2*NP.tile(NP.random.rand(1,3),(N,1))-1) + 0.1*NP.random.randn(N,3)
    acc2_dist = 0.1 * (2*NP.tile(NP.random.rand(1,3),(N,1))-1) + 0.1*NP.random.randn(N,3)
    gyr1_dist = 2*NP.pi/360.0 * (2*NP.tile(NP.random.rand(1,3),(N,1))-1) + 2*NP.pi/360.0 * NP.random.randn(N,3)
    gyr2_dist = 2*NP.pi/360.0 * (2*NP.tile(NP.random.rand(1,3),(N,1))-1) + 2*NP.pi/360.0 * NP.random.randn(N,3)
    # add the noise and bias
    acc1_noisy = acc1_ + acc1_dist
    gyr1_noisy = gyr1_ + gyr1_dist
    acc2_noisy = acc2_ + acc2_dist
    gyr2_noisy = gyr2_ + gyr2_dist
    def tv_p_real_now(current_time):
        tv_p_real = NP.array([0.0, 0.0])
        return tv_p_real
    def tv_u_real_now(current_time):
        ii = int(round(current_time/0.01))
        tv_u_real = NP.squeeze(vertcat(acc1_[ii], gyr1_[ii], acc2_[ii],gyr2_[ii]))
        tv_u_real_noisy = NP.squeeze(vertcat(acc1_noisy[ii], gyr1_noisy[ii], acc2_noisy[ii],gyr2_noisy[ii]))
        return tv_u_real, tv_u_real_noisy
    """
    --------------------------------------------------------------------------
    template_simulator: plotting options
    --------------------------------------------------------------------------
    """

    # Choose the indices of the states to plot
    plot_states = [0,1,8,9,10,14,15,16]
    # Choose the indices of the controls to plot
    plot_control = [0]
    # Plot animation (False or True)
    plot_anim = False
    # Export to matlab (for better plotting or postprocessing)
    export_to_matlab = True
    export_name = "mpc_result.mat"  # Change this name if desired

    """
    --------------------------------------------------------------------------
    template_simulator: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """

    simulator_dict = {'integration_tool':integration_tool,'plot_states':plot_states,
    'plot_control': plot_control,'plot_anim': plot_anim,'export_to_matlab': export_to_matlab,'export_name': export_name, 'p_real_now':p_real_now,
    't_step_simulator': t_step_simulator, 'integrator_opts': opts, 'tv_p_real_now':tv_p_real_now, 'tv_u_real_now': tv_u_real_now}

    simulator_1 = core_do_mpc.simulator(model, simulator_dict)

    return simulator_1
