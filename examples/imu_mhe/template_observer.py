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
import core_do_mpc
import numpy as NP
import pdb
def observer(model):

    method = 'state-feedback' # 'EKF' or 'MHE' or 'state-feedback'

    """
    --------------------------------------------------------------------------
    template_observer: preprocess model information
    --------------------------------------------------------------------------
    """

    # Variables
    x = model.x
    u = model.u
    p = model.p
    tv_p = model.tv_p
    y = model.y
    meas_fcn = Function("meas_fcn",[x,u,p,tv_p],[y])

    nx = x.size(1)
    nu = u.size(1)
    np = p.size(1)
    ny = y.size(1)

    """
    --------------------------------------------------------------------------
    template_observer: integration options
    --------------------------------------------------------------------------
    """
    # Choose the simulator time step
    t_step_observer = 0.01
    # Simulation time
    t_end = 10.0
    # Choose options for the integrator
    opts = {"abstol":1e-10,"reltol":1e-10, 'tf':t_step_observer}
    # Choose integrator: for example 'cvodes' for ODEs or 'idas' for DAEs
    integration_tool = 'cvodes'

    """
    --------------------------------------------------------------------------
    template_observer: tuning parameters
    --------------------------------------------------------------------------
    """

    # Prediction horizon
    n_horizon = 20
    # Robust horizon, set to 0 for standard NMPC
    n_robust = 0
    # open_loop robust NMPC (1) or multi-stage NMPC (0). Only important if n_robust > 0
    open_loop = 0
    # Choose if optimal control instead of MPC
    optimal_control = 0
    # Sampling time
    t_step = 5.0/3600.0
    # Choose type of state discretization (collocation or multiple-shooting)
    state_discretization = 'discrete-time'
    # Degree of interpolating polynomials: 1 to 5
    poly_degree = 2
    # Collocation points: 'legendre' or 'radau'
    collocation = 'radau'
    # Number of finite elements per control interval
    n_fin_elem = 1
    # NLP Solver and linear solver
    nlp_solver = 'ipopt'
    qp_solver = 'qpoases'

    # It is highly recommended that you use a more efficient linear solver
    # such as the hsl linear solver MA27, which can be downloaded as a precompiled
    # library and can be used by IPOPT on run time

    linear_solver = 'ma27'

    # GENERATE C CODE shared libraries NOTE: Not currently supported
    generate_code = 0

    """
    --------------------------------------------------------------------------
    template_optimizer: uncertain parameters
    --------------------------------------------------------------------------
    """
    # Define the different possible values of the uncertain parameters in the scenario tree
    delH_R_values = NP.array([950.0, 950.0 * 1.30, 950.0 * 0.70])
    k_0_values = NP.array([7.0*1.00, 7.0*1.30, 7.0*0.70])
    uncertainty_values = NP.array([delH_R_values, k_0_values])

    """
    --------------------------------------------------------------------------
    template_optimizer: time-varying parameters
    --------------------------------------------------------------------------
    """
    # Only necessary if time-varying paramters defined in the model
    # The length of the vector for each parameter should be the prediction horizon
    # The vectos for each parameter might chance at each sampling time
    number_steps = int(t_end/t_step) + 1
    # Number of time-varying parameters
    n_tv_p = 2
    tv_p_values = NP.resize(NP.array([]),(number_steps,n_tv_p,n_horizon))
    for time_step in range (number_steps):
        if time_step < number_steps/2:
            tv_param_1_values = 0.6*NP.ones(n_horizon)
        else:
            tv_param_1_values = 0.8*NP.ones(n_horizon)
        tv_param_2_values = 0.9*NP.ones(n_horizon)
        tv_p_values[time_step] = NP.array([tv_param_1_values,tv_param_2_values])
    # Parameteres of the NLP which may vary along the time (For example a set point that varies at a given time)
    set_point = SX.sym('set_point')
    parameters_nlp = NP.array([set_point])

    """
    --------------------------------------------------------------------------
    template_observer: tuning parameters mhe
    --------------------------------------------------------------------------
    """

    # P_states = NP.diag(NP.ones(nx))*0.0
    P_states = NP.diag([1,1,1,0.01,0.01,0.01,0.01,0.01,0.01])*0.000

    P_param = NP.diag(NP.ones([np]))

    P_inputs = NP.diag(NP.ones([nu]))*0

    P_meas = NP.diag([10, 1, 1, 1, 1, 10, 1])
    # P_meas = NP.diag([1, 1, 1, 10000, 1, 1, 1, 1, 1, 1])

    """
    --------------------------------------------------------------------------
    template_observer: tuning parameters EKF
    --------------------------------------------------------------------------
    """

    c_pR = 5.0
    m_W_0 = 10000.0
    m_A_0 = 853.0*1.0  #3700.0
    m_P_0 = 26.5
    T_R_0  = 90 + 273.15
    T_S_0  = 90 + 273.15
    Tout_M_0  = 90 + 273.15
    T_EK_0 = 35 + 273.15
    Tout_AWT_0= 35 + 273.15
    delH_R_real = 950.0*1.00
    T_adiab_0		= m_A_0*delH_R_real/((m_W_0+m_A_0+m_P_0)*c_pR)+T_R_0

    accum_momom_0   = 300.0

    x_init = NP.array([m_W_0, m_A_0, m_P_0, T_R_0, T_S_0, Tout_M_0, T_EK_0, Tout_AWT_0, accum_momom_0,T_adiab_0])

    P_init = NP.diag([0.0001,0.01,0.0001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.01,2,0.02])
    # P_init = NP.ones([12,12])
    # P_init[10,10] = 27000
    # P_init[11,11] = 10

    Q = NP.diag([0.01,0.01,0.01,0.0001,0.0001,0.0001,0.0001,0.0001,0.1,0.1,0.05,0.00001])
    #Q = NP.diag([0,0,0,0.00,0.00,0.00,0.00,0.00,0.0000,0.0,0.01,0.001])
    R = NP.diag([0.1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0])

    # Q_update = 'sensitivity' # monte-carlo or sensitivity
    #
    # CP = NP.diag([27075.0, 6.1777]) #covariance of the parameters
    #
    # n_sim = 500 #number of monte-carlo simulations to obtain process noise covariance matrix

    """
    --------------------------------------------------------------------------
    template_observer: tuning parameters EKF
    --------------------------------------------------------------------------
    """

    alpha = 1e-3
    kappa = 0
    beta = 2

    """
    --------------------------------------------------------------------------
    template_observer: measurement function
    --------------------------------------------------------------------------
    """

    noise = 'gaussian'
    # mag = NP.ones(ny)*0.001 #standard deviation
    mag = NP.array([0.1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0])*1


    """
    --------------------------------------------------------------------------
    template_observer: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    observer_dict = {'n_horizon':n_horizon,'state_discretization':state_discretization,
    'poly_degree':poly_degree,'collocation':collocation,'n_fin_elem':n_fin_elem,
    'nlp_solver':nlp_solver,'qp_solver':qp_solver,'linear_solver':linear_solver,
    'generate_code':generate_code,'x':x,'meas_fcn':meas_fcn,'noise':noise,
    'mag':mag,'t_step':t_step,'open_loop':open_loop,'n_robust':n_robust,
    'integration_tool':integration_tool,'method':method,
    't_step_observer': t_step_observer, 'integrator_opts': opts,
    'P_states': P_states, 'P_param': P_param, 'P_inputs': P_inputs,
    'P_meas': P_meas, 'uncertainty_values':uncertainty_values,
    'tv_p_values':tv_p_values,'x_init':x_init,'P_init':P_init,
    'Q':Q,'R':R}

    observer_1 = core_do_mpc.observer(model,observer_dict)

    return observer_1
