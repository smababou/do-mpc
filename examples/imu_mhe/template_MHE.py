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
from scipy.linalg import block_diag

def observer(model):

    """
    --------------------------------------------------------------------------
    template_observer: settings
    --------------------------------------------------------------------------
    """

    method = 'MHE'
    open_loop = False
    t_step = 0.01 # Sampling time
    parameter_estimation = False

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
    template_observer: tuning parameters mhe
    --------------------------------------------------------------------------
    """

    nx = model.x.size(1)
    np = model.p.size(1)
    nu = model.u.size(1)
    ny = model.y.size(1)

    P_states = n_horizon * 0.001 * NP.diag(NP.ones(nx))

    P_param = 0 * NP.diag(NP.ones([np]))
    # Different penalties for each input
    P_gyr = 1.0/(2*NP.pi/360.0) * NP.diag(NP.ones([nu/4]))
    P_acc = 1.0/(0.1) * NP.diag(NP.ones([nu/4]))

    P_inputs = 1 * block_diag(P_acc, P_gyr, P_acc, P_gyr)

    # P_meas = 10 * NP.diag(NP.ones(ny))
    # Choose if you want to discard the first constraint
    P_meas = 10 * NP.diag(NP.array([1,1]))

    """
    --------------------------------------------------------------------------
    template_observer: initial estimates and bounds
    --------------------------------------------------------------------------
    """


    x_init = model.ocp.x0
    p_init = NP.array([0.0,0.0])

    p_lb = NP.array([950.0, 7.0]) * 0.7
    p_ub = NP.array([950.0, 7.0]) * 1.3

    """
    --------------------------------------------------------------------------
    template_observer: measurement function
    --------------------------------------------------------------------------
    """

    noise = 'uniform'
    mag = NP.ones(ny)*0.000 #width


    """
    --------------------------------------------------------------------------
    template_observer: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    observer_dict = {'n_horizon':n_horizon,'state_discretization':state_discretization,
    'poly_degree':poly_degree,'collocation':collocation,'n_fin_elem':n_fin_elem,
    'nlp_solver':nlp_solver,'qp_solver':qp_solver,'linear_solver':linear_solver,
    'generate_code':generate_code,
    'noise':noise,'mag':mag,'n_robust':n_robust,
    'P_states': P_states, 'P_param': P_param, 'P_inputs': P_inputs,
    'P_meas': P_meas, 'uncertainty_values':uncertainty_values,
    'method':method,'open_loop':open_loop,'t_step_observer':t_step,
    'x_init':x_init,'p_init':p_init,'p_lb':p_lb,'p_ub':p_ub,
    'parameter_estimation':parameter_estimation}

    observer_1 = core_do_mpc.observer(model,observer_dict)

    return observer_1
