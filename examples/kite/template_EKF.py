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

    """
    --------------------------------------------------------------------------
    template_observer: settings
    --------------------------------------------------------------------------
    """

    method = 'EKF'
    open_loop = False
    t_step = 0.05 # Sampling time
    parameter_estimation = True

    """
    --------------------------------------------------------------------------
    template_observer: tuning parameters EKF
    --------------------------------------------------------------------------
    """

    theta_0 = 0.29359907+0.05
    phi_0 = 0.52791537
    psi_0 = 0.0
    E_0 = 5.0
    c_0 = 0.028
    v_0 = 10.0

    if parameter_estimation:
        x_init = NP.array([theta_0, phi_0, psi_0, E_0, v_0])
        P = NP.diag([0.01,0.01,0.01,0.1,0.2])
        Q = NP.diag([0.001,0.001,0.05,0.005,1.0])
    else:
        x_init = NP.array([theta_0, phi_0, psi_0])
        P = NP.diag([0.001,0.001,0.001])
        Q = NP.diag([0.0,0.0,0.0])

    R = NP.diag([0.01,0.01])

    """
    --------------------------------------------------------------------------
    template_observer: measurement function
    --------------------------------------------------------------------------
    """

    noise = 'gaussian'
    mag = NP.array([0.01, 0.01])

    """
    --------------------------------------------------------------------------
    template_observer: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """

    observer_dict = {'method':method,'t_step_observer':t_step,
                     'parameter_estimation':parameter_estimation,
                     'noise':noise, 'mag':mag,
                     'x_init':x_init, 'open_loop':open_loop,
                     'P':P, 'Q':Q, 'R':R}

    observer_1 = core_do_mpc.observer(model,observer_dict)

    return observer_1
