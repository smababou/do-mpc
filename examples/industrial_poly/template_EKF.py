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
    open_loop = True
    t_step = 1.0/3600.0 # Sampling time
    parameter_estimation = True

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

    x_init = NP.array([m_W_0, m_A_0, m_P_0, T_R_0, T_S_0, Tout_M_0, T_EK_0, Tout_AWT_0, accum_momom_0])

    p_init = NP.array([950.0, 7.5])*1.2

    P = NP.diag([0.0001,0.01,0.0001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,2,0.02])

    Q = NP.diag([0.01,0.01,0.01,0.0001,0.0001,0.0001,0.0001,0.0001,0.1,0.05,0.00001])

    R = NP.diag([0.1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0])

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

    observer_dict = {'method':method,'t_step_observer':t_step,
                     'parameter_estimation':parameter_estimation,
                     'noise':noise, 'mag':mag,
                     'x_init':x_init, 'p_init':p_init,
                     'open_loop':open_loop,
                     'P':P, 'Q':Q, 'R':R}

    observer_1 = core_do_mpc.observer(model,observer_dict)

    return observer_1
