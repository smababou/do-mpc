#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2018 Sergio Lucia, Alexandru Tatulea-Codrean
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

import sys
sys.path.append("/home/casadi-linux-py27-v3.4.5-64bit")
from casadi import *
import numpy as NP
import core_do_mpc
def model():

    """
    --------------------------------------------------------------------------
    template_model: define the non-uncertain parameters
    --------------------------------------------------------------------------
    """

#    K0_ab = 1.287e12 # K0 [h^-1]
#    K0_bc = 1.287e12 # K0 [h^-1]
#    K0_ad = 9.043e9 # K0 [l/mol.h]
#    R_gas = 8.3144621e-3 # Universal gas constant
#    E_A_ab = 9758.3*1.00 #* R_gas# [kj/mol]
#    E_A_bc = 9758.3*1.00 #* R_gas# [kj/mol]
#    E_A_ad = 8560.0*1.0 #* R_gas# [kj/mol]
#    H_R_ab = 4.2 # [kj/mol A]
#    H_R_bc = -11.0 # [kj/mol B] Exothermic
#    H_R_ad = -41.85 # [kj/mol A] Exothermic
#    Rou = 0.9342 # Density [kg/l]
#    Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
#    Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
#    A_R = 0.215 # Area of reactor wall [m^2]
#    V_R = 10.01 #0.01 # Volume of reactor [l]
#    m_k = 5.0 # Coolant mass[kg]
#    T_in = 130.0 # Temp of inflow [Celsius]
#    K_w = 4032.0 # [kj/h.m^2.K]
#    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]


    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    alpha   = SX.sym("alpha")
    beta    = SX.sym("beta")

    # Define the differential states as CasADi symbols
    x     = SX.sym("x")              # x position of the robot
    y     = SX.sym("y")              # y position of the robot
    theta = SX.sym("theta")          # heading angle of the robot
    
   # Define the algebraic states as CasADi symbols


    # Define the control inputs as CasADi symbols
    v     = SX.sym("v")               # linear velocity of the robot
    w     = SX.sym("w")               # angular velocity of the robot

    # Define time-varying parameters that can change at each step of the prediction and at each sampling time of the MPC controller. For example, future weather predictions

    tv_param_1 = SX.sym("tv_param_1")
    tv_param_2 = SX.sym("tv_param_2")


    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    # Define the algebraic equations

#    K_1 = beta * K0_ab * exp((-E_A_ab)/((T_R+273.15)))
#    K_2 =  K0_bc * exp((-E_A_bc)/((T_R+273.15)))
#    K_3 = K0_ad * exp((-alpha*E_A_ad)/((T_R+273.15)))

    # Define the differential equations
    dd = SX.sym("dd",3)
    dd[0]  = v*cos(theta)*alpha
    dd[1]  = v*sin(theta)*beta
    dd[2]  = w

    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = vertcat(x,y,theta)
    
    _u = vertcat(v,w)
    
    _xdot = vertcat(dd)
    
    _p = vertcat(alpha, beta)
    
    _z = []                                # toggle if there are no AE in your model

    _tv_p = vertcat(tv_param_1, tv_param_2)


    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states

    x_init      = 5.0
    y_init      = 5.0
    theta_init  = 0.0
    x0 = NP.array([x_init, y_init, theta_init])

    # Bounds on the states. Use "inf" for unconstrained states

    x_lb     =   0.0;             x_ub  = 30.0
    y_lb     =   0.0;             y_ub  = 30.0
    theta_lb =   -2*pi;       theta_ub  = 2*pi
    x_lb = NP.array([x_lb, y_lb, theta_lb])
    x_ub = NP.array([x_ub, y_ub, theta_ub])

    # Bounds on the control inputs. Use "inf" for unconstrained inputs

    v_lb    = -1.4;       v_ub = 1.4 ;     v_init = 0.0    ;
    w_lb    = -1.0;       w_ub = 1.0 ;     w_init = 0.0    ;
    u_lb = NP.array([v_lb, w_lb])
    u_ub = NP.array([v_ub, w_ub])
    u0 = NP.array([v_init,w_init])

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.array([1.0, 1.0, 1.0])
    u_scaling = NP.array([1.0, 1.0])

    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    cons = vertcat([])
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    cons_ub = NP.array([])

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
    lterm =  14*(x-15)**2 + 6*(y-15)**2 + 8*(theta-pi/2)**2 + v**2 + w**2

    # Mayer term
    mterm =  0

    # Penalty term for the control movements
    rterm = NP.array([0.0001, 0.0001])



    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z,'x0': x0,'x_lb': x_lb,'x_ub': x_ub, 'u0':u0, 'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'u_scaling':u_scaling, 'cons':cons,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb,'tv_p':_tv_p, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm}

    model = core_do_mpc.model(model_dict)

    return model
