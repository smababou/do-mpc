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
import pdb
from casadi import *
import numpy as NP
import core_do_mpc
def model():

    """
    --------------------------------------------------------------------------
    template_model: define the non-uncertain parameters
    --------------------------------------------------------------------------
    """

    L_tether = 400.0        # Tether length [m]
    A = 300.0               # Area of the kite  [m^2]
    h_min = 100.0         # minimum height [m]
    rho = 1.0               # [kg/m^3]
    beta = 0
    E_0 = 5.0
    c_tilde = 0.028

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    # E_0   = SX.sym("E_0")
    # c_tilde  = SX.sym("c_tilde")
    v_0 = SX.sym("v_0")

    # Define the differential states as CasADi symbols

    theta    = SX.sym("theta") # zenith angle
    phi    = SX.sym("phi") # azimuth angle
    psi    = SX.sym("psi") # orientation kite

    # Define the algebraic states as CasADi symbols

    # Define the control inputs as CasADi symbols

    u_tilde      = SX.sym("u_tilde") # Vdot/V_R [h^-1]

    # Define time-varying parameters that can change at each step of the prediction and at each sampling time of the MPC controller. For example, future weather predictions

    tv_param_1 = SX.sym("tv_param_1")
    tv_param_2 = SX.sym("tv_param_2")

    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    # Define the algebraic equations

    E 		= E_0 - c_tilde * u_tilde**2
    v_a 	= v_0 * E * cos(theta)
    P_D 	= (rho * v_0**2)/2.0
    T_F		= (P_D * A * cos(theta)**2 * (E+1.0) * sqrt(E**2+1.0)) * (cos(theta) * cos(beta) + sin(theta) * sin(beta) * sin(phi))

    # Define the differential equations

    dtheta  = v_a / L_tether * (cos(psi) - tan(theta)/E)
    dphi    = -v_a / (L_tether * sin(theta)) * sin(psi)
    dpsi    = v_a/L_tether * u_tilde + dphi *(cos(theta))

    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = vertcat(theta, phi, psi)

    _y = vertcat(theta, phi)

    _u = vertcat(u_tilde)

    _xdot = vertcat(dtheta, dphi, dpsi)

    _p = vertcat(v_0)

    _z = []

    _tv_p = vertcat(tv_param_1, tv_param_2)

    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial condition for the states
    theta_0 = 0.29359907+0.05
    phi_0 = 0.52791537
    psi_0 = 0.0
    x0 = NP.array([theta_0, phi_0, psi_0])

    # Bounds on the states. Use "inf" for unconstrained states
    theta_lb = 0.0;			  theta_ub = 0.5*pi
    phi_lb = -0.5*pi;		      phi_ub = 0.5*pi
    psi_lb = -5.0*pi;		  psi_ub = 5.0*pi
    x_lb = NP.array([theta_lb, phi_lb, psi_lb])
    x_ub = NP.array([theta_ub, phi_ub, psi_ub])

    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    u_tilde_lb = -10.0;                 u_tilde_ub = 10.0;

    u_lb = NP.array([u_tilde_lb])
    u_ub = NP.array([u_tilde_ub])
    u0 = NP.array([0.0])

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.array([1.0, 1.0, 1.0])
    u_scaling = NP.array([1.0])
    y_scaling = NP.array([1.0,1.0])

    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    height_kite = L_tether * sin(theta) * cos(phi)
    cons = vertcat(-height_kite)
    # Define the upper bounds of the constraint (leave it empty if not necessary)

    cons_ub = NP.array(-h_min)

    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 0
    # l1 - Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    penalty_term_cons = NP.array([1e4])
    # Maximum violation for the upper and lower bounds
    maximum_violation = NP.array([10.0])

    # Define the terminal constraint (leave it empty if not necessary)
    cons_terminal = vertcat([])
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
    lterm =  -T_F
    # Mayer term
    mterm =  -T_F
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
