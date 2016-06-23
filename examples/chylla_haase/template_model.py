# 	 -*- coding: utf-8 -*-
#
#    This file is part of DO-MPC
#
#    DO-MPC: An environment for the easy, modular and efficient implementation of
#            robust nonlinear model predictive control
#
#    The MIT License (MIT)
#
#    Copyright (c) 2014-2015 Sergio Lucia, Alexandru Tatulea-Codrean, Sebastian Engell
#                            TU Dortmund. All rights reserved
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
#
from casadi import *
import numpy as NP
import core_do_mpc

def model():

    """
    --------------------------------------------------------------------------
    template_model: define the non-uncertain parameters
    --------------------------------------------------------------------------
    """
    E      = 29560.89
    R      = 8.314
    c0     = 5.2*10**-5
    c1     = 16.4
    c2     = 2.3
    c3     = 1.563
    a0     = 555.556
    d0     = 0.814     #Constant
    CpM    = 1.675      #Specific heat at Constant Pressure
    P      = 1.594	 #Jacket Perimeter
    k0     = 55.0
    d1     = -5.13      #Constant
    CpP    = 3.140      #Specific heat at Constant Pressure
    B1     = 0.193		 #Reactor Bottom area
    k1     = 1000.00
    proM   = 900.00       #Density
    CpW    = 4.187	 #Specific heat at Constant Pressure
    B2     = 0.167	 #Jacket Bottom area
    k2     = 0.4
    proP   = 1040.00      #Density
    CpC    = 4.187      #Specific heat at Constant Pressure
    Tau    = 40.2       #Heating/Cooling Time constant
    mCdot  = 0.9412
    deltaH = 70152.16   #Reaction Enthaply
    proW   = 1000.00		 #Density
    MWM    = 104.00
    UAloss = 0.00567567 #Heat loss Coefficient

    mC     = 21.455
    mW     = 42.750

    Tinlet = 294.26
    Tsteam = 449.82
    #Transport Delay in Jacket and recirculation loop
    teta1 = 22.8
    teta2 = 15.0
    teta3 = teta1+teta2

    pur		= 1.0
    hf 		= 1/0.704 #(inverse value)
    Tamb 	= 280.382
    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols

    pur   = SX.sym("pur")
    # Define the differential states as CasADi symbols

    mM 		= SX.sym("mM")              #Monomer Mass
    mP 		= SX.sym("mP")			    #Polymer Mass
    TR  		= SX.sym("TR")				#Reactor Temperature
    Tjout 		= SX.sym("Tjout")			#Output Temperature of the coolant
    Tjin		= SX.sym("Tjin")			#Input Temperature of the coolant

    # Define the algebraic states as CasADi symbols
    mM_lag2     = SX.sym("mM_lag2")
    mP_lag2     = SX.sym("mP_lag2")
    T_lag2      = SX.sym("T_lag2")

    Tjin_lag1   = SX.sym("Tjin_lag1")   #Lag in the input Temperature of the Coolant
    Tjout_lag2  = SX.sym("Tjout_lag2")	#Lag in the Output Temperature of the Coolant

    Tjin_lag2   = SX.sym("Tjin_lag2")
    Tjin_lag3   = SX.sym("Tjin_lag3")

    # Define the control inputs as CasADi symbols
    VO  	= SX.sym("VO")
    mMdot	= SX.sym("mMdot")

    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    # Define aditional equations
    exp1=2.71828182846

    ## ----Empirical relations for the polymerization rate rP------
    f = mP/(mM + mP + mW)						#Auxiliary Variable
    muy = c0*(exp1**(c1*f))*(10.0**(c2*(a0/TR-c3)))   #Batch Viscosity

    kcon = k0*exp1**(-E/R/TR)*(k1*muy)**k2			#First-order Kinetic constant
    rP = pur*kcon*mM

    ##-----Heat Reaction---------------------
    Qrea = deltaH*rP/MWM

    ##----Average Cooling Jacket Temperature----------
    Tj = (Tjin + Tjout)/2

    ##----Heat Transfer Area A-----------------------
    A = (mM/proM + mP/proP + mW/proW)*P/B1 + B2

    ##----- Overall Heat Transfer coefficient-------
    Twall = (TR + Tj)/2									     #Wall Temperature
    muywall = c0*exp1**(c1*f)*10**(c2*(a0/Twall-c3))   #WAll Viscosity
    h3 = d0*exp1**(d1*muywall)							 #Film heat Transfer Coefficient
    Uover = 1/(1/h3+1/hf)

    ##-------Kp------------------------------------
    Kp = (0.5*NP.tanh(3*(VO-50))+0.5)*(0.15*30**(VO/50-2)*(Tsteam - Tjin)) + (1-(0.5*NP.tanh(3*(VO-50))+0.5))*(0.8*30**(-VO/50)*(Tinlet - Tjin))

    Tj_lag2 = (Tjin_lag2+Tjout_lag2)/2
    Twall_lag2 = (T_lag2+Tj_lag2)/2
    f_lag2 = mP_lag2/(mM_lag2+mP_lag2+mW)
    muywall_lag2 = c0*exp1**(c1*f_lag2)*10**(c2*(a0/Twall_lag2-c3))
    h_lag2 = d0*exp1**(d1*muywall_lag2)
    U_lag2 = 1.0/(1.0/h_lag2+1/hf)


    A_lag2 = (mM_lag2/proM + mP_lag2/proP + mW/proW)*P/B1 + B2
    dTjout_lag2 = 1/(mC*CpC)*(mCdot*CpC*(Tjin_lag3-Tjout_lag2) + U_lag2*A_lag2*(T_lag2-Tj_lag2))

    # Define the differential equations
    dmM = mMdot - rP
    dmP = rP
    dT     = 1/(mM*CpM + mP*CpP + mW*CpW)*(mMdot*CpM*(Tamb-TR) + Uover*A*(Tj-TR) + UAloss*(Tamb-TR) + Qrea)
    dTjout = 1/(mC*CpC)*(mCdot*CpC*(Tjin_lag1-Tjout) + Uover*A*(TR-Tj))
    dTjin  = dTjout_lag2 + (Tjout_lag2-Tjin)/Tau + Kp/Tau

    # Define the algebraic equations
    dmM_lag2 = mM_lag2 -(mM - teta2*dmM)
    dmP_lag2 = mP_lag2 - (mP - teta2*dmP)
    dT_lag2  = T_lag2 - (TR - teta2*dT)
    dTjout_lag2 = Tjout_lag2 - (Tjout - teta2 * dTjout)
    dTjin_lag1 = Tjin_lag1 - (Tjin - teta1 * dTjin)
    dTjin_lag2 = Tjin_lag2 - (Tjin - teta2*dTjin)
    dTjin_lag3 = Tjin_lag3 - (Tjin - teta3*dTjin)

    # Concatenate differential states, algebraic states, control inputs and right-hand-sides

    _x = vertcat(mM,mP,TR,Tjout,Tjin)

    #_z = []
    _z = vertcat(mM_lag2,mP_lag2,T_lag2,Tjout_lag2,Tjin_lag1,Tjin_lag2,Tjin_lag3)

    _u = vertcat(VO,mMdot)

    _p = vertcat(pur)

    _xdot = vertcat(dmM,dmP,dT,dTjout,dTjin)

    #_zdot = vertcat([])
    _zdot = vertcat(dmM_lag2,dmP_lag2,dT_lag2,dTjout_lag2,dTjin_lag1,dTjin_lag2,dTjin_lag3)

    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    # Initial conditions for Differential States
    mM_init 	= 0.0
    mP_init 	= 11.227
    T_init 		= 305.382
    Tjout_init 	= 305.382
    Tjin_init  	= 305.382

    #Initial conditions for Algebraic States
    mM_lag2_init 	   =  0.0
    mP_lag2_init 	   =  11.227
    T_lag2_init  	   =  305.382
    Tjout_lag2_init =  305.382
    Tjin_lag1_init  =  305.382
    Tjin_lag2_init  =  305.382
    Tjin_lag3_init  =  305.382
    max_feed        =   31.752

    x0 = NP.array([mM_init,mP_init,T_init,Tjout_init,Tjin_init])
    z0 = NP.array([mM_lag2_init, mP_lag2_init, T_lag2_init, Tjout_lag2_init, Tjin_lag1_init, Tjin_lag2_init, Tjin_lag3_init])
    # Bounds on the states. Use "inf" for unconstrained states
    mM_lb    = 0.0;                             mM_ub    = inf;
    mP_lb    = 0.0;					  	        mP_ub    = inf
    T_lb     = 0.0;					            T_ub     = 355.382+0.6
    Tjout_lb = 273.15+5.0;			            Tjout_ub = 273.15+95.0
    Tjin_lb  = 0.0;						        Tjin_ub  = 273.15+95.0

    mM_lag2_lb	  = 0.0;						mM_lag2_ub    = inf
    mP_lag2_lb	  = 0.0;					    mP_lag2_ub    = inf
    T_lag2_lb 	  = 0.0;						T_lag2_ub     = inf
    Tjout_lag2_lb = 0.0;						Tjout_lag2_ub = inf
    Tjin_lag1_lb  = 0.0;						Tjin_lag1_ub  = inf
    Tjin_lag2_lb  = 0.0;						Tjin_lag2_ub  = inf
    Tjin_lag3_lb  = 0.0;						Tjin_lag3_ub  = inf

    x_lb = NP.array([mM_lb, mP_lb, T_lb, Tjout_lb, Tjin_lb])
    x_ub = NP.array([mM_ub, mP_ub, T_ub, Tjout_ub, Tjin_ub])
    z_lb = NP.array([mM_lag2_lb, mP_lag2_lb, T_lag2_lb, Tjout_lag2_lb, Tjin_lag1_lb, Tjin_lag2_lb, Tjin_lag3_lb])
    z_ub = NP.array([mM_lag2_ub, mP_lag2_ub, T_lag2_ub, Tjout_lag2_ub, Tjin_lag1_ub, Tjin_lag2_ub, Tjin_lag3_ub])
    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    VO_lb    = 0.0;		                        VO_ub = 100.00 ;		         VO_init = 100.0	;
    mMdot_lb = 0.0;		                        mMdot_ub = 0.0;	                 mMdot_init = 0.005;

    u_lb=NP.array([VO_lb,mMdot_lb])
    u_ub=NP.array([VO_ub,mMdot_ub])
    u0 = NP.array([VO_init,mMdot_init])

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = NP.array([1.0, 1.0, 1.0, 1.0, 1.0])
    z_scaling = NP.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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
    lterm =  0
    #lterm =  - C_b
    # Mayer term
    mterm =  0

    # Penalty term for the control movements
    rterm = NP.array([0.0, 0.0])



    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x':_x,'u': _u, 'rhs':_xdot,'p': _p, 'z':_z, 'aes': _zdot,'x0': x0, 'z0':z0, 'x_lb': x_lb,'x_ub': x_ub, 'z_lb': z_lb,'z_ub': z_ub, 'u0':u0, 'u_lb':u_lb, 'u_ub':u_ub, 'x_scaling':x_scaling, 'z_scaling':z_scaling, 'u_scaling':u_scaling, 'cons':cons,
    "cons_ub": cons_ub, 'cons_terminal':cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'cons_terminal_ub':cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm,'lterm':lterm, 'rterm':rterm}
    
    model = core_do_mpc.model(model_dict)

    return model
