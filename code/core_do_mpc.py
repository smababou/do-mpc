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

import setup_nlp
from casadi import *
from casadi.tools import *
import data_do_mpc
import numpy as NP
import pdb
from threading import Thread
from datetime import datetime, timedelta


class ocp:
    """ A class that contains a full description of the optimal control problem and will be used in the model class. This is dependent on a specific element of a model class"""
    def __init__(self, param_dict, *opt):
        # Initial state and initial input
        self.x0 = param_dict["x0"]
        self.u0 = param_dict["u0"]
        # Bounds for the states
        self.x_lb = param_dict["x_lb"]
        self.x_ub = param_dict["x_ub"]
        # Bounds for the inputs
        self.u_lb = param_dict["u_lb"]
        self.u_ub = param_dict["u_ub"]
        # Scaling factors
        self.x_scaling = param_dict["x_scaling"]
        self.u_scaling = param_dict["u_scaling"]
        # Symbolic nonlinear constraints
        self.cons = param_dict["cons"]
        # Upper bounds (no lower bounds for nonlinear constraints)
        self.cons_ub = param_dict["cons_ub"]
        # Terminal constraints
        self.cons_terminal = param_dict["cons_terminal"]
        self.cons_terminal_lb = param_dict["cons_terminal_lb"]
        self.cons_terminal_ub = param_dict["cons_terminal_ub"]
        # Flag for soft constraints
        self.soft_constraint = param_dict["soft_constraint"]
        # Penalty term and maximum violation of soft constraints
        self.penalty_term_cons = param_dict["penalty_term_cons"]
        self.maximum_violation = param_dict["maximum_violation"]
        # Lagrange term, Mayer term, and term for input variations
        self.lterm = param_dict["lterm"]
        self.mterm = param_dict["mterm"]
        self.rterm = param_dict["rterm"]

class model:
    """A class for the definition model equations and optimal control problem formulation"""
    def __init__(self, param_dict, *opt):
        # Assert for define length of param_dict
        required_dimension = 25
        if not (len(param_dict) == required_dimension):            raise Exception("Model / OCP information is incomplete. The number of elements in the dictionary is not correct")
        # Assign the main variables describing the model equations
        self.x = param_dict["x"]
        self.u = param_dict["u"]
        self.p = param_dict["p"]
        self.z = param_dict["z"]
        self.rhs = param_dict["rhs"] # Right hand side of the DAE equations
        self.tv_p = param_dict["tv_p"]
         # Assign the main variables that describe the OCP
        self.ocp = ocp(param_dict)

    @classmethod
    def user_model(cls, param_dict, *opt):
        " This is open for the implementation of a user-defined model class"
        dummy = 1
        return cls(dummy)

class simulator:
    """A class for the definition model equations and optimal control problem formulation"""
    def __init__(self, model_simulator, param_dict, *opt):
        # Assert for define length of param_dict
        required_dimension = 10
        if not (len(param_dict) == required_dimension): raise Exception("Simulator information is incomplete. The number of elements in the dictionary is not correct")
        # Unscale the states on the rhs
        rhs_unscaled = substitute(model_simulator.rhs, model_simulator.x, model_simulator.x * model_simulator.ocp.x_scaling)/model_simulator.ocp.x_scaling
        rhs_unscaled = substitute(rhs_unscaled, model_simulator.u, model_simulator.u * model_simulator.ocp.u_scaling)
        dae = {'x':model_simulator.x, 'p':vertcat(model_simulator.u,model_simulator.p, model_simulator.tv_p), 'ode':rhs_unscaled}
        opts = param_dict["integrator_opts"]
        #NOTE: Check the scaling factors (appear to be fine)
        simulator_do_mpc = integrator("simulator", param_dict["integration_tool"], dae,  opts)
        self.simulator = simulator_do_mpc
        self.plot_states = param_dict["plot_states"]
        self.plot_control = param_dict["plot_control"]
        self.plot_anim = param_dict["plot_anim"]
        self.export_to_matlab = param_dict["export_to_matlab"]
        self.export_name = param_dict["export_name"]
        self.p_real_now = param_dict["p_real_now"]
        self.tv_p_real_now = param_dict["tv_p_real_now"]
        self.t_step_simulator = param_dict["t_step_simulator"]
        self.t0_sim = 0
        self.tf_sim = param_dict["t_step_simulator"]
        # NOTE:  The same initial condition than for the optimizer is imposed
        self.x0_sim = model_simulator.ocp.x0 / model_simulator.ocp.x_scaling
        self.xf_sim = 0
        # This is an index to account for the MPC iteration. Starts at 1
        self.mpc_iteration = 1
    @classmethod
    def user_simulator(cls, param_dict, *opt):
        " This is open for the implementation of a user-defined simulator class"
        dummy = 1
        return cls(dummy)

    @classmethod
    def application(cls, param_dict, *opt):
        " This is open for the implementation of connection to a real plant"
        dummy = 1
        return cls(dummy)

class optimizer:
    '''This is a class that defines a do-mpc optimizer. The class uses a local model, which
    can be defined independetly from the other modules. The parameters '''
    def __init__(self, optimizer_model, param_dict, *opt):
        # Set the local model to be used by the model
        self.optimizer_model = optimizer_model
        # Assert for the required size of the parameters
        required_dimension = 16
        if not (len(param_dict) == required_dimension): raise Exception("The length of the parameter dictionary is not correct!")
        # Define optimizer parameters
        self.n_horizon = param_dict["n_horizon"]
        self.t_step = param_dict["t_step"]
        self.n_robust = param_dict["n_robust"]
        self.state_discretization = param_dict["state_discretization"]
        self.poly_degree = param_dict["poly_degree"]
        self.collocation = param_dict["collocation"]
        self.n_fin_elem = param_dict["n_fin_elem"]
        self.generate_code = param_dict["generate_code"]
        self.open_loop = param_dict["open_loop"]
        self.t_end = param_dict["t_end"]
        self.nlp_solver = param_dict["nlp_solver"]
        self.linear_solver = param_dict["linear_solver"]
        self.qp_solver = param_dict["qp_solver"]
        # Define model uncertain parameters
        self.uncertainty_values = param_dict["uncertainty_values"]
        # Define time varying optimizer parameters
        self.tv_p_values = param_dict["tv_p_values"]
        self.parameters_nlp = param_dict["parameters_nlp"]
        # Initialize empty methods for completion later
        self.solver = []
        self.arg = []
        self.nlp_dict_out = []
        self.opt_result_step = []
        self.u_mpc = optimizer_model.ocp.u0
    @classmethod
    def user_optimizer(cls, optimizer_model, param_dict, *opt):
        "This method is open for the impelmentation of a user defined optimizer"
        dummy = 1
        return cls(dummy)

class observer:
    """A class for the definition model equations and optimal control problem formulation"""
    def __init__(self, model_observer, param_dict, *opt):
        self.x = param_dict['x']
    @classmethod
    def user_observer(cls, param_dict, *opt):
        " This is open for the implementation of a user-defined estimator class"
        dummy = 1
        return cls(dummy)

class configuration:
    """ A class for the definition of a do-mpc configuration that
    contains a model, optimizer, observer and simulator module """
    def __init__(self, model, optimizer, observer, simulator, states, inputs, horizon_1, moving_obst_1):
        # The four modules
        self.model = model
        self.optimizer = optimizer
        self.observer = observer
        self.simulator = simulator
        self.states = states		# Robot#s States
        self.inputs = inputs		# Robot#s Inputs
        self.horizon_1 = horizon_1  # Pridiction Horizon Publisher
        self.moving_obst_1 = moving_obst_1  # Random Moving Obstacle
        self.set_m_obst_weight = False
        # The data structure
        self.mpc_data = data_do_mpc.mpc_data(self)

    def setup_solver(self):
        # Call setup_nlp to generate the NLP
        nlp_dict_out = setup_nlp.setup_nlp(self.model, self.optimizer)
        # Set options
        opts = {}
        opts["expand"] = True
        opts["ipopt.linear_solver"] = self.optimizer.linear_solver
        #NOTE: this could be passed as parameters of the optimizer class
        opts["ipopt.max_iter"] = 500
        opts["ipopt.tol"] = 1e-6
        # Turn On/Off Initial Printings
        #opts["verbose_init"] = False
        #opts["verbose"] = False
        opts["print_time"] = False
        opts["ipopt.print_level"] = 0
        # Setup the solver
        solver = nlpsol("solver", self.optimizer.nlp_solver, nlp_dict_out['nlp_fcn'], opts)
        arg = {}
        # Initial condition
        arg["x0"] = nlp_dict_out['vars_init']
        # Bounds on x
        arg["lbx"] = nlp_dict_out['vars_lb']
        arg["ubx"] = nlp_dict_out['vars_ub']
        # Bounds on g
        arg["lbg"] = nlp_dict_out['lbg']
        arg["ubg"] = nlp_dict_out['ubg']
        # NLP parameters
        nu = self.model.u.size(1)
        ntv_p = self.model.tv_p.size(1)
        nk = self.optimizer.n_horizon
        parameters_setup_nlp = struct_symMX([entry("uk_prev",shape=(nu)), entry("TV_P",shape=(ntv_p,nk))])
        param = parameters_setup_nlp(0)
        # First value of the nlp parameters
        param["uk_prev"] = self.model.ocp.u0
        param["TV_P"] = self.optimizer.tv_p_values[0]
        arg["p"] = param
        # Add new attributes to the optimizer class
        self.optimizer.solver = solver
        self.optimizer.arg = arg
        self.optimizer.nlp_dict_out = nlp_dict_out

    def make_step_optimizer(self):
        arg = self.optimizer.arg
        #print('#####################Intial guess' ,arg['x0'], '\n', self.optimizer.nlp_dict_out['X_offset'])
        result = self.optimizer.solver(x0=arg['x0'], lbx=arg['lbx'], ubx=arg['ubx'], lbg=arg['lbg'], ubg=arg['ubg'], p = arg['p']) ## change intial guess
        time_now = datetime.now()
        # Store the full solution
        self.optimizer.opt_result_step = data_do_mpc.opt_result(result)
        # Extract the optimal control input to be applied
        nu = len(self.optimizer.u_mpc)
        U_offset = self.optimizer.nlp_dict_out['U_offset']
        v_opt = self.optimizer.opt_result_step.optimal_solution
        self.optimizer.u_mpc = NP.resize(NP.array(v_opt[U_offset[0][0]:U_offset[0][0]+nu]),(nu))
        # Publish N_horizon Topic
        nk = self.optimizer.n_horizon
        t0 = 0.0
        tf = self.optimizer.t_step * nk
        tgrid = NP.linspace(t0,t0+tf,nk+1)
        date_list = NP.array([time_now + timedelta(seconds=x) for x in tgrid])
        self.horizon_1.horizon_msg.time_horizon.hr = NP.array([x.time().hour for x in date_list])
        self.horizon_1.horizon_msg.time_horizon.min = NP.array([x.time().minute for x in date_list])
        self.horizon_1.horizon_msg.time_horizon.sec = NP.array([x.time().second for x in date_list])
        self.horizon_1.horizon_msg.time_horizon.msec = NP.array([x.time().microsecond for x in date_list])
        self.horizon_1.horizon_msg.n_horizon = nk
        X_offset = self.optimizer.nlp_dict_out['X_offset']
        nx = self.model.x.size(1)
        pp_horz = NP.array([0,0,0])
        for v in range(X_offset.size):
           pp_horz = NP.vstack([pp_horz, NP.resize(NP.array(v_opt[X_offset[v][0]:X_offset[v][0]+nx]),(nx))])
        pp_horz = pp_horz[1:]
        self.horizon_1.horizon_msg.x = pp_horz[:,0]
        self.horizon_1.horizon_msg.y = pp_horz[:,1]
        self.horizon_1.horizon_msg.theta = pp_horz[:,1]
        self.horizon_1.horizon_pub()
        #self.optimizer.nlp_dict_out['p'][-5] = obst_eqn_robot

    def horizon_checker(self):
        # obst_eqn_robot = ((self.states('States').ret.x-3.5)**2 + (self.states('States').ret.y-3.5)**2 - (1)**2) / 1
        # p_real[2:7] = obst_eqn_robot
        # obst_eqn_robot_1 = ((self.model.x[0]-3.5)**2 + (self.model.x[1]-3.5)**2 - (1)**2) / 1
        # self.model.p[2:7] = obst_eqn_robot_1
        a = NP.array([self.horizon_1.horizon_msg.x, self.horizon_1.horizon_msg.y]).T
        b = a-1
        b[-1] = b[-1] + 0.7
        dist = NP.array([np.linalg.norm(a[x]-b[x]) for x in range(a.shape[0])])

        for idx, val in np.ndenumerate(dist):
            if val<0.5:
                print val
                print idx[0]
                print self.horizon_1.horizon_msg.time_horizon.min[idx[0]]
                print self.horizon_1.horizon_msg.time_horizon.sec[idx[0]]
                print self.horizon_1.horizon_msg.time_horizon.msec[idx[0]]
        xsfafsasf

    def make_step_observer(self):
        self.make_measurement()
        self.observer.observed_states = self.simulator.measurement # NOTE: this is a dummy observer

    def make_step_simulator(self):
        # Extract the necessary information for the simulation
        u_mpc = self.optimizer.u_mpc
        #print('############__Inputs__############: ', u_mpc)
        object_1 = Thread(target=self.inputs, args=(u_mpc[0],u_mpc[1]))
        object_1.start()
        object_1.join()
        #self.inputs(u_mpc[0], u_mpc[1])
        # Use the real parameters
        p_real = self.simulator.p_real_now(self.simulator.t0_sim)
        tv_p_real = self.simulator.tv_p_real_now(self.simulator.t0_sim)
        step_index = int(self.simulator.t0_sim / self.simulator.t_step_simulator)
        if step_index > 0:
            tv_p_real[0:2] = self.moving_obst_1.tv_p_values[0,:,0]
        else:
            pass

        if np.linalg.norm(NP.array([self.states('States').ret.x, self.states('States').ret.y]) - NP.array([8, 8])) < 0.5:
            tv_p_real[4] = 4
        else: 
            tv_p_real[4] = 0

        if np.linalg.norm(NP.array([self.states('States').ret.x, self.states('States').ret.y]) - NP.array([tv_p_real[0], tv_p_real[1]])) < 2:
            tv_p_real[5] = 40
            tv_p_real[2] = 2
            tv_p_real[3] = 2
            self.set_m_obst_weight = True
        else: 
            tv_p_real[5] = 0
            tv_p_real[2] = 10
            tv_p_real[3] = 10
            self.set_m_obst_weight = False
        # tv_p_real = NP.array([0.0]*self.model.tv_p.size(1))
        if self.optimizer.state_discretization == 'discrete-time':
            rhs_unscaled = substitute(self.model.rhs, self.model.x, self.model.x * self.model.ocp.x_scaling)/self.model.ocp.x_scaling
            rhs_unscaled = substitute(rhs_unscaled, self.model.u, self.model.u * self.model.ocp.u_scaling)
            rhs_fcn = Function('rhs_fcn',[self.model.x,vertcat(self.model.u,self.model.p,self.model.tv_p)],[rhs_unscaled])
            x_next = rhs_fcn(self.simulator.x0_sim,vertcat(u_mpc,p_real,tv_p_real))
            self.simulator.xf_sim = NP.squeeze(NP.array(x_next))
        else:
            result  = self.simulator.simulator(x0 = self.simulator.x0_sim, p = vertcat(u_mpc,p_real,tv_p_real))
            self.simulator.xf_sim = NP.squeeze(result['xf'])
        # Update the initial condition for the next iteration
        self.simulator.x0_sim = NP.array([self.states('States').ret.x, self.states('States').ret.y, self.states('States').ret.theta])
        # print(self.simulator.x0_sim)
        #self.simulator.x0_sim = self.simulator.xf_sim
        # Correction for sizes of arrays when dimension is 1
        if self.simulator.xf_sim.shape ==  ():
            self.simulator.xf_sim = NP.array([self.simulator.xf_sim])
        # Update the mpc iteration index and the time
        self.simulator.mpc_iteration = self.simulator.mpc_iteration + 1
        self.simulator.t0_sim = self.simulator.tf_sim
        self.simulator.tf_sim = self.simulator.tf_sim + self.simulator.t_step_simulator

    def make_measurement(self):
        # NOTE: Here implement the own measurement function (or load it)
        # This is a dummy measurement
        self.simulator.measurement = self.simulator.xf_sim

    def prepare_next_iter(self):
        observed_states = self.observer.observed_states
        X_offset = self.optimizer.nlp_dict_out['X_offset']
        nx = self.model.x.size(1)
        nu = self.model.u.size(1)
        ntv_p = self.model.tv_p.size(1)
        nk = self.optimizer.n_horizon
        parameters_setup_nlp = struct_symMX([entry("uk_prev",shape=(nu)), entry("TV_P",shape=(ntv_p,nk))])
        param = parameters_setup_nlp(0)
        # First value of the nlp parameters
        param["uk_prev"] = self.optimizer.u_mpc
        step_index = int(self.simulator.t0_sim / self.simulator.t_step_simulator)
        if step_index > 0:
                self.optimizer.tv_p_values[0,0:2,:] = self.moving_obst_1.tv_p_values[0]
        else:
            pass

        if np.linalg.norm(NP.array([self.states('States').ret.x, self.states('States').ret.y]) - NP.array([8, 8])) < 0.5:
            self.optimizer.tv_p_values[0,4,:] = NP.resize(NP.array([4.0]),(1,nk))
        else: 
            self.optimizer.tv_p_values[0,4,:] = NP.resize(NP.array([0.0]),(1,nk))

        if self.set_m_obst_weight == True:
            self.optimizer.tv_p_values[0,5,:] = NP.resize(NP.array([40.0]),(1,nk))
            self.optimizer.tv_p_values[0,2:4,:] = NP.resize(NP.array([2.0]),(2,nk))
        else:
            self.optimizer.tv_p_values[0,5,:] = NP.resize(NP.array([0.0]),(1,nk))
            self.optimizer.tv_p_values[0,2:4,:] = NP.resize(NP.array([10.0]),(2,nk))
        param["TV_P"] = self.optimizer.tv_p_values[0]
        # param["TV_P"] = NP.resize(NP.array([4.0]),(ntv_p,nk))
        # Enforce the observed states as initial point for next optimization

        self.optimizer.arg['lbx'][X_offset[0,0]:X_offset[0,0]+nx] = observed_states
        self.optimizer.arg['ubx'][X_offset[0,0]:X_offset[0,0]+nx] = observed_states
        self.optimizer.arg["x0"] = self.optimizer.opt_result_step.optimal_solution  ## change
        #print(self.optimizer.opt_result_step.optimal_solution[X_offset+2])
        # Pass as parameter the used control input
        self.optimizer.arg['p'] = param

    def store_mpc_data(self):
        mpc_iteration = self.simulator.mpc_iteration - 1 #Because already increased in the simulator
        data = self.mpc_data
        data.mpc_states = NP.append(data.mpc_states, [self.simulator.xf_sim], axis = 0)
        data.mpc_control = NP.append(data.mpc_control, [self.optimizer.u_mpc], axis = 0)
        #data.mpc_alg = NP.append(data.mpc_alg, [NP.zeros(NP.size(self.model.z))], axis = 0) # TODO: To be completed for DAEs
        data.mpc_time = NP.append(data.mpc_time, [[self.simulator.t0_sim]], axis = 0)
        data.mpc_cost = NP.append(data.mpc_cost, self.optimizer.opt_result_step.optimal_cost, axis = 0)
        #data.mpc_ref = NP.append(data.mpc_ref, [[0]], axis = 0) # TODO: To be completed
        stats = self.optimizer.solver.stats()
        data.mpc_cpu = NP.append(data.mpc_cpu, [[stats['t_wall_solver']]], axis = 0)
        data.mpc_parameters = NP.append(data.mpc_parameters, [self.simulator.p_real_now(self.simulator.t0_sim)], axis = 0)
