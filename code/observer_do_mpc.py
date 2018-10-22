import numpy as NP
from casadi import *
from casadi.tools import *
import setup_mhe
from scipy.linalg import expm
import pdb

class ekf:
    "A class for the definition of Extended Kalman Filter for state estimation"
    def __init__(self,model,param_dict):
        param_est = param_dict["parameter_estimation"]
        x = model.x
        u = model.u
        p = model.p
        tv_p = model.tv_p
        nx = x.size(1)
        np = p.size(1)
        f = model.rhs
        if param_est:
            f = vertcat(f,DM(NP.zeros(np)))
        h = model.y
        # h = substitute(model_observer.y,u,u*model_observer.ocp.u_scaling)
        # h = substitute(h,x,x*model_observer.ocp.x_scaling)/model_observer.ocp.y_scaling
        # variables
        self.x_hat = param_dict["x_init"]
        # tuning parameters
        self.Q = param_dict["Q"]
        self.R = param_dict["R"]
        self.P = param_dict["P"]
        # state transition and observation matrix
        if param_est:
            F = jacobian(f,vertcat(x,p))
            H = jacobian(h,vertcat(x,p))
        else:
            F = jacobian(f,x)
            H = jacobian(h,x)
        self.F = Function("F",[x,u,p,tv_p],[F])
        self.H = Function("H",[x,u,p,tv_p],[H])

class mhe:
    "A class for the definition of a Moving Horizon Estimator"
    def __init__(self,model,param_dict):
        # parameters
        self.n_horizon = param_dict["n_horizon"]
        self.n_robust = param_dict["n_robust"]
        # discretization
        self.state_discretization = param_dict["state_discretization"]
        self.poly_degree = param_dict["poly_degree"]
        self.collocation = param_dict["collocation"]
        self.n_fin_elem = param_dict["n_fin_elem"]
        # solver
        self.nlp_solver = param_dict["nlp_solver"]
        self.linear_solver = param_dict["linear_solver"]
        self.qp_solver = param_dict["qp_solver"]
        # Tuning parameters
        self.P_states = param_dict["P_states"]
        self.P_inputs = param_dict["P_inputs"]
        self.P_param = param_dict["P_param"]
        self.P_meas = param_dict["P_meas"]
        # counter for initialization
        self.count = 0
        self.x_hat = param_dict["x_init"]
        self.p_hat = param_dict["p_init"]
        # setup optimal control problem
        self.mhe_dict_out = setup_mhe.setup_mhe(model,self,param_dict)
        # Set options
        opts = {}
        opts["expand"] = True
        opts["ipopt.linear_solver"] = self.linear_solver
        opts["ipopt.max_iter"] = 500
        opts["ipopt.tol"] = 1e-6
        opts["ipopt.ma27_liw_init_factor"] =  100.0
        opts["ipopt.ma27_la_init_factor"] =  100.0
        opts["ipopt.ma27_meminc_factor"] =  2.0
        # Setup the solver
        solver = nlpsol("solver", self.nlp_solver, self.mhe_dict_out['nlp_fcn'], opts)
        arg = {}
        # Initial condition
        arg["x0"] = self.mhe_dict_out['vars_init']
        # Bounds on x
        arg["lbx"] = self.mhe_dict_out['vars_lb']
        arg["ubx"] = self.mhe_dict_out['vars_ub']
        # Bounds on g
        arg["lbg"] = self.mhe_dict_out['lbg']
        arg["ubg"] = self.mhe_dict_out['ubg']
        # NLP parameters
        nx = model.x.size(1)
        nu = model.u.size(1)
        np = model.p.size(1)
        ntv_p = model.tv_p.size(1)
        ny = model.y.size(1)
        nk = self.n_horizon
        parameters_setup_mhe = struct_symMX([entry("TV_P",shape=(ntv_p,nk)),
                                             entry("Y_MEAS",shape=(ny,nk)), entry("X_EST",shape=(nx,1)),
                                             entry("U_MEAS", shape=(nu,nk)), entry("P_EST", shape=(np,1))])
        param = parameters_setup_mhe(0)
        # First value of the nlp parameters
        param["TV_P"] = NP.ones([ntv_p,nk])
        param["Y_MEAS"] = NP.ones([ny,nk])
        param["X_EST"] = NP.zeros([nx,1])
        param["P_EST"] = NP.zeros([np,1])
        param["U_MEAS"] = NP.zeros([nu,nk])
        arg["p"] = param
        # Add new attributes to the observer class
        self.solver = solver
        self.arg = arg

def make_step_observer(conf):
    data = conf.mpc_data
    if conf.observer.method == 'state-feedback':
        rep = int(round(conf.optimizer.t_step/conf.simulator.t_step_simulator))
        for n_rep in range(rep):
            conf.make_step_simulator()
            conf.store_mpc_data()
        conf.observer.observed_states = conf.simulator.xf_sim

    elif conf.observer.method == 'EKF':
        # preprocess data and derive info
        nx = conf.model.x.size(1)
        np = conf.model.p.size(1)
        if conf.observer.param_est:
            nxp = nx+np
        else:
            nxp = nx
        rep_est = int(round(conf.optimizer.t_step/conf.observer.t_step))
        rep_sim = int(round(conf.observer.t_step/conf.simulator.t_step_simulator))
        P = conf.observer.ekf.P
        Q = conf.observer.ekf.Q
        R = conf.observer.ekf.R
        u_mpc = conf.optimizer.u_mpc*conf.model.ocp.u_scaling
        p_real = conf.simulator.p_real_batch
        tv_p_real = conf.simulator.tv_p_real_now(conf.simulator.t0_sim)
        for est in range(rep_est):
            for sim in range(rep_sim):
                conf.make_step_simulator()
            make_measurement(conf)
            xk = NP.reshape(conf.observer.ekf.x_hat,(-1,1))
            zk = NP.reshape(conf.observer.measurement,(-1,1))

            # Predict states
            for sim in range(rep_sim):
                if conf.observer.param_est:
                    xk[:nx,0]  = NP.squeeze((conf.simulator.simulator(x0 = xk[:nx,:], p = vertcat(u_mpc,xk[nx:,0],tv_p_real)))['xf'])
                else:
                    xk[:nx,0]  = NP.squeeze((conf.simulator.simulator(x0 = xk[:nx,:], p = vertcat(u_mpc,p_real,tv_p_real)))['xf'])
            # Predict covariance
            if conf.observer.param_est:
                H = conf.observer.ekf.H(xk[:nx],u_mpc,xk[nx:],tv_p_real)
                F = conf.observer.ekf.F(xk[:nx],u_mpc,xk[nx:],tv_p_real)
            else:
                H = conf.observer.ekf.H(xk[:nx],u_mpc,p_real,tv_p_real)
                F = conf.observer.ekf.F(xk[:nx],u_mpc,p_real,tv_p_real)
            F = expm(conf.observer.t_step*NP.atleast_2d(F))
            P = mtimes(mtimes(F,P),F.T)+Q

            # innovation
            S = inv(mtimes(H,mtimes(P,H.T))+R)

            # compute Kalman gain
            K = mtimes(mtimes(P,H.T),S)

            # residual
            if conf.observer.param_est:
                yk = zk - conf.observer.meas_fcn(xk[:nx],u_mpc,xk[nx:],tv_p_real)
            else:
                yk = zk - conf.observer.meas_fcn(xk,u_mpc,p_real,tv_p_real)

            # update state estimate
            xk = xk + mtimes(K,yk)

            #update covariance estimate
            conf.observer.ekf.x_hat = NP.squeeze(xk)
        if conf.observer.open_loop:
            conf.observer.observed_states = conf.simulator.xf_sim
        else:
            conf.observer.observed_states = conf.observer.x_hat[:nx]/conf.model.ocp.x_scaling
        conf.store_est_data()

    elif conf.observer.method == 'MHE':
        data = conf.mpc_data
        nk = conf.observer.mhe.n_horizon
        count = conf.observer.mhe.count
        rep_est = int(round(conf.optimizer.t_step/conf.observer.t_step))
        rep_sim = int(round(conf.observer.t_step/conf.simulator.t_step_simulator))
        for n_est in range(rep_est):
            for sim in range(rep_sim):
                conf.make_step_simulator()
            make_measurement(conf)
            if conf.observer.mhe.count >= conf.observer.mhe.n_horizon:
                X_offset = conf.observer.mhe.mhe_dict_out['X_offset']
                U_offset = conf.observer.mhe.mhe_dict_out['U_offset']
                nx = conf.model.x.size(1)
                np = conf.model.p.size(1)
                arg = conf.observer.mhe.arg
                # update parameters of optimizer
                param = arg['p']
                param["Y_MEAS"] = data.y_meas[count-nk:count,:].T
                param["X_EST"] = NP.reshape(data.est_states[count-nk-1,:],(-1,1))
                param["U_MEAS"] = data.u_meas[count-nk:count,:].T
                arg['p'] = param
                # optimization
                result = conf.observer.mhe.solver(x0=arg['x0'], lbx=arg['lbx'], ubx=arg['ubx'], lbg=arg['lbg'], ubg=arg['ubg'], p = arg['p'])
                conf.observer.optimal_solution = result['x']
                conf.observer.mhe.x_hat = NP.squeeze(conf.observer.optimal_solution[X_offset[-1][0]:X_offset[-1][0]+nx])
                if conf.observer.param_est:
                    conf.observer.mhe.p_hat = NP.squeeze(conf.observer.optimal_solution[:np])
                if conf.observer.open_loop:
                    conf.observer.observed_states = conf.simulator.xf_sim
                else:
                    conf.observer.observed_states = conf.observer.mhe.x_hat
            else:
                # open loop estimation (simulation) for initialization
                p_est = conf.observer.mhe.p_hat
                u_mpc = conf.optimizer.u_mpc
                tv_p_real = conf.simulator.tv_p_real_now(conf.simulator.t0_sim)
                for sim in range(rep_sim):
                    conf.observer.mhe.x_hat  = NP.squeeze((conf.simulator.simulator(x0 =conf.observer.mhe.x_hat, p = vertcat(u_mpc,p_est,tv_p_real)))['xf'])
                if conf.observer.open_loop:
                    conf.observer.observed_states = conf.simulator.xf_sim
                else:
                    conf.observer.observed_states = conf.observer.mhe.x_hat
            conf.store_est_data()


def make_measurement(conf):
    # preprocess data
    x = conf.simulator.xf_sim
    u = conf.optimizer.u_mpc
    p = conf.simulator.p_real_batch
    tv_p = conf.simulator.tv_p_real_now(conf.simulator.t0_sim)
    conf.observer.measurement = conf.observer.meas_fcn(x,u,p,tv_p)
    # add noise
    conf.observer.measurement += NP.random.normal(0,conf.observer.mag)
