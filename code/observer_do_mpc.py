import numpy as NP
from casadi import *
from casadi.tools import *
import setup_mhe
import data_do_mpc
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
        rhs_fun = Function("rhs_fun",[x,u,p],[model.rhs])
        ems = 50
        delta_t = 0.05
        if param_est:
            # if not (self.optimizer.state_discretization == 'discrete-time'):
            #     f = vertcat(f,DM(NP.zeros(np)))
            # else: # NOTE: somehow add handling of discrete systems
            f = vertcat(f,DM(NP.ones(np)))
            # f = vertcat(f,DM(NP.ones(2)))
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
            # F = jacobian(f,vertcat(x,p[:2]))
            # H = jacobian(h,vertcat(x,p[:2]))
        else:
            F = jacobian(f,x)
            H = jacobian(h,x)
        self.F = Function("F",[x,u,p,tv_p],[F])
        self.H = Function("H",[x,u,p,tv_p],[H])
        for i in range(ems):
            if i == 0:
                x_pred_new = x + delta_t/ems * rhs_fun(x,u,p)
            else:
                x_pred_new = x_pred_old + delta_t/ems * rhs_fun(x_pred_old,u,p)
            x_pred_old = x_pred_new
        self.x_pred = Function("x_pred",[x,u,p],[x_pred_new])

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
            # conf.store_mpc_data()
        conf.observer.observed_states = conf.simulator.xf_sim

    elif conf.observer.method == 'EKF':
        # preprocess data and derive info
        nx = conf.model.x.size(1)
        np = conf.model.p.size(1)
        # np = 2
        if conf.observer.param_est:
            nxp = nx+np
        else:
            nxp = nx
        rep_est = int(round(conf.optimizer.t_step/conf.observer.t_step))
        rep_sim = int(round(conf.observer.t_step/conf.simulator.t_step_simulator))
        P = NP.copy(conf.observer.ekf.P)
        Q = conf.observer.ekf.Q
        R = conf.observer.ekf.R
        u_mpc = conf.optimizer.u_mpc*conf.model.ocp.u_scaling
        # p_real = conf.simulator.p_real_batch
        p_real = NP.array([10.0]) # NOTE: hard-coded
        tv_p_real = conf.simulator.tv_p_real_now(conf.simulator.t0_sim)
        for est in range(rep_est):
            for sim in range(rep_sim):
                conf.make_step_simulator()
            make_measurement(conf)
            xk = NP.copy(NP.reshape(conf.observer.ekf.x_hat,(-1,1)))
            zk = NP.copy(NP.reshape(conf.observer.measurement,(-1,1)))

            # Predict states
            # for sim in range(rep_sim):
            #     if conf.observer.param_est:
            #         xk[:nx,0]  = NP.squeeze((conf.simulator.simulator(x0 = xk[:nx,:], p = vertcat(u_mpc,xk[nx:,0],tv_p_real)))['xf'])
            #     else:
            #         xk[:nx,0]  = NP.squeeze((conf.simulator.simulator(x0 = xk[:nx,:], p = vertcat(u_mpc,p_real,tv_p_real)))['xf'])
            # NOTE: for realistic results, use state prediction without integration
            xk[:nx,0] = NP.squeeze(conf.observer.ekf.x_pred(xk[:nx,:],u_mpc,xk[nx:,0]))
            # xk[:nx,0] = NP.squeeze(conf.observer.ekf.x_pred(xk[:nx,:],u_mpc,NP.hstack([xk[nx:nx+2,0],0.0])))

            # Predict covariance
            if conf.observer.param_est:
                H = conf.observer.ekf.H(xk[:nx],u_mpc,xk[nx:],tv_p_real)
                F = conf.observer.ekf.F(conf.observer.ekf.x_hat[:nx],u_mpc,xk[nx:],tv_p_real)
                # H = conf.observer.ekf.H(xk[:nx],u_mpc,NP.hstack([xk[nx:nx+2,0],0.0]),tv_p_real)
                # F = conf.observer.ekf.F(conf.observer.ekf.x_hat[:nx],u_mpc,NP.hstack([xk[nx:nx+2,0],0.0]),tv_p_real)
            else:
                H = conf.observer.ekf.H(xk[:nx],u_mpc,p_real,tv_p_real)
                F = conf.observer.ekf.F(conf.observer.ekf.x_hat,u_mpc,p_real,tv_p_real)
            if not (conf.optimizer.state_discretization == 'discrete-time'):
                F = expm(conf.observer.t_step*NP.atleast_2d(F))
            P = mtimes(mtimes(F,P),F.T)+Q

            # innovation
            S = inv(mtimes(H,mtimes(P,H.T))+R)

            # compute Kalman gain
            K = mtimes(mtimes(P,H.T),S)

            # residual
            if conf.observer.param_est:
                yk = zk - conf.observer.meas_fcn(xk[:nx],u_mpc,xk[nx:],tv_p_real)
                # yk = zk - conf.observer.meas_fcn(xk[:nx],u_mpc,NP.hstack([xk[nx:nx+2,0],0.0]),tv_p_real)
            else:
                yk = zk - conf.observer.meas_fcn(xk,u_mpc,p_real,tv_p_real)

            # update state estimate
            xk = xk + mtimes(K,yk)

            #update covariance estimate
            conf.observer.ekf.x_hat = NP.squeeze(xk)
            conf.observer.ekf.P = mtimes((NP.diag(NP.ones(nxp)) - mtimes(K,H)),P)
            # conf.store_est_data()

        if conf.observer.open_loop:
            conf.observer.observed_states = conf.simulator.xf_sim
        else:
            conf.observer.observed_states = conf.observer.ekf.x_hat[:nx]/conf.model.ocp.x_scaling

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
            # pdb.set_trace()
            if conf.observer.mhe.count >= conf.observer.mhe.n_horizon:
                X_offset = conf.observer.mhe.mhe_dict_out['X_offset']
                U_offset = conf.observer.mhe.mhe_dict_out['U_offset']
                nx = conf.model.x.size(1)
                np = conf.model.p.size(1)
                arg = conf.observer.mhe.arg
                # update parameters of optimizer
                param = arg['p']
                param["Y_MEAS"] = data.y_meas[count-nk:count,:].T
                # last estimate at the beginning of the estimation window
                if conf.observer.mhe.count == conf.observer.mhe.n_horizon:
                    conf.observer.mhe.last_estimate = NP.reshape(data.est_states[count-nk+1,:],(-1,1))
                param["X_EST"] = conf.observer.mhe.last_estimate
                # param["X_EST"] = NP.reshape(data.mpc_states[count-nk,:],(-1,1))
                # pdb.set_trace()
                # param["U_MEAS"] = data.u_meas[count-nk:count,:].T
                param["U_MEAS"] = horzcat(data.u_meas[count-nk+1:count+1,:].T, conf.optimizer.u_mpc_meas)
                arg['p'] = param
                # optimization
                # "Hard" fix of the initial point of the mhe window
                # arg['lbx'][X_offset[0,0]:X_offset[0,0]+nx] = data.est_states[count-nk+1,:]
                # arg['ubx'][X_offset[0,0]:X_offset[0,0]+nx] = data.est_states[count-nk+1,:]
                # "Hard" fix of the initial point of the mhe window to the real state (unrealistic)
                # arg['lbx'][X_offset[0,0]:X_offset[0,0]+nx] = data.mpc_states[count-nk+1,:]
                # arg['ubx'][X_offset[0,0]:X_offset[0,0]+nx] = data.mpc_states[count-nk+1,:]

                # Choose good initial guess
                if count > conf.observer.mhe.n_horizon+1:
                    arg["x0"] = conf.observer.optimal_solution
                result = conf.observer.mhe.solver(x0=arg['x0'], lbx=arg['lbx'], ubx=arg['ubx'], lbg=arg['lbg'], ubg=arg['ubg'], p = arg['p'])
                conf.observer.optimal_solution = result['x']
                optimal_cost = data_do_mpc.opt_result(result)
                conf.observer.optimal_cost = optimal_cost.optimal_cost
                conf.observer.mhe.x_hat = NP.squeeze(conf.observer.optimal_solution[X_offset[-1][0]:X_offset[-1][0]+nx])
                # Store the estimated state at the last to last point of the window to be used in the next iteration
                conf.observer.mhe.last_estimate = NP.squeeze(conf.observer.optimal_solution[X_offset[1][0]:X_offset[1][0]+nx])
                # pdb.set_trace()
                if conf.observer.param_est:
                    conf.observer.mhe.p_hat = NP.squeeze(conf.observer.optimal_solution[:np])
                if conf.observer.open_loop:
                    conf.observer.observed_states = conf.simulator.xf_sim
                else:
                    conf.observer.observed_states = conf.observer.mhe.x_hat
            else:
                # open loop estimation (simulation) for initialization
                p_est = conf.observer.mhe.p_hat
                u_mpc = conf.optimizer.u_mpc_meas
                tv_p_real = conf.simulator.tv_p_real_now(conf.simulator.t0_sim)
                if count == 0:
                    conf.observer.mhe.x_hat = conf.simulator.x0_sim
                else:
                    conf.observer.mhe.x_hat = conf.simulator.xf_sim
                # for sim in range(rep_sim):
                #     if conf.optimizer.state_discretization == 'discrete-time':
                #         rhs_unscaled = substitute(conf.model.rhs, conf.model.x, conf.model.x * conf.model.ocp.x_scaling)/conf.model.ocp.x_scaling
                #         rhs_unscaled = substitute(rhs_unscaled, conf.model.u, conf.model.u * conf.model.ocp.u_scaling)
                #         rhs_fcn = Function('rhs_fcn',[conf.model.x,vertcat(conf.model.u,conf.model.p,conf.model.tv_p)],[rhs_unscaled])
                #         x_next = rhs_fcn(conf.observer.mhe.x_hat,vertcat(u_mpc,p_est,tv_p_real))
                #         conf.observer.mhe.x_hat = NP.squeeze(NP.array(x_next))
                #     else:
                #         conf.observer.mhe.x_hat  = NP.squeeze((conf.simulator.simulator(x0 =conf.observer.mhe.x_hat, p = vertcat(u_mpc,p_est,tv_p_real)))['xf'])
                if conf.observer.open_loop:
                    conf.observer.observed_states = conf.simulator.xf_sim
                else:
                    conf.observer.observed_states = conf.observer.mhe.x_hat
                conf.observer.optimal_cost = NP.array([[0.0]])
            conf.store_est_data()
            print("-------------------------")
            # print("Error in estimated states:", conf.simulator.xf_sim - conf.observer.mhe.x_hat)
            print("count", count)


def make_measurement(conf):
    # preprocess data
    x = conf.simulator.x0_sim
    u = conf.optimizer.u_mpc
    p = conf.simulator.p_real_now(conf.simulator.t0_sim)
    tv_p = conf.simulator.tv_p_real_now(conf.simulator.t0_sim)
    # conf.observer.measurement = conf.observer.meas_fcn(x,u,p,tv_p)
    # add noise
    # conf.observer.measurement += NP.random.normal(0,conf.observer.mag)
    # In this case the measurement is 0 all the time
    conf.observer.measurement = NP.array([0.0,0.0])
