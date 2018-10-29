import numpy as NP
from casadi import *
from casadi.tools import *
import pdb

class projector:
    def __init__(self,conf):
        # parameters
        self.flaaaag = False
        L = 400.0
        ub = 100.0
        self.ems = 20 # euler mean steps
        self.delta_t = conf.optimizer.t_step

        # get symbolics
        rhs = conf.model.rhs
        con = -conf.model.ocp.cons[0]
        x_sym = conf.model.x
        u_sym = conf.model.u
        p_sym = conf.model.p

        # sizes
        nx = x_sym.shape[0]
        nu = u_sym.shape[0]
        np = p_sym.shape[0]

        # make functions
        self.con_fun = Function("con_fun",[x_sym],[con])
        rhs_fun = Function("rhs_fun",[x_sym,u_sym,p_sym],[rhs])
        self.rhs_fun = rhs_fun
        x_eul = x_sym + self.delta_t * rhs
        x_eul_m = x_sym + self.delta_t/self.ems * rhs
        x_eul_fun = Function("x_eul_fun",[x_sym,u_sym,p_sym],[x_eul])

        # prediction
        for i in range(self.ems):
            if i == 0:
                x_pred_new = x_sym + self.delta_t/self.ems * rhs_fun(x_sym,u_sym,p_sym)
            else:
                x_pred_new = x_pred_old + self.delta_t/self.ems * rhs_fun(x_pred_old,u_sym,p_sym)
            x_pred_old = x_pred_new
        self.x_pred = Function("x_pred",[x_sym,u_sym,p_sym],[x_pred_new])

        # solver options
        opts = {}
        opts["expand"] = True
        opts["ipopt.linear_solver"] = 'ma27'
        opts["ipopt.max_iter"] = 1000
        opts["ipopt.ma27_la_init_factor"] = 50.0
        opts["ipopt.ma27_liw_init_factor"] = 50.0
        opts["ipopt.ma27_meminc_factor"] = 10.0
        opts["ipopt.tol"] = 1e-6

        ### Build optimization problem for projection
        # simplified system dynamics
        self.A_tilde = Function("A_tilde",[x_sym,u_sym,p_sym],[jacobian(x_eul_m,x_sym)])
        self.B_tilde = Function("B_tilde",[x_sym,u_sym,p_sym],[jacobian(x_eul_m,u_sym)])


        param_proj = struct_symSX(
            [entry("uk", shape=(nu)), entry("xk", shape=(nx)),
             entry("pk", shape=(np))])
        UK = param_proj["uk"]
        XK = param_proj["xk"]
        PK = param_proj["pk"]

        self.param = param_proj
        x_sym_new = SX.sym("x_sym_new",3,1)
        con_lin = self.con_fun(x_sym) + mtimes(jacobian(con,x_sym),x_sym_new - x_sym)
        self.con_lin_fun = Function("con_lin_fun",[x_sym_new,x_sym],[con_lin])

        # optimization variable
        u_hat = SX.sym("u_hat",nu,1)
        eps = SX.sym("eps",4,1)

        # objective
        # x0 = x_eul_fun(XK,u_hat,PK)
        # A_tilde_k = A_tilde(XK,u_hat,PK)
        # B_tilde_k = B_tilde(XK,u_hat,PK)
        # x_new = x0 + mtimes(A_tilde_k,XK) + mtimes(B_tilde_k,u_hat-UK)
        # x_new = self.x_pred(XK,u_hat,PK)
        # x_new = x_eul_fun(XK,u_hat,PK)
        # x_new = self.x_pred(XK,u_hat,PK) # NOTE: iterative euler

        for i in range(self.ems):
            if i == 0:
                x_mean = x_sym + self.delta_t/self.ems * rhs_fun(x_sym,u_sym,p_sym) + mtimes(self.B_tilde(x_sym,u_sym,p_sym),u_hat-u_sym)
            else:
                x_mean = x_sym + self.delta_t/self.ems * rhs_fun(x_sym,u_sym,p_sym) + mtimes(self.A_tilde(x_sym,u_sym,p_sym),x_mean-x_sym) + mtimes(self.B_tilde(x_sym,u_sym,p_sym),u_hat-u_sym)
        self.x_pred_lin = Function("x_pred_lin",[x_sym,u_sym,p_sym,u_hat],[x_mean])

        x_new_1 = self.x_pred_lin(XK,UK,NP.array([4.0,7.0]),u_hat)
        x_new_2 = self.x_pred_lin(XK,UK,NP.array([4.0,13.0]),u_hat)
        x_new_3 = self.x_pred_lin(XK,UK,NP.array([6.0,7.0]),u_hat)
        x_new_4 = self.x_pred_lin(XK,UK,NP.array([6.0,13.0]),u_hat)

        x_pred_1 = self.x_pred_lin(XK,UK,NP.array([4.0,7.0]),UK)
        x_pred_2 = self.x_pred_lin(XK,UK,NP.array([4.0,13.0]),UK)
        x_pred_3 = self.x_pred_lin(XK,UK,NP.array([6.0,7.0]),UK)
        x_pred_4 = self.x_pred_lin(XK,UK,NP.array([6.0,13.0]),UK)

        # objective
        # J = 1e0*(UK - u_hat)**2 + 1e4*(100.0 - self.con_fun(x_new))**2
        w_soft = 1e3
        J = (UK - u_hat)**2 + w_soft*(eps[0,0])**2 + w_soft*(eps[1,0])**2 + w_soft*(eps[2,0])**2 + w_soft*(eps[3,0])**2

        # constraints
        # h1 = self.con_fun(x_new_1)
        # h2 = self.con_fun(x_new_2)
        # h3 = self.con_fun(x_new_3)
        # h4 = self.con_fun(x_new_4)

        h1 = self.con_lin_fun(x_new_1, x_pred_1)
        h2 = self.con_lin_fun(x_new_2, x_pred_2)
        h3 = self.con_lin_fun(x_new_3, x_pred_3)
        h4 = self.con_lin_fun(x_new_4, x_pred_4)

        g = []
        g.append(x_new_1)
        g.append(x_new_2)
        g.append(x_new_3)
        g.append(x_new_4)
        g.append(h1+eps[0,0])
        g.append(h2+eps[1,0])
        g.append(h3+eps[2,0])
        g.append(h4+eps[3,0])
        g.append(eps[0,0])
        g.append(eps[1,0])
        g.append(eps[2,0])
        g.append(eps[3,0])
        g = vertcat(*g)

        lbg = []
        lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        lbg.append(100.0)
        lbg.append(100.0)
        lbg.append(100.0)
        lbg.append(100.0)
        lbg.append(NP.ones([4,1])*0.0)
        self.lbg = vertcat(*lbg)

        ubg = []
        ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        ubg.append(inf)
        ubg.append(inf)
        ubg.append(inf)
        ubg.append(inf)
        ubg.append(NP.ones([4,1])*inf)
        self.ubg = vertcat(*ubg)

        # bliblablubb
        nlp_fcn = {'f': J, 'x': vertcat(u_hat,eps), 'p': param_proj, 'g': g}

        # setup solver
        self.solver = nlpsol("solver", 'ipopt', nlp_fcn, opts)

def make_step_projection(conf):

    # unwrap
    proj = conf.projector

    # get symbolics
    x_sym = conf.model.x
    u_sym = conf.model.u
    p_sym = conf.model.p

    # sizes
    nx = x_sym.shape[0]
    nu = u_sym.shape[0]
    np = p_sym.shape[0]

    # current values
    xk = NP.copy(conf.observer.observed_states)
    xk = NP.copy(conf.simulator.x0_sim)
    uk = NP.copy(conf.optimizer.u_mpc)
    pk = NP.copy(conf.simulator.p_real_batch)
    # pk = NP.copy(conf.observer.ekf.x_hat[nx:])

    # predict state
    xp1 = proj.x_pred(xk,uk,NP.array([4.0,7.0]))
    xp2 = proj.x_pred(xk,uk,NP.array([4.0,13.0]))
    xp3 = proj.x_pred(xk,uk,NP.array([6.0,7.0]))
    xp4 = proj.x_pred(xk,uk,NP.array([6.0,13.0]))

    con1 = proj.con_fun(xp1)
    con2 = proj.con_fun(xp2)
    con3 = proj.con_fun(xp3)
    con4 = proj.con_fun(xp4)

    if  (con1<100.0) or (con2<100.0) or (con3<100.0) or (con4<100.0):
        param_k = proj.param(0)
        param_k["uk"] = uk
        param_k["xk"] = xk
        param_k["pk"] = pk
        result = proj.solver(x0=vertcat(uk,NP.ones([4,1])), lbx=-10.0, ubx=10.0, lbg=proj.lbg, ubg=proj.ubg, p=param_k)
        u_opt = result["x"]
        # pdb.set_trace()
        # proj.flaaaag = True
        conf.optimizer.u_mpc = NP.reshape(u_opt[0],(1,-1))

        # conf.make_step_simulator()
        # print(conf.simulator.xf_sim)
        # print(proj.x_pred_lin(xk,uk,pk,u_opt))
        # print(proj.x_pred(xk,u_opt,pk))
        # pdb.set_trace()

    # # linearized system
    # x_eul = x_sym + delta_t * rhs
    # x_eul_fun = Function("x_eul_fun",[x_sym,u_sym,p_sym],[x_eul])
    # B_tilda = jacobian(x_eul,u_sym)
    # B_tilda_fun = Function("B_tilda",[x_sym,u_sym,p_sym],[B_tilda])
    # B_tilda = substitute(B_tilda,vertcat(x_sym,u_sym,p_sym),vertcat(xk,uk,pk))
    #
    # # Linearize system #NOTE: if robust -> several linear systems
    # dx = Function('dx',[vertcat(x_sym,u_sym,p_sym)],[rhs])#(rhs,vertcat(x_sym,u_sym,p_sym),vertcat(xk,uk,pk))
    # # Predict states
    #
    # # cons fun
    # con = -conf.model.ocp.cons[0]
    # con_fun = Function('con_fun',[x_sym],[con])
    # con_dx = jacobian(con,x_sym)
    #
    # if (con_fun(xp) <= -conf.model.ocp.cons_ub):
    #     pdb.set_trace()
    #     # linearize constraint xp
    #     u_hat = SX.sym("u_hat")
    #     du = mtimes(B_tilda,(u_hat-uk))
    #     h = con + mtimes(con_dx,du)
    #     h = substitute(h,p_sym,pk)
    #     h = substitute(h,x_sym,xk)
    #     dx_lin = jacobian(rhs,vertcat(x_sym,u_sym))
    #     dx_lin = substitute(dx_lin,p_sym,pk)
    #
    #     # lhs_p =
    #
    #     con_lin_fun = Function('con_lin_fun',[x_sym],[jacobian(-conf.model.ocp.cons[0],x_sym)])
    #     con_lin_xp = con_lin_fun(xp)
    #
    #     # variables and paramters
    #     u_hat = MX.sym("u_hat",nu,1)
    #

    #
    #
    #     xn = x_eul_fun(XK,u_hat,PK)
    #
    #     # objective
    #     J = (UK - u_hat)**2
    #


    #
    #     # get solution
    #     param_true = param_proj(0)
    #     param_true["uk"] = uk
    #     param_true["xk"] = xk
    #     param_true["pk"] = pk
    #     result = solver(x0=uk, lbx=-1000.0, ubx=1000.0, lbg=lbg, ubg=ubg, p=param_true)
    #     u_opt = result["x"][:nu]
