import numpy as NP
from casadi import *
from casadi.tools import *
import osqp
import scipy.sparse as sparse
import pdb

class projector:

    def __init__(self,conf):

        # parameters
        L = 400.0
        self.use_osqp = True
        self.h_min = 100.0
        self.ems = 10 # euler mean steps
        self.delta_t = conf.optimizer.t_step
        self.nf = 2 # prediction horizon projection
        w_soft = 10

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
        ems_pred = 10.0
        for i in range(int(ems_pred)):
            if i == 0:
                x_pred_new = x_sym + self.delta_t/ems_pred * rhs_fun(x_sym,u_sym,p_sym)
            else:
                x_pred_new = x_pred_old + self.delta_t/ems_pred * rhs_fun(x_pred_old,u_sym,p_sym)
            x_pred_old = x_pred_new
        self.x_pred = Function("x_pred",[x_sym,u_sym,p_sym],[x_pred_new])

        ### Build optimization problem for projection
        # simplified system dynamics
        self.A_tilde = Function("A_tilde",[x_sym,u_sym,p_sym],[jacobian(x_eul_m,x_sym)])
        self.B_tilde = Function("B_tilde",[x_sym,u_sym,p_sym],[jacobian(x_eul_m,u_sym)])


        param_proj = struct_symSX(
            [entry("uk", shape=(nu)), entry("xk", shape=(nx)),
             entry("pk",shape=(np)), entry("xv", shape=(nx*self.nf,4))])
        UK = param_proj["uk"]
        XK = param_proj["xk"]
        PK = param_proj["pk"]
        XV = param_proj["xv"]

        self.param = param_proj
        x_sym_new = SX.sym("x_sym_new",3,1)
        con_lin = self.con_fun(x_sym) + mtimes(jacobian(con,x_sym),x_sym_new - x_sym)
        self.con_lin_fun = Function("con_lin_fun",[x_sym_new,x_sym],[con_lin])

        # optimization variable
        u_hat = SX.sym("u_hat",nu,1)
        eps = SX.sym("eps",4,1)
        eps = SX.sym("eps",4*self.nf,1)
        x_old = SX.sym("x_old",3,1)
        for i in range(1):
            if i == 0:
                x_mean = x_old + self.delta_t/1.0 * rhs_fun(x_old,u_sym,p_sym) + mtimes(self.A_tilde(x_sym,u_sym,p_sym),x_old-x_sym)+ mtimes(self.B_tilde(x_sym,u_sym,p_sym),u_hat-u_sym)
            else:
                x_mean = x_sym + self.delta_t/1.0 * rhs_fun(x_sym,u_sym,p_sym) + mtimes(self.A_tilde(x_sym,u_sym,p_sym),x_mean-x_sym) + mtimes(self.B_tilde(x_sym,u_sym,p_sym),u_hat-u_sym)
        self.x_pred_lin = Function("x_pred_lin",[x_sym,x_old,u_sym,p_sym,u_hat],[x_mean])

        # prediction
        # x_pred_1 = self.x_pred(XK,UK,NP.array([4.0,7.0]))
        # x_pred_2 = self.x_pred(XK,UK,NP.array([4.0,13.0]))
        # x_pred_3 = self.x_pred(XK,UK,NP.array([6.0,7.0]))
        # x_pred_4 = self.x_pred(XK,UK,NP.array([6.0,13.0]))

        # linear case
        # x_new_1 = self.x_pred_lin(x_pred_1,XK,UK,NP.array([4.0,7.0]),u_hat)
        # x_new_2 = self.x_pred_lin(x_pred_2,XK,UK,NP.array([4.0,13.0]),u_hat)
        # x_new_3 = self.x_pred_lin(x_pred_3,XK,UK,NP.array([6.0,7.0]),u_hat)
        # x_new_4 = self.x_pred_lin(x_pred_4,XK,UK,NP.array([6.0,13.0]),u_hat)

        # nonlinear case
        # x_new_1 = self.x_pred(XK,u_hat,PK)
        x_new_1 = self.x_pred(XK,u_hat,NP.array([3.0,5.0]))
        x_new_2 = self.x_pred(XK,u_hat,NP.array([3.0,15.0]))
        x_new_3 = self.x_pred(XK,u_hat,NP.array([7.0,5.0]))
        x_new_4 = self.x_pred(XK,u_hat,NP.array([7.0,15.0]))

        # x_new_5 = self.x_pred(XK,u_hat,NP.array([5.0,7.0]))
        # x_new_6 = self.x_pred(XK,u_hat,NP.array([5.0,13.0]))
        # x_new_7 = self.x_pred(XK,u_hat,NP.array([6.0,10.0]))
        # x_new_8 = self.x_pred(XK,u_hat,NP.array([4.0,10.0]))
        # x_new_9 = self.x_pred(XK,u_hat,NP.array([5.0,10.0]))

        # solver for qp
        self.solver_osqp = osqp.OSQP()
        self.H = Function("H",[x_sym],[jacobian(con,x_sym)])
        self.P = sparse.csc_matrix(NP.diag(NP.array([1.0,w_soft,w_soft,w_soft,w_soft])))
        self.q = NP.array([0.0,0.0,0.0,0.0,0.0])

        # nonlinear solver
        # objective
        J = (UK - u_hat)**2 + w_soft*sum1(eps**1)

        # constraints
        # nonlinear
        h1 = self.con_fun(x_new_1)
        h2 = self.con_fun(x_new_2)
        h3 = self.con_fun(x_new_3)
        h4 = self.con_fun(x_new_4)
        # h5 = self.con_fun(x_new_5)
        # h6 = self.con_fun(x_new_6)
        # h7 = self.con_fun(x_new_7)
        # h8 = self.con_fun(x_new_8)
        # h9 = self.con_fun(x_new_9)

        #linear
        # h1 = self.con_lin_fun(x_new_1, x_pred_1)
        # h2 = self.con_lin_fun(x_new_2, x_pred_2)
        # h3 = self.con_lin_fun(x_new_3, x_pred_3)
        # h4 = self.con_lin_fun(x_new_4, x_pred_4)

        g = []
        g.append(x_new_1)
        g.append(x_new_2)
        g.append(x_new_3)
        g.append(x_new_4)
        # g.append(x_new_5)
        # g.append(x_new_6)
        # g.append(x_new_7)
        # g.append(x_new_8)
        # g.append(x_new_9)
        g.append(h1+eps[0,0])
        g.append(h2+eps[1,0])
        g.append(h3+eps[2,0])
        g.append(h4+eps[3,0])
        # g.append(h1+eps[0,0])
        # g.append(h2+eps[1,0])
        # g.append(h3+eps[2,0])
        # g.append(h4+eps[3,0])
        # g.append(h5+eps[4,0])
        # g.append(h6+eps[5,0])
        # g.append(h7+eps[6,0])
        # g.append(h8+eps[7,0])
        # g.append(h9+eps[8,0])
        for i in range(self.nf-1):
            g.append(self.con_fun(XV[(i+1)*nx:(i+2)*nx,0]+(XV[0:nx,0]-x_new_1))+eps[i*4+4,0])
            g.append(self.con_fun(XV[(i+1)*nx:(i+2)*nx,1]+(XV[0:nx,1]-x_new_2))+eps[i*4+5,0])
            g.append(self.con_fun(XV[(i+1)*nx:(i+2)*nx,2]+(XV[0:nx,2]-x_new_3))+eps[i*4+6,0])
            g.append(self.con_fun(XV[(i+1)*nx:(i+2)*nx,3]+(XV[0:nx,3]-x_new_4))+eps[i*4+7,0])
        g.append(eps)
        # g.append(eps[1,0])
        # g.append(eps[2,0])
        # g.append(eps[3,0])
        # g.append(eps[4,0])
        # g.append(eps[5,0])
        # g.append(eps[6,0])
        # g.append(eps[7,0])
        # g.append(eps[8,0])
        g = vertcat(*g)

        lbg = []
        lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        # lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        # lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        # lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        # lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        # lbg.append(NP.array([0.0,-0.5*pi,-1.0*pi]))
        for i in range((self.nf-1)*4):
            lbg.append(self.h_min)
        lbg.append(self.h_min)
        lbg.append(self.h_min)
        lbg.append(self.h_min)
        lbg.append(self.h_min)
        # lbg.append(self.h_min)
        # lbg.append(self.h_min)
        # lbg.append(self.h_min)
        # lbg.append(self.h_min)
        # lbg.append(self.h_min)
        lbg.append(NP.zeros([4*self.nf,1]))
        self.lbg = vertcat(*lbg)

        ubg = []
        ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        # ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        # ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        # ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        # ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        # ubg.append(NP.array([0.5*pi,0.5*pi,1.0*pi]))
        for i in range((self.nf-1)*4):
            ubg.append(inf)
        ubg.append(inf)
        ubg.append(inf)
        ubg.append(inf)
        ubg.append(inf)
        # ubg.append(inf)
        # ubg.append(inf)
        # ubg.append(inf)
        # ubg.append(inf)
        # ubg.append(inf)
        ubg.append(NP.ones([4*self.nf,1])*inf)
        self.ubg = vertcat(*ubg)

        # bliblablubb
        nlp_fcn = {'f': J, 'x': vertcat(u_hat,eps), 'p': param_proj, 'g': g}

        # solver options
        opts = {}
        opts["expand"] = True
        opts["ipopt.linear_solver"] = 'ma27'
        opts["ipopt.max_iter"] = 1000
        opts["ipopt.ma27_la_init_factor"] = 50.0
        opts["ipopt.ma27_liw_init_factor"] = 50.0
        opts["ipopt.ma27_meminc_factor"] = 10.0
        opts["ipopt.tol"] = 1e-6

        # setup solver
        self.solver_nlp = nlpsol("solver", 'ipopt', nlp_fcn, opts)

def make_step_projection(conf,solver):

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
    # xk = NP.copy(conf.observer.observed_states)
    xk = NP.copy(conf.simulator.x0_sim) # NOTE:
    uk = NP.copy(conf.optimizer.u_mpc)
    # pk = NP.copy(conf.observer.ekf.x_hat[nx:])
    pk = NP.copy(conf.simulator.p_real_batch)

    # parameter_settings
    p1 = NP.array([3.0,5.0])
    p2 = NP.array([3.0,15.0])
    p3 = NP.array([7.0,5.0])
    p4 = NP.array([7.0,15.0])
    tvp = NP.array([1.0,1.0])

    if solver == "osqp":

        xp1 = proj.x_pred(xk,uk,p1)
        xp2 = proj.x_pred(xk,uk,p2)
        xp3 = proj.x_pred(xk,uk,p3)
        xp4 = proj.x_pred(xk,uk,p4)

    elif solver == "ipopt":

        # result  = conf.simulator.simulator(x0 = xk, p = vertcat(uk,pk,tvp))
        # xp1 = NP.squeeze(result['xf'])

        u_lb = NP.array([-10.0])
        u_ub = NP.array([10.0])
        x_lb = NP.array([0.15, -1.25,-3.3])
        x_ub = NP.array([1.0, 1.3, 3.3])

        xv = NP.zeros([nx*proj.nf,4])

        for i in range(proj.nf):

            if i == 0:

                result  = conf.simulator.simulator(x0 = xk, p = vertcat(uk,p1,tvp))
                xv[:nx,0] = NP.squeeze(result['xf'])

                result  = conf.simulator.simulator(x0 = xk, p = vertcat(uk,p2,tvp))
                xv[:nx,1] = NP.squeeze(result['xf'])

                result  = conf.simulator.simulator(x0 = xk, p = vertcat(uk,p3,tvp))
                xv[:nx,2] = NP.squeeze(result['xf'])

                result  = conf.simulator.simulator(x0 = xk, p = vertcat(uk,p4,tvp))
                xv[:nx,3] = NP.squeeze(result['xf'])

            else:

                x_in_scaled = NP.atleast_2d(((xv[(i-1)*nx:i*nx,0]) - x_lb) / (x_ub - x_lb))
                u_opt_scaled = NP.squeeze(conf.loaded_model.predict(x_in_scaled))
                u_opt = u_opt_scaled * (u_ub - u_lb) + u_lb
                u_opt_lim = NP.maximum(NP.minimum(u_opt,u_ub),u_lb)

                result  = conf.simulator.simulator(x0 = xv[(i-1)*nx:i*nx,0], p = vertcat(u_opt_lim,p1,tvp))
                xv[i*nx:(i+1)*nx,0] = NP.squeeze(result['xf'])

                x_in_scaled = NP.atleast_2d(((xv[(i-1)*nx:i*nx,1]) - x_lb) / (x_ub - x_lb))
                u_opt_scaled = NP.squeeze(conf.loaded_model.predict(x_in_scaled))
                u_opt = u_opt_scaled * (u_ub - u_lb) + u_lb
                u_opt_lim = NP.maximum(NP.minimum(u_opt,u_ub),u_lb)

                result  = conf.simulator.simulator(x0 = xv[(i-1)*nx:i*nx,1], p = vertcat(u_opt_lim,p2,tvp))
                xv[i*nx:(i+1)*nx,1] = NP.squeeze(result['xf'])

                x_in_scaled = NP.atleast_2d(((xv[(i-1)*nx:i*nx,2]) - x_lb) / (x_ub - x_lb))
                u_opt_scaled = NP.squeeze(conf.loaded_model.predict(x_in_scaled))
                u_opt = u_opt_scaled * (u_ub - u_lb) + u_lb
                u_opt_lim = NP.maximum(NP.minimum(u_opt,u_ub),u_lb)

                result  = conf.simulator.simulator(x0 = xv[(i-1)*nx:i*nx,2], p = vertcat(u_opt_lim,p3,tvp))
                xv[i*nx:(i+1)*nx,2] = NP.squeeze(result['xf'])

                x_in_scaled = NP.atleast_2d(((xv[(i-1)*nx:i*nx,3]) - x_lb) / (x_ub - x_lb))
                u_opt_scaled = NP.squeeze(conf.loaded_model.predict(x_in_scaled))
                u_opt = u_opt_scaled * (u_ub - u_lb) + u_lb
                u_opt_lim = NP.maximum(NP.minimum(u_opt,u_ub),u_lb)

                result  = conf.simulator.simulator(x0 = xv[(i-1)*nx:i*nx,3], p = vertcat(u_opt_lim,p4,tvp))
                xv[i*nx:(i+1)*nx,3] = NP.squeeze(result['xf'])


    # compute predicted height
    viol = NP.zeros(proj.nf)
    for i in range(proj.nf):
        viol[i] = NP.array(proj.con_fun(xv[i*nx:(i+1)*nx,:]) < proj.h_min).any()


    if  viol.any():

    # if (con1<proj.h_min):

        if solver == "osqp":

            # compute A
            H_1 = proj.H(xp1)
            H_2 = proj.H(xp2)
            H_3 = proj.H(xp3)
            H_4 = proj.H(xp4)
            At_1 = proj.A_tilde(xp1,uk,p1)
            At_2 = proj.A_tilde(xp2,uk,p2)
            At_3 = proj.A_tilde(xp3,uk,p3)
            At_4 = proj.A_tilde(xp4,uk,p4)
            Bt_1 = proj.B_tilde(xp1,uk,p1)
            Bt_2 = proj.B_tilde(xp2,uk,p2)
            Bt_3 = proj.B_tilde(xp3,uk,p3)
            Bt_4 = proj.B_tilde(xp4,uk,p4)
            Bt_vert = NP.vstack([Bt_1,Bt_2,Bt_3,Bt_4])
            Ht_1 = mtimes(H_1,Bt_1)
            Ht_2 = mtimes(H_2,Bt_2)
            Ht_3 = mtimes(H_3,Bt_3)
            Ht_4 = mtimes(H_4,Bt_4)
            A_ur = NP.zeros([12,4])
            A_ul = NP.vstack([Bt_1,Bt_2,Bt_3,Bt_4])
            A_u = NP.hstack([A_ul,A_ur])
            A_mr = NP.diag(NP.ones(4))
            A_ml = NP.vstack([Ht_1,Ht_2,Ht_3,Ht_4])
            A_m = NP.hstack([A_ml,A_mr])
            A_l = NP.diag(NP.ones(5))
            # A = sparse.csc_matrix(NP.vstack([A_u,A_m,A_l]))
            A = sparse.csc_matrix(NP.vstack([A_m,A_l]))

            # update q
            proj.q[0] = -uk

            # compute bounds
            # l = NP.reshape(NP.array([0.0,-0.5*pi,-1.0*pi,0.0,-0.5*pi,-1.0*pi,0.0,-0.5*pi,-1.0*pi,0.0,-0.5*pi,-1.0*pi,proj.h_min,proj.h_min,proj.h_min,proj.h_min,-10.0,0.0,0.0,0.0,0.0]),(-1,1))
            # u = NP.reshape(NP.array([0.5*pi,0.5*pi,1.0*pi,0.5*pi,0.5*pi,1.0*pi,0.5*pi,0.5*pi,1.0*pi,0.5*pi,0.5*pi,1.0*pi,1e10,1e10,1e10,1e10,10.0,1e10,1e10,1e10,1e10]),(-1,1))
            x_lb = NP.array([0.0,-0.5*pi,-1.0*pi])
            x_ub = NP.array([0.5*pi,0.5*pi,1.0*pi])
            #
            # x_lb_new = NP.array([-0.5*pi,-1.0*pi,-1.5*pi])
            # x_ub_new = NP.array([1.0*pi,1.0*pi,1.5*pi])
            #
            # l = NP.reshape(NP.hstack([x_lb_new,x_lb_new,x_lb_new,x_lb_new,proj.h_min,proj.h_min,proj.h_min,proj.h_min,-10.0,0.0,0.0,0.0,0.0]),(-1,1))
            # u = NP.reshape(NP.hstack([x_ub_new,x_ub_new,x_ub_new,x_ub_new,1e10,1e10,1e10,1e10,10.0,1e10,1e10,1e10,1e10]),(-1,1))

            l = NP.reshape(NP.array([proj.h_min,proj.h_min,proj.h_min,proj.h_min,-10.0,0.0,0.0,0.0,0.0]),(-1,1))
            u = NP.reshape(NP.array([1e10,1e10,1e10,1e10,10.0,1e10,1e10,1e10,1e10]),(-1,1))

            # Setting 1
            xk = NP.reshape(xk,(-1,1))
            dif = xk - xp1
            HAxp = mtimes(H_1,At_1)
            abl = proj.rhs_fun(xk,uk,p1)
            # l[12] = l[12] - con1 + mtimes(H_1,xp1) - mtimes(H_1,xk) - proj.delta_t * mtimes(H_1,abl) - mtimes(HAxp,dif)
            l[0] = l[0] - con1 + mtimes(H_1,xp1) - mtimes(H_1,xk) - proj.delta_t * mtimes(H_1,abl) - mtimes(HAxp,dif)
            # l[:3] = l[:3] - xk - proj.delta_t * abl - mtimes(At_1,dif) + mtimes(Bt_1,uk)
            # u[:3] = u[:3] - xk - proj.delta_t * abl - mtimes(At_1,dif) + mtimes(Bt_1,uk)
#
            dif = xk - xp2
            HAxp = mtimes(H_2,At_2)
            abl = proj.rhs_fun(xk,uk,p2)
            # l[13] = l[13] - con2 + mtimes(H_2,xp2) - mtimes(H_2,xk) - proj.delta_t * mtimes(H_2,abl) - mtimes(HAxp,dif)
            l[1] = l[1] - con2 + mtimes(H_2,xp2) - mtimes(H_2,xk) - proj.delta_t * mtimes(H_2,abl) - mtimes(HAxp,dif)
            # l[3:6] = l[3:6] - xk - proj.delta_t * abl - mtimes(At_2,dif) + mtimes(Bt_2,uk)
            # l[3:6] = l[3:6] - xk - proj.delta_t * abl - mtimes(At_2,dif) + mtimes(Bt_2,uk)

            dif = xk - xp3
            HAxp = mtimes(H_3,At_3)
            abl = proj.rhs_fun(xk,uk,p3)
            # l[14] = l[14] - con3 + mtimes(H_3,xp3) - mtimes(H_3,xk) - proj.delta_t * mtimes(H_3,abl) - mtimes(HAxp,dif)
            l[2] = l[2] - con3 + mtimes(H_3,xp3) - mtimes(H_3,xk) - proj.delta_t * mtimes(H_3,abl) - mtimes(HAxp,dif)
            # l[6:9] = l[6:9] - xk - proj.delta_t * abl - mtimes(At_3,dif) + mtimes(Bt_3,uk)
            # l[6:9] = l[6:9] - xk - proj.delta_t * abl - mtimes(At_3,dif) + mtimes(Bt_3,uk)

            dif = xk - xp4
            HAxp = mtimes(H_4,At_4)
            abl = proj.rhs_fun(xk,uk,p4)
            # l[15] = l[15] - con4 + mtimes(H_4,xp4) - mtimes(H_4,xk) - proj.delta_t * mtimes(H_4,abl) - mtimes(HAxp,dif)
            l[3] = l[3] - con4 + mtimes(H_4,xp4) - mtimes(H_4,xk) - proj.delta_t * mtimes(H_4,abl) - mtimes(HAxp,dif)
            # l[9:12] = l[9:12] - xk - proj.delta_t * abl - mtimes(At_4,dif) + mtimes(Bt_4,uk)
            # l[9:12] = l[9:12] - xk - proj.delta_t * abl - mtimes(At_4,dif) + mtimes(Bt_4,uk)

            # setup solver
            proj.solver_osqp.setup(proj.P, proj.q, A, NP.squeeze(l), NP.squeeze(u))

            # solve problem
            res = proj.solver_osqp.solve()
            conf.optimizer.u_mpc = res.x[0]

            # check if violation
            xp1 = proj.x_pred(xk,res.x[0],p1)
            xp2 = proj.x_pred(xk,res.x[0],p2)
            xp3 = proj.x_pred(xk,res.x[0],p3)
            xp4 = proj.x_pred(xk,res.x[0],p4)
            xp_vert = NP.vstack([xp1,xp2,xp3,xp4])
            l_vert = NP.reshape(NP.hstack([x_lb,x_lb,x_lb,x_lb]),(-1,1))
            u_vert = NP.reshape(NP.hstack([x_ub,x_ub,x_ub,x_ub]),(-1,1))
            v_low = (xp_vert < l_vert).any()
            v_up = (xp_vert > u_vert).any()
            if v_low or v_up:
                pdb.set_trace()

        elif solver == "ipopt":

            param_k = proj.param(0)
            param_k["uk"] = uk
            param_k["xk"] = xk
            param_k["pk"] = pk
            # xv =
            param_k["xv"] = xv
            result = proj.solver_nlp(x0=vertcat(uk,NP.zeros([4*proj.nf,1])), lbx=-10.0, ubx=10.0, lbg=proj.lbg, ubg=proj.ubg, p=param_k)
            u_opt = result["x"]
            conf.optimizer.u_mpc = NP.reshape(u_opt[0],(1,-1))