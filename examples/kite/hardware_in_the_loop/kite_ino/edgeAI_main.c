#include <stddef.h>
#include <string.h>
#include "edgeAI_main.h"
#include "edgeAI_const.h"
#include "rhs.h"
#include "F_fun.h"

static void make_scale(
	struct edgeAI_ctl *ctl
	);

static void make_unscale(
	struct edgeAI_ctl *ctl
	);

void make_dnn_step(struct edgeAI_ctl *ctl)
{
	real_t x_layer_1[10], x_layer_2[10], x_layer_3[10], x_layer_4[10], x_layer_5[10], x_layer_6[10], x_layer_7[1];

	make_scale(ctl);

	mtx_times_vec_dense(x_layer_1,ctl->dnn->kernel_1,ctl->in_scaled,10,3);
	mtx_add(x_layer_1,x_layer_1,ctl->dnn->bias_1,10,1);
	mtx_tanh(x_layer_1,10);

	mtx_times_vec_dense(x_layer_2,ctl->dnn->kernel_2,x_layer_1,10,10);
	mtx_add(x_layer_2,x_layer_2,ctl->dnn->bias_2,10,1);
	mtx_tanh(x_layer_2,10);

	mtx_times_vec_dense(x_layer_3,ctl->dnn->kernel_3,x_layer_2,10,10);
	mtx_add(x_layer_3,x_layer_3,ctl->dnn->bias_3,10,1);
	mtx_tanh(x_layer_3,10);

	mtx_times_vec_dense(x_layer_4,ctl->dnn->kernel_4,x_layer_3,10,10);
	mtx_add(x_layer_4,x_layer_4,ctl->dnn->bias_4,10,1);
	mtx_tanh(x_layer_4,10);

	mtx_times_vec_dense(x_layer_5,ctl->dnn->kernel_5,x_layer_4,10,10);
	mtx_add(x_layer_5,x_layer_5,ctl->dnn->bias_5,10,1);
	mtx_tanh(x_layer_5,10);

	mtx_times_vec_dense(x_layer_6,ctl->dnn->kernel_6,x_layer_5,10,10);
	mtx_add(x_layer_6,x_layer_6,ctl->dnn->bias_6,10,1);
	mtx_tanh(x_layer_6,10);

	mtx_times_vec_dense(x_layer_7,ctl->dnn->kernel_7,x_layer_6,1,10);
	mtx_add(ctl->out_scaled,x_layer_7,ctl->dnn->bias_7,1,1);
	make_unscale(ctl);


	return;
}

void make_ekf_step(struct edgeAI_ctl *ctl)
{

	// sizes
	const uint32_t nx = 5, ny = 2;
	uint32_t i, j, k;

	for (k=0; k<3; k++) {

		// predict states
		uint32_t n_in = x_pred_n_in();
		uint32_t n_out = x_pred_n_out();

		uint32_t sz_arg=n_in, sz_res=n_out, sz_iw=0, sz_w=0;
		if (x_pred_work(&sz_arg, &sz_res, &sz_iw, &sz_w)) return 1;

		real_t *arg[sz_arg];
	    real_t *res[sz_res];
	    uint32_t iw[sz_iw];
	    real_t w[sz_w];

		real_t xk[3];
		xk[0] = ctl->ekf->x_hat[0];
		xk[1] = ctl->ekf->x_hat[1];
		xk[2] = ctl->ekf->x_hat[2];
	    real_t p[2];
		p[0] = ctl->ekf->x_hat[3];
		p[1] = ctl->ekf->x_hat[4];
	    real_t u[1];
		u[0] = ctl->out[0];

		x_pred_incref();

		arg[0] = xk;
	    arg[1] = u;
	    arg[2] = p;
	    res[0] = xk;

	    if (x_pred(arg, res, iw, w, 0)) return 1;

		x_pred_decref();

		// state transition
		n_in = F_fun_n_in();
		n_out = F_fun_n_out();

		sz_arg=n_in, sz_res=n_out, sz_iw=0, sz_w=0;
		if (F_fun_work(&sz_arg, &sz_res, &sz_iw, &sz_w)) return 1;

		// real_t *arg[sz_arg];
	    // real_t *res[sz_res];
	    // uint32_t iw[sz_iw];
	    // real_t w[sz_w];

		F_fun_incref();
		real_t xm[3];
		xm[0] = ctl->ekf->x_hat[0];
		xm[1] = ctl->ekf->x_hat[1];
		xm[2] = ctl->ekf->x_hat[2];
		arg[0] = xm;
	    arg[1] = u;
	    arg[2] = p;
	    res[0] = ctl->ekf->F;

	    if (F_fun(arg, res, iw, w, 0)) return 1;

		F_fun_decref();

		// predict covariance
		real_t t_step = 0.05;
		real_t F_mean_1[nx*nx];
		real_t F_mean_fac[nx*nx];
		real_t F_mean_2[nx*nx];// = { 0 };
		real_t F_exp[nx*nx];// = { 0 };
		// adapt to time step
		for (i=0; i<nx*nx; i++) {
			ctl->ekf->F[i] = t_step*ctl->ekf->F[i];
		}
		//  initialize with identity matrix
		for (i=0; i<nx; i++) {
			F_exp[i*nx] = 1.0;
			F_mean_2[i*nx] = 1.0;
		}
		real_t fac = 1;
		// taylor approximation of matrix exponential
		for (i=0; i<12; i++) {
			mtx_times_mtx(F_mean_1,F_mean_2,ctl->ekf->F,nx,nx,nx);
			fac = fac*(i+1);
			for (j=0; j<nx*nx; j++) {
				F_mean_fac[j] = F_mean_1[j]/fac;
			}
			for (j=0; j<nx*nx; j++) {
				F_mean_2[j] = F_mean_1[j];
			}
			mtx_add(F_exp,F_exp,F_mean_fac,nx,nx);
		}

		real_t F_exp_T[nx*nx], PH_55_1[nx*nx], PH_55_2[nx*nx];
		mtx_transpose(F_exp_T,F_exp,nx,nx);
		mtx_times_mtx(PH_55_1,F_exp,ctl->ekf->P,nx,nx,nx);
		mtx_times_mtx(PH_55_2,PH_55_1,F_exp_T,nx,nx,nx);
		mtx_add(ctl->ekf->P,PH_55_2,ctl->ekf->Q,nx,nx);

		// innovation
		real_t S[ny*ny], PH_52[nx*ny],PH_22[ny*ny];
		mtx_times_mtx(PH_52,ctl->ekf->P,ctl->ekf->HT,nx,nx,ny);
		mtx_times_mtx(PH_22,ctl->ekf->H,PH_52,ny,nx,ny);
		mtx_add(PH_22,PH_22,ctl->ekf->R,nx,ny);
		mtx_inv_2x2(S,PH_22);

		// compute Kalman gain
		real_t K[nx*ny];
		mtx_times_mtx(K,PH_52,S,nx,ny,ny);

		// residual
		real_t yk[ny], xk_meas[ny];
		xk_meas[0] = xk[0];
		xk_meas[1] = xk[1];
		mtx_substract(yk,ctl->ekf->y,xk_meas,ny,1);

		// update state estimate
		real_t PH_51[5];
		mtx_times_vec_dense(PH_51,K,yk,nx,ny);
		mtx_add(ctl->ekf->x_hat,ctl->ekf->x_hat,PH_51,nx,1);

		// update covariance estimate
		mtx_times_mtx(PH_55_1,K,ctl->ekf->H,nx,ny,nx);
		mtx_substract(PH_55_2,ctl->ekf->I,PH_55_1,nx,ny);
		mtx_times_mtx(PH_55_1,PH_55_2,ctl->ekf->P,nx,nx,nx);
		for (i=0; i<nx*nx; i++) {
			ctl->ekf->P[i] = PH_55_1[i];
		};

	}

	ctl->in[0] = ctl->ekf->x_hat[0];
	ctl->in[1] = ctl->ekf->x_hat[2];
	ctl->in[2] = ctl->ekf->x_hat[3];

	return;

}

static void make_scale(struct edgeAI_ctl *ctl)
{
	real_t in_dif[SIZE_INPUT];

	mtx_substract(in_dif, ctl->in, ctl->dnn->low_in, 3, 1);
	mult_scale(ctl->in_scaled, in_dif, ctl->dnn->dif_inv_in, 3, 1);


	return;
}

static void make_unscale(struct edgeAI_ctl *ctl)
{
	real_t out_scaled_biased[SIZE_OUTPUT];

	mult_scale(out_scaled_biased, ctl->out_scaled, ctl->dnn->dif_out, 1, 1);
	mtx_add(ctl->out, out_scaled_biased, ctl->dnn->low_out, 1, 1);


	return;
}
