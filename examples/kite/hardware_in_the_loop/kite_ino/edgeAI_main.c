#include <stddef.h>
#include <string.h>
#include <math.h>
#include "edgeAI_main.h"
#include "edgeAI_const.h"

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
	const uint32_t nx = 5, ny = 2, ems = 50;
	const real_t dt = 0.05;
	uint32_t i, j;
	real_t r1, r2, xk[3];

	// predict states
	for (i=0; i<3; i++) {
		xk[i] = ctl->ekf->x_hat[i];
	}
	for (i=0; i<ems; i++) {

		// state xk[0]
		r1 =(ctl->ekf->x_hat[3]-(0.028*pow(ctl->out[0],2)));
		xk[0] = xk[0] + dt/ems * ((((ctl->ekf->x_hat[4]*r1)*cos(xk[0]))/400.0)*(cos(xk[2])-(tan(xk[0])/r1)));
		// state phi
		xk[1] = xk[1] + dt/ems * (-((((ctl->ekf->x_hat[4]*(ctl->ekf->x_hat[3]-(0.028*pow(ctl->out[0],2))))*cos(xk[0]))/(400.0*sin(xk[0])))*sin(xk[2])));
		// state xk[2]
		r2 =((ctl->ekf->x_hat[4]*(ctl->ekf->x_hat[3]-(0.028*pow(ctl->out[0],2))))*cos(xk[0]));
		xk[2] = xk[2] + dt/ems * (((r2/400.0)*ctl->out[0])-(((r2/(400.0*sin(xk[0])))*sin(xk[2]))*cos(xk[0])));

	}

	// state transition
	const real_t c_tilde = 0.028;
	const real_t L_tether = 400.0;
	const real_t p1 = 0.0025;
	real_t h1, h2, h3, h4, h5; // auxiliary terms
	h1 = ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2));
	h2 = ctl->ekf->x_hat[4]*h1;
	ctl->ekf->F[0] = -(((cos(ctl->ekf->x_hat[2])-(tan(ctl->ekf->x_hat[0])/h1))*(p1*(h2*sin(ctl->ekf->x_hat[0]))))+(((h2*cos(ctl->ekf->x_hat[0]))/L_tether)*((1./h1)/pow(cos(ctl->ekf->x_hat[0]),2))));
	ctl->ekf->F[1] = 0.0;
	ctl->ekf->F[2] = -((((ctl->ekf->x_hat[4]*(ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2))))*cos(ctl->ekf->x_hat[0]))/L_tether)*sin(ctl->ekf->x_hat[2]));
	h1 = ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2));
	h2 = tan(ctl->ekf->x_hat[0])/h1;
	h3 = cos(ctl->ekf->x_hat[0]);
	ctl->ekf->F[3] = ((cos(ctl->ekf->x_hat[2])-h2)*(p1*(h3*ctl->ekf->x_hat[4])))+((((ctl->ekf->x_hat[4]*h1)*h3)/L_tether)*(h2/h1));
	h1 = ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2));
	ctl->ekf->F[4] = (cos(ctl->ekf->x_hat[2])-(tan(ctl->ekf->x_hat[0])/h1))*(p1*(cos(ctl->ekf->x_hat[0])*h1));
	h1 = ctl->ekf->x_hat[4]*(ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2)));
	h2 = L_tether*sin(ctl->ekf->x_hat[0]);
	ctl->ekf->F[5] = sin(ctl->ekf->x_hat[2])*(((h1*sin(ctl->ekf->x_hat[0]))/h2)+((((h1*cos(ctl->ekf->x_hat[0]))/h2)/h2)*(L_tether*cos(ctl->ekf->x_hat[0]))));
	ctl->ekf->F[6] = 0.0;
	ctl->ekf->F[7] = -((((ctl->ekf->x_hat[4]*(ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2))))*cos(ctl->ekf->x_hat[0]))/(L_tether*sin(ctl->ekf->x_hat[0])))*cos(ctl->ekf->x_hat[2]));
	ctl->ekf->F[8] = -(sin(ctl->ekf->x_hat[2])*((cos(ctl->ekf->x_hat[0])*ctl->ekf->x_hat[4])/(L_tether*sin(ctl->ekf->x_hat[0]))));
	ctl->ekf->F[9] = -(sin(ctl->ekf->x_hat[2])*((cos(ctl->ekf->x_hat[0])*(ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2))))/(L_tether*sin(ctl->ekf->x_hat[0]))));
	h1 = sin(ctl->ekf->x_hat[2]);
	h2 = ctl->ekf->x_hat[4]*(ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2)));
	h3 = h2*sin(ctl->ekf->x_hat[0]);
	h4 = L_tether*sin(ctl->ekf->x_hat[0]);
	h5 = (h2*cos(ctl->ekf->x_hat[0]))/h4;
	ctl->ekf->F[10] = ((cos(ctl->ekf->x_hat[0])*(h1*((h3/h4)+((h5/h4)*(L_tether*cos(ctl->ekf->x_hat[0]))))))+((h5*h1)*sin(ctl->ekf->x_hat[0])))-(ctl->out[0]*(p1*h3));
	ctl->ekf->F[11] = 0.0;
	ctl->ekf->F[12] = -(cos(ctl->ekf->x_hat[0])*((((ctl->ekf->x_hat[4]*(ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2))))*cos(ctl->ekf->x_hat[0]))/(L_tether*sin(ctl->ekf->x_hat[0])))*cos(ctl->ekf->x_hat[2])));
	h1 = cos(ctl->ekf->x_hat[0])*ctl->ekf->x_hat[4];
	ctl->ekf->F[13] = ((ctl->out[0]*(p1*h1))-(cos(ctl->ekf->x_hat[0])*(sin(ctl->ekf->x_hat[2])*(h1/(L_tether*sin(ctl->ekf->x_hat[0]))))));
	h1 = cos(ctl->ekf->x_hat[0])*(ctl->ekf->x_hat[3]-(c_tilde*pow(ctl->out[0],2)));
	ctl->ekf->F[14] = ((ctl->out[0]*(p1*h1))-(cos(ctl->ekf->x_hat[0])*(sin(ctl->ekf->x_hat[2])*(h1/(L_tether*sin(ctl->ekf->x_hat[0]))))));
	for (i=15; i<nx*nx; i++) {
		ctl->ekf->F[i] = 0.0;
	}


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
	for (i=0; i<nx*nx; i++) {
        F_exp[i] = 0.0;
        F_mean_2[i] = 0.0;
    }
	for (i=0; i<nx; i++) {
		F_exp[i*nx+i] = 1.0;
		F_mean_2[i*nx+i] = 1.0;
	}
	uint32_t fac = 1;
	// taylor approximation of matrix exponential
	for (i=0; i<8; i++) {
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

	real_t F_exp_T[nx*nx], PH_55_1[nx*nx], PH_55_2[nx*nx], PH_55_3[nx*nx];
	mtx_transpose(F_exp_T,F_exp,nx,nx);
	mtx_times_mtx(PH_55_1,F_exp,ctl->ekf->P,nx,nx,nx);
	mtx_times_mtx(PH_55_2,PH_55_1,F_exp_T,nx,nx,nx);
	mtx_add(PH_55_3,PH_55_2,ctl->ekf->Q,nx,nx);

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
	real_t PH_51_1[5], PH_51_2[5], xk_aug[5];
	mtx_times_vec_dense(PH_51_1,K,yk,nx,ny);
    xk_aug[0] = xk[0];
    xk_aug[1] = xk[1];
    xk_aug[2] = xk[2];
    xk_aug[3] = ctl->ekf->x_hat[3];
    xk_aug[4] = ctl->ekf->x_hat[4];
	mtx_add(PH_51_2,xk_aug,PH_51_1,nx,1);
	for (i=0; i<5; i++) {
		ctl->ekf->x_hat[i] = PH_51_2[i];
	};


	// update covariance estimate
	real_t PH_55_4[nx*nx], PH_55_5[nx*nx];
	mtx_times_mtx(PH_55_1,K,ctl->ekf->H,nx,ny,nx);
	mtx_substract(PH_55_4,ctl->ekf->I,PH_55_1,nx,nx);
	mtx_times_mtx(PH_55_5,PH_55_4,PH_55_3,nx,nx,nx);
	for (i=0; i<nx*nx; i++) {
		ctl->ekf->P[i] = PH_55_5[i];
	};

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
