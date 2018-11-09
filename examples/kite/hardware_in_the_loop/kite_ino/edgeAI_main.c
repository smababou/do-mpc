#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "edgeAI_main.h"
#include "edgeAI_const.h"
#include "workspace.h"
#include "osqp.h"

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

void make_projection_step(struct edgeAI_ctl *ctl)
{

	// predict states
	const uint32_t ems_pred = 4;
	const c_float ems_pred_f = 4.0;
	const c_float dt = 0.15;
	uint32_t i,j;
	c_float r1[PARAM_SETTINGS], r2[PARAM_SETTINGS], xp1[STATES], xp2[STATES], xp3[STATES], xp4[STATES];
	c_float uk = (c_float) ctl->out[0];

	for (i=0; i<3; i++) {

		xp1[i] = (c_float) ctl->ekf->x_hat[i];
		xp2[i] = (c_float) ctl->ekf->x_hat[i];
		xp3[i] = (c_float) ctl->ekf->x_hat[i];
		xp4[i] = (c_float) ctl->ekf->x_hat[i];

	}

	for (i=0; i<ems_pred; i++) {

		// state theta
		r1[0] =(4.0-(0.028*pow(uk,2)));
		r1[1] =(4.0-(0.028*pow(uk,2)));
		r1[2] =(6.0-(0.028*pow(uk,2)));
		r1[3] =(6.0-(0.028*pow(uk,2)));
		xp1[0] = xp1[0] + dt/ems_pred_f * ((((7.0*r1[0])*cos(xp1[0]))/400.0)*(cos(xp1[2])-(tan(xp1[0])/r1[0])));
		xp2[0] = xp2[0] + dt/ems_pred_f * ((((13.0*r1[1])*cos(xp2[0]))/400.0)*(cos(xp2[2])-(tan(xp2[0])/r1[1])));
		xp3[0] = xp3[0] + dt/ems_pred_f * ((((7.0*r1[2])*cos(xp3[0]))/400.0)*(cos(xp3[2])-(tan(xp3[0])/r1[2])));
		xp4[0] = xp4[0] + dt/ems_pred_f * ((((13.0*r1[3])*cos(xp4[0]))/400.0)*(cos(xp4[2])-(tan(xp4[0])/r1[3])));

		// state phi
		xp1[1] = xp1[1] + dt/ems_pred_f * (-((((7.0*(4.0-(0.028*pow(uk,2))))*cos(xp1[0]))/(400.0*sin(xp1[0])))*sin(xp1[2])));
		xp2[1] = xp2[1] + dt/ems_pred_f * (-((((13.0*(4.0-(0.028*pow(uk,2))))*cos(xp2[0]))/(400.0*sin(xp2[0])))*sin(xp2[2])));
		xp3[1] = xp3[1] + dt/ems_pred_f * (-((((7.0*(6.0-(0.028*pow(uk,2))))*cos(xp3[0]))/(400.0*sin(xp3[0])))*sin(xp3[2])));
		xp4[1] = xp4[1] + dt/ems_pred_f * (-((((13.0*(6.0-(0.028*pow(uk,2))))*cos(xp4[0]))/(400.0*sin(xp4[0])))*sin(xp4[2])));

		// state psi
		r2[0] =((7.0*(4.0-(0.028*pow(uk,2))))*cos(xp1[0]));
		r2[1] =((13.0*(4.0-(0.028*pow(uk,2))))*cos(xp2[0]));
		r2[2] =((7.0*(6.0-(0.028*pow(uk,2))))*cos(xp3[0]));
		r2[3] =((13.0*(6.0-(0.028*pow(uk,2))))*cos(xp4[0]));
		xp1[2] = xp1[2] + dt/ems_pred_f * (((r2[0]/400.0)*uk)-(((r2[0]/(400.0*sin(xp1[0])))*sin(xp1[2]))*cos(xp1[0])));
		xp2[2] = xp2[2] + dt/ems_pred_f * (((r2[1]/400.0)*uk)-(((r2[1]/(400.0*sin(xp2[0])))*sin(xp2[2]))*cos(xp2[0])));
		xp3[2] = xp3[2] + dt/ems_pred_f * (((r2[2]/400.0)*uk)-(((r2[2]/(400.0*sin(xp3[0])))*sin(xp3[2]))*cos(xp3[0])));
		xp4[2] = xp4[2] + dt/ems_pred_f * (((r2[3]/400.0)*uk)-(((r2[3]/(400.0*sin(xp4[0])))*sin(xp4[2]))*cos(xp4[0])));

	}

	// predict heights
	c_float h1, h2, h3, h4;
	const c_float min_height = 100.0, L_tether = 400.0;
	h1 = L_tether * sin(xp1[0]) * cos(xp1[1]);
	h2 = L_tether * sin(xp2[0]) * cos(xp2[1]);
	h3 = L_tether * sin(xp3[0]) * cos(xp3[1]);
	h4 = L_tether * sin(xp4[0]) * cos(xp4[1]);

	// check if violation predicted
	uint32_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;
	if (h1 < min_height) {
		v1 = 1;
	}
	if (h2 < min_height) {
		v2 = 1;
	}
	if (h3 < min_height) {
		v3 = 1;
	}
	if (h4 < min_height) {
		v4 = 1;
	}

	// if violation -> projection
	if (v1 || v2 || v3 || v4) {

		// update linear cost
		c_float q[] = {-uk, 0.0, 0.0, 0.0, 0.0};
		c_int e_q = osqp_update_lin_cost(&workspace, q);

		// update constraint matrix
		c_float H_1[] = { L_tether*cos(xp1[0])*cos(xp1[1]), -L_tether*sin(xp1[0])*sin(xp1[1]), 0 };
		c_float H_2[] = { L_tether*cos(xp2[0])*cos(xp2[1]), -L_tether*sin(xp2[0])*sin(xp2[1]), 0 };
		c_float H_3[] = { L_tether*cos(xp3[0])*cos(xp3[1]), -L_tether*sin(xp3[0])*sin(xp3[1]), 0 };
		c_float H_4[] = { L_tether*cos(xp4[0])*cos(xp4[1]), -L_tether*sin(xp4[0])*sin(xp4[1]), 0 };

		c_float HP1, HP2, HP3, HP4;
		HP1 = ((cos(xp1[1])*(L_tether*cos(xp1[0])))*xp1[0])-(((L_tether*sin(xp1[0]))*sin(xp1[1]))*xp1[1]);
		HP2 = ((cos(xp2[1])*(L_tether*cos(xp2[0])))*xp2[0])-(((L_tether*sin(xp2[0]))*sin(xp2[1]))*xp2[1]);
		HP3 = ((cos(xp3[1])*(L_tether*cos(xp3[0])))*xp3[0])-(((L_tether*sin(xp3[0]))*sin(xp3[1]))*xp3[1]);
		HP4 = ((cos(xp4[1])*(L_tether*cos(xp4[0])))*xp4[0])-(((L_tether*sin(xp4[0]))*sin(xp4[1]))*xp4[1]);

		c_float Bt_1[STATES], Bt_2[STATES], Bt_3[STATES], Bt_4[STATES];
		c_float E_0;
		c_float v_0;


		E_0 = 4.0;
		v_0 = 7.0;
		Bt_1[0] = -(0.0375*(((cos(xp1[2])-(tan(xp1[0])/(E_0-(0.028*pow(uk,2)))))*(0.0025*(cos(xp1[0])*(v_0*(0.028*(uk+uk))))))+((((v_0*(E_0-(0.028*pow(uk,2))))*cos(xp1[0]))/L_tether)*(((tan(xp1[0])/(E_0-(0.028*pow(uk,2))))/(E_0-(0.028*pow(uk,2))))*(0.028*(uk+uk))))));
		Bt_1[1] = (0.0375*(sin(xp1[2])*((cos(xp1[0])*(v_0*(0.028*(uk+uk))))/(L_tether*sin(xp1[0])))));
		Bt_1[2] = 0.0375*(((((v_0*(E_0-(0.028*pow(uk,2))))*cos(xp1[0]))/L_tether)-(uk*(0.0025*(cos(xp1[0])*(v_0*(0.028*(uk+uk)))))))+(cos(xp1[0])*(sin(xp1[2])*((cos(xp1[0])*(v_0*(0.028*(uk+uk))))/(L_tether*sin(xp1[0]))))));

		E_0 = 4.0;
		v_0 = 13.0;
		Bt_2[0] = -(0.0375*(((cos(xp1[2])-(tan(xp1[0])/(E_0-(0.028*pow(uk,2)))))*(0.0025*(cos(xp1[0])*(v_0*(0.028*(uk+uk))))))+((((v_0*(E_0-(0.028*pow(uk,2))))*cos(xp1[0]))/L_tether)*(((tan(xp1[0])/(E_0-(0.028*pow(uk,2))))/(E_0-(0.028*pow(uk,2))))*(0.028*(uk+uk))))));
		Bt_2[1] = (0.0375*(sin(xp1[2])*((cos(xp1[0])*(v_0*(0.028*(uk+uk))))/(L_tether*sin(xp1[0])))));
		Bt_2[2] = 0.0375*(((((v_0*(E_0-(0.028*pow(uk,2))))*cos(xp1[0]))/L_tether)-(uk*(0.0025*(cos(xp1[0])*(v_0*(0.028*(uk+uk)))))))+(cos(xp1[0])*(sin(xp1[2])*((cos(xp1[0])*(v_0*(0.028*(uk+uk))))/(L_tether*sin(xp1[0]))))));

		E_0 = 6.0;
		v_0 = 7.0;
		Bt_3[0] = -(0.0375*(((cos(xp1[2])-(tan(xp1[0])/(E_0-(0.028*pow(uk,2)))))*(0.0025*(cos(xp1[0])*(v_0*(0.028*(uk+uk))))))+((((v_0*(E_0-(0.028*pow(uk,2))))*cos(xp1[0]))/L_tether)*(((tan(xp1[0])/(E_0-(0.028*pow(uk,2))))/(E_0-(0.028*pow(uk,2))))*(0.028*(uk+uk))))));
		Bt_3[1] = (0.0375*(sin(xp1[2])*((cos(xp1[0])*(v_0*(0.028*(uk+uk))))/(L_tether*sin(xp1[0])))));
		Bt_3[2] = 0.0375*(((((v_0*(E_0-(0.028*pow(uk,2))))*cos(xp1[0]))/L_tether)-(uk*(0.0025*(cos(xp1[0])*(v_0*(0.028*(uk+uk)))))))+(cos(xp1[0])*(sin(xp1[2])*((cos(xp1[0])*(v_0*(0.028*(uk+uk))))/(L_tether*sin(xp1[0]))))));

		E_0 = 6.0;
		v_0 = 13.0;
		Bt_4[0] = -(0.0375*(((cos(xp1[2])-(tan(xp1[0])/(E_0-(0.028*pow(uk,2)))))*(0.0025*(cos(xp1[0])*(v_0*(0.028*(uk+uk))))))+((((v_0*(E_0-(0.028*pow(uk,2))))*cos(xp1[0]))/L_tether)*(((tan(xp1[0])/(E_0-(0.028*pow(uk,2))))/(E_0-(0.028*pow(uk,2))))*(0.028*(uk+uk))))));
		Bt_4[1] = (0.0375*(sin(xp1[2])*((cos(xp1[0])*(v_0*(0.028*(uk+uk))))/(L_tether*sin(xp1[0])))));
		Bt_4[2] = 0.0375*(((((v_0*(E_0-(0.028*pow(uk,2))))*cos(xp1[0]))/L_tether)-(uk*(0.0025*(cos(xp1[0])*(v_0*(0.028*(uk+uk)))))))+(cos(xp1[0])*(sin(xp1[2])*((cos(xp1[0])*(v_0*(0.028*(uk+uk))))/(L_tether*sin(xp1[0]))))));

		c_float Ht_1, Ht_2, Ht_3, Ht_4;
		Ht_1 = H_1[0]*Bt_1[0] + H_1[1]*Bt_1[1] + H_1[2]*Bt_1[2];
		Ht_2 = H_2[0]*Bt_2[0] + H_2[1]*Bt_2[1] + H_2[2]*Bt_2[2];
		Ht_3 = H_3[0]*Bt_3[0] + H_3[1]*Bt_3[1] + H_3[2]*Bt_3[2];
		Ht_4 = H_4[0]*Bt_4[0] + H_4[1]*Bt_4[1] + H_4[2]*Bt_4[2];

		c_float A_new[16];
		c_int A_new_index[16];

		A_new[0] = Bt_1[0];
		A_new[1] = Bt_1[1];
		A_new[2] = Bt_1[2];

		A_new[3] = Bt_2[0];
		A_new[4] = Bt_2[1];
		A_new[5] = Bt_2[2];

		A_new[6] = Bt_3[0];
		A_new[7] = Bt_3[1];
		A_new[8] = Bt_3[2];

		A_new[9] = Bt_4[0];
		A_new[10] = Bt_4[1];
		A_new[11] = Bt_4[2];

		A_new[12] = Ht_1;
		A_new[13] = Ht_2;
		A_new[14] = Ht_3;
		A_new[15] = Ht_4;

		c_int i;
		for (i=0; i<16; i++) {
			A_new_index[i] = i;
		}

		c_int e_A = osqp_update_A(&workspace,A_new,A_new_index,16);

		// update bounds
		c_float l[] = {
			0.0,
			-0.5*M_PI,
			-1.0*M_PI,
			0.0,
			-0.5*M_PI,
			-1.0*M_PI,
			0.0,
			-0.5*M_PI,
			-1.0*M_PI,
			0.0,
			-0.5*M_PI,
			-1.0*M_PI,
			100.0,
			100.0,
			100.0,
			100.0,
			-10.0,
			0.0,
			0.0,
			0.0,
			0.0
		};
		c_float u[] = {
			0.5*M_PI,
			0.5*M_PI,
			1.0*M_PI,
			0.5*M_PI,
			0.5*M_PI,
			1.0*M_PI,
			0.5*M_PI,
			0.5*M_PI,
			1.0*M_PI,
			0.5*M_PI,
			0.5*M_PI,
			1.0*M_PI,
			500.0,
			500.0,
			500.0,
			500.0,
			10.0,
			1000.0,
			1000.0,
			1000.0,
			1000.0
		};

		l[0] = l[0] - xp1[0] + Bt_1[0]*uk;
		l[1] = l[1] - xp1[1] + Bt_1[1]*uk;
		l[2] = l[2] - xp1[2] + Bt_1[2]*uk;
		l[3] = l[3] - xp2[0] + Bt_2[0]*uk;
		l[4] = l[4] - xp2[1] + Bt_2[1]*uk;
		l[5] = l[5] - xp2[2] + Bt_2[2]*uk;
		l[6] = l[6] - xp3[0] + Bt_3[0]*uk;
		l[7] = l[7] - xp3[1] + Bt_3[1]*uk;
		l[8] = l[8] - xp3[2] + Bt_3[2]*uk;
		l[9] = l[9] - xp4[0] + Bt_4[0]*uk;
		l[10] = l[10] - xp4[1] + Bt_4[1]*uk;
		l[11] = l[11] - xp4[2] + Bt_4[2]*uk;

		l[12] = l[12] - h1 + Ht_1*uk;
		l[13] = l[13] - h2 + Ht_2*uk;
		l[14] = l[14] - h3 + Ht_3*uk;
		l[15] = l[15] - h4 + Ht_4*uk;

		u[0] = u[0] - xp1[0] + Bt_1[0]*uk;
		u[1] = u[1] - xp1[1] + Bt_1[1]*uk;
		u[2] = u[2] - xp1[2] + Bt_1[2]*uk;
		u[3] = u[3] - xp2[0] + Bt_2[0]*uk;
		u[4] = u[4] - xp2[1] + Bt_2[1]*uk;
		u[5] = u[5] - xp2[2] + Bt_2[2]*uk;
		u[6] = u[6] - xp3[0] + Bt_3[0]*uk;
		u[7] = u[7] - xp3[1] + Bt_3[1]*uk;
		u[8] = u[8] - xp3[2] + Bt_3[2]*uk;
		u[9] = u[9] - xp4[0] + Bt_4[0]*uk;
		u[10] = u[10] - xp4[1] + Bt_4[1]*uk;
		u[11] = u[11] - xp4[2] + Bt_4[2]*uk;

		u[12] = u[12] - h1 + Ht_1*uk;
		u[13] = u[13] - h2 + Ht_2*uk;
		u[14] = u[14] - h3 + Ht_3*uk;
		u[15] = u[15] - h4 + Ht_4*uk;

		c_int e_bounds = osqp_update_bounds(&workspace, l, u);

		// solve optimization problem
		osqp_solve(&workspace);

		// update optimal solution
		ctl->out[0] = (real_t) (&workspace)->solution->x[0];

	}


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
