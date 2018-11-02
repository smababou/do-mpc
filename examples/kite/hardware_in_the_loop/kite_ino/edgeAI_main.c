#include <stddef.h>
#include <string.h>
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
