#ifndef EDGEAI_ARCH_H
#define EDGEAI_ARCH_H
#include "mc04types.h"
typedef float32_t real_t;

struct edgeAI_dnn{
const real_t *dif_inv_in;
const real_t *low_in;
const real_t *low_out;
const real_t *dif_out;
const real_t *kernel_1;
const real_t *bias_1;
const real_t *kernel_2;
const real_t *bias_2;
const real_t *kernel_3;
const real_t *bias_3;
const real_t *kernel_4;
const real_t *bias_4;
const real_t *kernel_5;
const real_t *bias_5;
const real_t *kernel_6;
const real_t *bias_6;
const real_t *kernel_7;
const real_t *bias_7;
};

struct edgeAI_ekf{
const real_t *Q;
const real_t *R;
real_t *P;
real_t *F;
real_t *H;
real_t *x_hat;
};

struct edgeAI_ctl{
struct edgeAI_dnn *dnn;
struct edgeAI_ekf *ekf;
real_t *in;
real_t *in_scaled;
real_t *out;
real_t *out_scaled;
};


#endif
