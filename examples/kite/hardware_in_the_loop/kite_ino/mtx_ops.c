#include "mtx_ops.h"
#ifndef MATH
#define MATH
#include <math.h>

#endif

void mtx_times_vec_dense(real_t pout[], const real_t pmtx[], const real_t pvec[], const uint32_t rows, const uint32_t cols)
{
	uint32_t i, j, k = 0;

	for (i = 0; i < rows; i++) {
		pout[i] = 0;
		for (j = 0; j < cols; j++) {
			 pout[i] += pmtx[k] * pvec[j];
				k++;
		}
	}

	return;
}

void mtx_times_vec_sparse(real_t pout[], const real_t data[], const uint32_t ptr[], const uint32_t ind[], const real_t vec[], const uint32_t rows)
{
	uint32_t i, j;

	for (i = 0; i < rows; i++) {
		pout[i] = 0;
		for (j = ptr[i]; j < ptr[i+1]; j++) {
			pout[i] += data[j]*vec[ind[j]];
		}
	}

	return;
}

void mult_scale(real_t out[], const real_t in[], const real_t sca[], const uint32_t rows, const uint32_t cols)
{
	uint32_t k;

	for (k = 0; k < rows * cols; k++) {
	out[k] = in[k] * sca[k];
	}

	return;
}

void mtx_add(real_t pmtxc[], const real_t pmtxa[], const real_t pmtxb[], const uint32_t rows, const uint32_t cols)
{
	uint32_t k;

	for (k = 0; k < rows * cols; k++) {
		pmtxc[k] = pmtxa[k] + pmtxb[k];
	}

	return;
}

void mtx_substract(real_t pmtxc[], const real_t pmtxa[], const real_t pmtxb[], const uint32_t rows, const uint32_t cols)
{
	uint32_t k;

	for (k = 0; k < rows * cols; k++) {
		pmtxc[k] = pmtxa[k] - pmtxb[k];
	}

	return;
}

void mtx_tanh(real_t vec[], const uint32_t rows)
{
	uint32_t i;

	for (i = 0; i < rows; i++) {
		vec[i] = tanh(vec[i]);
	}
}

void mtx_times_mtx(real_t out[], const real_t mtx1[], const real_t mtx2[], const uint32_t rows1, const uint32_t rowcol, const uint32_t cols2)
 {
	uint32_t i, j, k, m = 0;

	for (i = 0; i < rows1; i++) {
		for (j = 0; j < cols2; j++) {
			out[k] = 0;
			for (m = 0; m < rowcol; m++) {
				out[k] += mtx1[i*rowcol+m] * mtx2[j+m*cols2];
			}
			k++;
		}
	}

	return;
 }
