#ifndef MTX_OPS_H
#define MTX_OPS_H
#include "edgeAI_arch.h"

extern void mtx_times_vec_dense(
	real_t pout[],
	const real_t pmtx[],
	const real_t pvec[],
	const uint32_t rows,
	const uint32_t cols
	);

extern void mtx_times_vec_sparse(
	real_t pout[],
	const real_t data[],
	const uint32_t ptr[],
	const uint32_t ind[],
	const real_t vec[],
	const uint32_t rows
	);

extern void mult_scale(
	real_t out[],
	const real_t in[],
	const real_t sca[],
	const uint32_t rows,
	const uint32_t cols
	);

extern void mtx_add(
	real_t pmtxc[],
	const real_t pmtxa[],
	const real_t pmtxb[],
	const uint32_t rows,
	const uint32_t cols
	);

extern void mtx_substract(
	real_t pmtxc[],
	const real_t pmtxa[],
	const real_t pmtxb[],
	const uint32_t rows,
	const uint32_t cols
	);

extern void mtx_tanh(
	real_t vec[],
	const uint32_t rows
	);

extern void mtx_times_mtx(
	real_t out[],
	const real_t mtx1[],
	const real_t mtx2[],
	const uint32_t rows1,
	const uint32_t rowcol,
	const uint32_t cols2
);

extern void mtx_transpose(
	real_t out[],
	const real_t mtx_in[],
	const uint32_t rows,
	const uint32_t cols
);

extern void mtx_inv_2x2(
	real_t out[],
	const real_t mtx_in[]
);

#endif
