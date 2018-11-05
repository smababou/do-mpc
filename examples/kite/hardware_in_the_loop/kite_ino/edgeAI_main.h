#ifndef EDGEAI_MAIN_H
#define EDGEAI_MAIN_H
#include "edgeAI_arch.h"
#include "mtx_ops.h"
#include "edgeAI_const.h"

extern void make_dnn_step(
	struct edgeAI_ctl *ctl
	);

extern void make_ekf_step(
	struct edgeAI_ctl *ctl
	);

// extern void make_projection_step(
// 	struct edgeAI_ctl * ctl
// 	);

#endif
