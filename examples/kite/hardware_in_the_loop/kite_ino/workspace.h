#include "types.h"
#include "constants.h"
#include "qdldl.h"

#include "qdldl_interface.h"

// Define data structure
c_int Pdata_i[5] = {
0,
1,
2,
3,
4,
};
c_int Pdata_p[6] = {
0,
1,
2,
3,
4,
5,
};
c_float Pdata_x[5] = {
(c_float)1.00000000000000000000,
(c_float)1000.00000000000000000000,
(c_float)1000.00000000000000000000,
(c_float)1000.00000000000000000000,
(c_float)1000.00000000000000000000,
};
csc Pdata = {5, 5, 5, Pdata_p, Pdata_i, Pdata_x, -1};
c_int Adata_i[25] = {
0,
1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,
16,
12,
17,
13,
18,
14,
19,
15,
20,
};
c_int Adata_p[6] = {
0,
17,
19,
21,
23,
25,
};
c_float Adata_x[25] = {
(c_float)-0.00010316851802469795,
(c_float)-0.00037598179706502418,
(c_float)0.00095777895799382539,
(c_float)-0.00020267310505870021,
(c_float)-0.00066162998403492630,
(c_float)0.00181074324472943276,
(c_float)-0.00010730749460080279,
(c_float)-0.00036181046536744245,
(c_float)0.00220417777493541129,
(c_float)-0.00021580193153769456,
(c_float)-0.00061181642799999390,
(c_float)0.00413976648312083719,
(c_float)0.00495426235185531666,
(c_float)0.00543105211842250662,
(c_float)0.00355581715777675278,
(c_float)0.00062036056174415155,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
};
csc Adata = {25, 21, 5, Adata_p, Adata_i, Adata_x, -1};
c_float qdata[5] = {
(c_float)-4.72667007927723403782,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
};
c_float ldata[21] = {
(c_float)-0.34466238068545584916,
(c_float)-2.30354029649703795357,
(c_float)-2.24902111446600150302,
(c_float)-0.34890301569811627003,
(c_float)-2.32124987168326857301,
(c_float)-2.29416705717424740385,
(c_float)-0.34778984770410614757,
(c_float)-2.31483537669062444664,
(c_float)-2.27718056697457571502,
(c_float)-0.35505346788765485000,
(c_float)-2.34100758443900369699,
(c_float)-2.34526202222721735779,
(c_float)30.13803515631289542398,
(c_float)26.65324429163064223758,
(c_float)27.61780640214026050216,
(c_float)21.92596246593510045386,
(c_float)-10.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
};
c_float udata[21] = {
(c_float)1.22613394610944070884,
(c_float)0.83805235709275527345,
(c_float)4.03416419271358428489,
(c_float)1.22189331109678045451,
(c_float)0.82034278190652454299,
(c_float)3.98901825000533882815,
(c_float)1.22300647909079041042,
(c_float)0.82675727689916889140,
(c_float)4.00600474020501096106,
(c_float)1.21574285890724165249,
(c_float)0.80058506915078953003,
(c_float)3.93792328495236887420,
(c_float)430.13803515631286700227,
(c_float)426.65324429163069908100,
(c_float)427.61780640214027471302,
(c_float)421.92596246593507203215,
(c_float)10.00000000000000000000,
(c_float)1000.00000000000000000000,
(c_float)1000.00000000000000000000,
(c_float)1000.00000000000000000000,
(c_float)1000.00000000000000000000,
};
OSQPData data = {5, 21, &Pdata, &Adata, qdata, ldata, udata};

// Define settings structure
OSQPSettings settings = {(c_float)0.10000000000000000555, (c_float)0.00000100000000000000, 0, 1, 0, (c_float)5.00000000000000000000,
#ifdef PROFILING
(c_float)0.40000000000000002220,
#endif  // PROFILING
10000, (c_float)0.00100000000000000002, (c_float)0.00100000000000000002, (c_float)0.00010000000000000000, (c_float)0.00010000000000000000, (c_float)1.60000000000000008882, 0, 0, 25, 1,
#ifdef PROFILING
(c_float)0.00000000000000000000
#endif  // PROFILING
};

// Define scaling structure
OSQPScaling scaling;

// Define linsys_solver structure
c_int linsys_solver_L_i[25] = {
1,
11,
3,
10,
5,
9,
7,
8,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
};
c_int linsys_solver_L_p[27] = {
0,
1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,
16,
17,
18,
19,
20,
21,
22,
23,
24,
25,
25,
};
c_float linsys_solver_L_x[25] = {
(c_float)-0.10000000000000000555,
(c_float)0.00099990000899920000,
(c_float)-0.10000000000000000555,
(c_float)0.00099990000899920000,
(c_float)-0.10000000000000000555,
(c_float)0.00099990000899920000,
(c_float)-0.10000000000000000555,
(c_float)0.00099990000899920000,
(c_float)-0.00006202985380927694,
(c_float)-0.00035554616471634531,
(c_float)-0.00054305091218105291,
(c_float)-0.00049537670246860602,
(c_float)-0.10000000000000000555,
(c_float)-0.00041397664831208374,
(c_float)0.00006118164279999940,
(c_float)0.00002158019315376946,
(c_float)-0.00022041777749354113,
(c_float)0.00003618104653674424,
(c_float)0.00001073074946008028,
(c_float)-0.00018107432447294329,
(c_float)0.00006616299840349263,
(c_float)0.00002026731050587002,
(c_float)-0.00009577789579938255,
(c_float)0.00003759817970650242,
(c_float)0.00001031685180246980,
};
csc linsys_solver_L = {25, 26, 26, linsys_solver_L_p, linsys_solver_L_i, linsys_solver_L_x, -1};
c_float linsys_solver_Dinv[26] = {
(c_float)-0.10000000000000000555,
(c_float)0.00099990000899920000,
(c_float)-0.10000000000000000555,
(c_float)0.00099990000899920000,
(c_float)-0.10000000000000000555,
(c_float)0.00099990000899920000,
(c_float)-0.10000000000000000555,
(c_float)0.00099990000899920000,
(c_float)-0.09999000199961007029,
(c_float)-0.09999000199961007029,
(c_float)-0.09999000199961007029,
(c_float)-0.09999000199961007029,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)-0.10000000000000000555,
(c_float)0.90908227699387833542,
};
c_int linsys_solver_P[26] = {
22,
1,
23,
2,
24,
3,
25,
4,
20,
19,
18,
17,
21,
16,
15,
14,
13,
12,
11,
10,
9,
8,
7,
6,
5,
0,
};
c_float linsys_solver_bp[26];
c_int linsys_solver_Pdiag_idx[5] = {
0,
1,
2,
3,
4,
};
c_int linsys_solver_KKT_i[51] = {
0,
1,
0,
2,
3,
2,
4,
5,
4,
6,
7,
6,
7,
8,
5,
9,
3,
10,
1,
11,
12,
13,
14,
15,
16,
17,
18,
19,
20,
21,
22,
23,
24,
25,
24,
23,
22,
21,
20,
19,
18,
17,
16,
15,
14,
13,
11,
10,
9,
8,
12,
};
c_int linsys_solver_KKT_p[27] = {
0,
1,
3,
4,
6,
7,
9,
10,
12,
14,
16,
18,
20,
21,
22,
23,
24,
25,
26,
27,
28,
29,
30,
31,
32,
33,
51,
};
c_float linsys_solver_KKT_x[51] = {
(c_float)-10.00000000000000000000,
(c_float)1000.00000099999999747524,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)1000.00000099999999747524,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)1000.00000099999999747524,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)1000.00000099999999747524,
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)1.00000099999999991773,
(c_float)-0.00010316851802469795,
(c_float)-0.00037598179706502418,
(c_float)0.00095777895799382539,
(c_float)-0.00020267310505870021,
(c_float)-0.00066162998403492630,
(c_float)0.00181074324472943276,
(c_float)-0.00010730749460080279,
(c_float)-0.00036181046536744245,
(c_float)0.00220417777493541129,
(c_float)-0.00021580193153769456,
(c_float)-0.00061181642799999390,
(c_float)0.00413976648312083719,
(c_float)0.00495426235185531666,
(c_float)0.00543105211842250662,
(c_float)0.00355581715777675278,
(c_float)0.00062036056174415155,
(c_float)1.00000000000000000000,
};
csc linsys_solver_KKT = {51, 26, 26, linsys_solver_KKT_p, linsys_solver_KKT_i, linsys_solver_KKT_x, -1};
c_int linsys_solver_PtoKKT[5] = {
33,
1,
4,
7,
10,
};
c_int linsys_solver_AtoKKT[25] = {
34,
35,
36,
37,
38,
39,
40,
41,
42,
43,
44,
45,
46,
47,
48,
49,
50,
18,
2,
16,
5,
14,
8,
12,
11,
};
c_int linsys_solver_rhotoKKT[21] = {
32,
31,
30,
29,
28,
27,
26,
25,
24,
23,
22,
21,
19,
17,
15,
13,
20,
0,
3,
6,
9,
};
c_float linsys_solver_D[26] = {
(c_float)-10.00000000000000000000,
(c_float)1000.10000100000002021261,
(c_float)-10.00000000000000000000,
(c_float)1000.10000100000002021261,
(c_float)-10.00000000000000000000,
(c_float)1000.10000100000002021261,
(c_float)-10.00000000000000000000,
(c_float)1000.10000100000002021261,
(c_float)-10.00099990000899907727,
(c_float)-10.00099990000899907727,
(c_float)-10.00099990000899907727,
(c_float)-10.00099990000899907727,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)-10.00000000000000000000,
(c_float)1.10001044493658506340,
};
c_int linsys_solver_etree[26] = {
1,
11,
3,
10,
5,
9,
7,
8,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
25,
-1,
};
c_int linsys_solver_Lnz[26] = {
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
0,
};
c_int linsys_solver_iwork[78] = {
24,
23,
22,
21,
20,
19,
18,
17,
16,
15,
14,
13,
11,
10,
9,
8,
12,
25,
25,
25,
25,
25,
25,
25,
25,
25,
12,
140278183285744,
140280168888896,
140278183286384,
140278183606064,
140278184123968,
140278183288704,
140278183553808,
140278183544496,
140278184124208,
140280168888896,
140278183058072,
140278183180048,
140278183058000,
140278187844928,
140278183056704,
140278183540712,
140278183181456,
140278183570736,
140280168888896,
140278183544392,
140278183181016,
140278183261936,
140278183260496,
140280168888896,
140280168888896,
1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,
16,
17,
18,
19,
20,
21,
22,
23,
24,
25,
25,
};
c_int linsys_solver_bwork[26] = {
0,
0,
0,
4294967296,
1,
140280166940352,
1,
4294967768,
1,
140280166940352,
1,
561111789064504,
140278183622480,
140280166940352,
2,
4762890196,
32423,
997323336,
32423,
997877313,
140278183621392,
140279760232208,
1,
561111707418985,
140278183621904,
140280166940352,
};
c_float linsys_solver_fwork[26] = {
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
};
qdldl_solver linsys_solver = {QDLDL_SOLVER, &solve_linsys_qdldl, &update_linsys_solver_matrices_qdldl, &update_linsys_solver_rho_vec_qdldl, &linsys_solver_L, linsys_solver_Dinv, linsys_solver_P, linsys_solver_bp, linsys_solver_Pdiag_idx, 5, &linsys_solver_KKT, linsys_solver_PtoKKT, linsys_solver_AtoKKT, linsys_solver_rhotoKKT, linsys_solver_D, linsys_solver_etree, linsys_solver_Lnz, linsys_solver_iwork, linsys_solver_bwork, linsys_solver_fwork};

// Define solution
c_float xsolution[5];
c_float ysolution[21];

OSQPSolution solution = {xsolution, ysolution};

// Define info
OSQPInfo info = {0, "Unsolved", OSQP_UNSOLVED, 0.0, 0.0, 0.0};

// Define workspace
c_float work_rho_vec[21] = {
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
(c_float)0.10000000000000000555,
};
c_float work_rho_inv_vec[21] = {
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
(c_float)10.00000000000000000000,
};
c_int work_constr_type[21] = {
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
};
c_float work_x[5];
c_float work_y[21];
c_float work_z[21];
c_float work_xz_tilde[26];
c_float work_x_prev[5];
c_float work_z_prev[21];
c_float work_Ax[21];
c_float work_Px[5];
c_float work_Aty[5];
c_float work_delta_y[21];
c_float work_Atdelta_y[5];
c_float work_delta_x[5];
c_float work_Pdelta_x[5];
c_float work_Adelta_x[21];
c_float work_D_temp[5];
c_float work_D_temp_A[5];
c_float work_E_temp[21];

OSQPWorkspace workspace = {
&data, (LinSysSolver *)&linsys_solver,
work_rho_vec, work_rho_inv_vec,
work_constr_type,
work_x, work_y, work_z, work_xz_tilde,
work_x_prev, work_z_prev,
work_Ax, work_Px, work_Aty,
work_delta_y, work_Atdelta_y,
work_delta_x, work_Pdelta_x, work_Adelta_x,
work_D_temp, work_D_temp_A, work_E_temp,
&settings, &scaling, &solution, &info};
