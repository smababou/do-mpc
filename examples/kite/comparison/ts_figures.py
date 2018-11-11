#load needed modules
import numpy as NP
import scipy.io as sio
import pdb

# parameters
path_to_data = 'results_sampling_time/'
# path_to_data = '../data'
offset = 0
n_batches = 50
mean_all = False

# sizes
nx = 3
nu = 1
np = 2

# additional params
L_tether = 400.0
h_min = 100.0
A = 300.0
pi = NP.pi
beta = 0;
c_tilde = 0.028

# load data
data_005 = []
for i in range(offset, offset + n_batches):
    data_005.append(NP.load(path_to_data + "data_batch_NN_0.05_" + str(i) + ".npy"))

data_010 = []
for i in range(offset, offset + n_batches):
    data_010.append(NP.load(path_to_data + "data_batch_NN_0.1_" + str(i) + ".npy"))

data_015 = []
for i in range(offset, offset + n_batches):
    data_015.append(NP.load(path_to_data + "data_batch_NN_0.15_" + str(i) + ".npy"))

# initialize lists
TF_005 = []
mean_viol_005 = []
max_viol_005 = []
n_viol_005 = []

TF_010 = []
mean_viol_010 = []
max_viol_010 = []
n_viol_010 = []

TF_015 = []
mean_viol_015 = []
max_viol_015 = []
n_viol_015 = []

# start plotting
for i in range(n_batches):

    # MPC results
    t_005 = data_005[i][:,0]
    x_r_005 = data_005[i][:,1:1+nx]
    x_e_005 = data_005[i][:,1+nx:1+2*nx]
    u_005 = data_005[i][:,1+2*nx:1+2*nx+nu]
    p_r_005 = data_005[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_005 = data_005[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    # compute tension
    v_0 = NP.reshape(p_r_005[:,1],(-1,1))
    P_D = NP.square(v_0)/2.0
    E_0 = p_r_005[0,0]
    E = E_0 - c_tilde * NP.square(u_005)
    theta = NP.reshape(x_r_005[:,0],(-1,1))
    phi = NP.reshape(x_r_005[:,1],(-1,1))
    T_F = (A * P_D * NP.square(NP.cos(theta)) * (E+1.0) * NP.sqrt(NP.square(E)+1)) * (NP.cos(theta) * NP.cos(beta) + NP.sin(theta) * NP.sin(beta) * NP.sin(phi))
    TF_005.append(NP.mean(T_F))

    # compute violations
    h = L_tether * NP.sin(theta) * NP.cos(phi)
    v_ind = h < h_min
    nv = sum(v_ind)
    if nv > 0:
        v_array = h[v_ind]
        v_array = NP.abs(v_array - h_min)
        v_max = NP.max(v_array)
        if not mean_all:
            v_mean = NP.mean(v_array)
        else:
            v_mean = NP.sum(v_array)/data_005[i].shape[0]
    else:
        v_max = 0.0
        v_mean = 0.0

    mean_viol_005.append(v_mean)
    max_viol_005.append(v_max)
    n_viol_005.append(nv)



    # MPC results
    t_010 = data_010[i][:,0]
    x_r_010 = data_010[i][:,1:1+nx]
    x_e_010 = data_010[i][:,1+nx:1+2*nx]
    u_010 = data_010[i][:,1+2*nx:1+2*nx+nu]
    p_r_010 = data_010[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_010 = data_010[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    # compute tension
    v_0 = NP.reshape(p_r_010[:,1],(-1,1))
    P_D = NP.square(v_0)/2.0
    E_0 = p_r_010[0,0]
    E = E_0 - c_tilde * NP.square(u_010)
    theta = NP.reshape(x_r_010[:,0],(-1,1))
    phi = NP.reshape(x_r_010[:,1],(-1,1))
    T_F = (A * P_D * NP.square(NP.cos(theta)) * (E+1.0) * NP.sqrt(NP.square(E)+1)) * (NP.cos(theta) * NP.cos(beta) + NP.sin(theta) * NP.sin(beta) * NP.sin(phi))
    TF_010.append(NP.mean(T_F))

    # compute violations
    h = L_tether * NP.sin(theta) * NP.cos(phi)
    v_ind = h < h_min
    nv = sum(v_ind)
    if nv > 0:
        v_array = h[v_ind]
        v_array = NP.abs(v_array - h_min)
        v_max = NP.max(v_array)
        if not mean_all:
            v_mean = NP.mean(v_array)
        else:
            v_mean = NP.sum(v_array)/data_010[i].shape[0]
    else:
        v_max = 0.0
        v_mean = 0.0

    mean_viol_010.append(v_mean)
    max_viol_010.append(v_max)
    n_viol_010.append(nv)



    # NN results
    t_015 = data_015[i][:,0]
    x_r_015 = data_015[i][:,1:1+nx]
    x_e_015 = data_015[i][:,1+nx:1+2*nx]
    u_015 = data_015[i][:,1+2*nx:1+2*nx+nu]
    p_r_015 = data_015[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_015 = data_015[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    # compute tension
    v_0 = NP.reshape(p_r_015[:,1],(-1,1))
    P_D = NP.square(v_0)/2.0
    E_0 = p_r_015[0,0]
    E = E_0 - c_tilde * NP.square(u_015)
    theta = NP.reshape(x_r_015[:,0],(-1,1))
    phi = NP.reshape(x_r_015[:,1],(-1,1))
    T_F = (A * P_D * NP.square(NP.cos(theta)) * (E+1.0) * NP.sqrt(NP.square(E)+1)) * (NP.cos(theta) * NP.cos(beta) + NP.sin(theta) * NP.sin(beta) * NP.sin(phi))
    TF_015.append(NP.mean(T_F))

    # compute violations
    h = L_tether * NP.sin(theta) * NP.cos(phi)
    v_ind = h < h_min
    nv = sum(v_ind)
    if nv > 0:
        v_array = h[v_ind]
        v_array = NP.abs(v_array - h_min)
        v_max = NP.max(v_array)
        if not mean_all:
            v_mean = NP.mean(v_array)
        else:
            v_mean = NP.sum(v_array)/data_015[i].shape[0]
    else:
        v_max = 0.0
        v_mean = 0.0

    mean_viol_015.append(v_mean)
    max_viol_015.append(v_max)
    n_viol_015.append(nv)

# comparison
TF_005 = NP.vstack(TF_005)
mean_viol_005 = NP.vstack(mean_viol_005)
max_viol_005 = NP.vstack(max_viol_005)
n_viol_005 = NP.vstack(n_viol_005)

TF_010 = NP.vstack(TF_010)
mean_viol_010 = NP.vstack(mean_viol_010)
max_viol_010 = NP.vstack(max_viol_010)
n_viol_010 = NP.vstack(n_viol_010)

TF_015 = NP.vstack(TF_015)
mean_viol_015 = NP.vstack(mean_viol_015)
max_viol_015 = NP.vstack(max_viol_015)
n_viol_015 = NP.vstack(n_viol_015)

print("\t\t0.05s \t\t0.10s \t\t0.15s")
print("Tension:\t%8.4f,\t%8.4f,\t%8.4f" % (NP.mean(TF_005), NP.mean(TF_010), NP.mean(TF_015)))
print("V (mean):\t%8.4f,\t%8.4f,\t%8.4f" % (NP.mean(mean_viol_005), NP.mean(mean_viol_010), NP.mean(mean_viol_015)))
print("V (max):\t%8.4f\t%8.4f,\t%8.4f" % (NP.max(max_viol_005), NP.max(max_viol_010), NP.max(max_viol_015)))
# print("Violation (number):\tMPC\t%8.4f,\tNN\t%8.4f" % (NP.mean(TF_MPC), NP.mean(TF_NN)))
