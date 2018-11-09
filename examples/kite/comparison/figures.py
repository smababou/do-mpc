#load needed modules
import numpy as NP
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb

plt.rcParams.update({'font.size': 12, 'lines.linewidth' : 2.0,'svg.fonttype': 'none',
    'axes.unicode_minus':False, })

# parameters
path_to_data = 'results/'
# path_to_data = '../data'
offset = 0
n_batches = 1

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
data_MPC = []
for i in range(offset, offset + n_batches):
    data_MPC.append(NP.load(path_to_data + "data_batch_MPC_" + str(i) + ".npy"))

data_NN = []
for i in range(offset, offset + n_batches):
    data_NN.append(NP.load(path_to_data + "data_batch_NN_" + str(i) + ".npy"))

data_MPC_P = []
for i in range(offset, offset + n_batches):
    data_MPC_P.append(NP.load(path_to_data + "data_batch_MPC_proj_" + str(i) + ".npy"))

data_NN_P = []
for i in range(offset, offset + n_batches):
    data_NN_P.append(NP.load(path_to_data + "data_batch_NN_proj_" + str(i) + ".npy"))

# initialize lists
TF_MPC = []
mean_viol_MPC = []
max_viol_MPC = []
n_viol_MPC = []

TF_MPC_P = []
mean_viol_MPC_P = []
max_viol_MPC_P = []
n_viol_MPC_P = []

TF_NN = []
mean_viol_NN = []
max_viol_NN = []
n_viol_NN = []

TF_NN_P = []
mean_viol_NN_P = []
max_viol_NN_P = []
n_viol_NN_P = []

# start plotting
for i in range(n_batches):

    # MPC results
    t_MPC = data_MPC[i][:,0]
    x_r_MPC = data_MPC[i][:,1:1+nx]
    x_e_MPC = data_MPC[i][:,1+nx:1+2*nx]
    u_MPC = data_MPC[i][:,1+2*nx:1+2*nx+nu]
    p_r_MPC = data_MPC[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_MPC = data_MPC[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    # compute tension
    v_0 = NP.reshape(p_r_MPC[:,1],(-1,1))
    P_D = NP.square(v_0)/2.0
    E_0 = p_r_MPC[0,0]
    E = E_0 - c_tilde * NP.square(u_MPC)
    theta = NP.reshape(x_r_MPC[:,0],(-1,1))
    phi = NP.reshape(x_r_MPC[:,1],(-1,1))
    T_F = A * P_D * NP.square(NP.cos(theta)) * NP.sqrt(NP.square(E)+1) * (NP.cos(theta) * NP.cos(beta) + NP.sin(theta) * NP.sin(beta) * NP.sin(phi));
    TF_MPC.append(NP.mean(T_F))

    # compute violations
    h = L_tether * NP.sin(theta) * NP.cos(phi)
    v_ind = h < h_min
    nv = sum(v_ind)
    if nv > 0:
        v_array = h[v_ind]
        v_array = NP.abs(v_array - h_min)
        v_max = NP.max(v_array)
        v_mean = NP.mean(v_array)
    else:
        v_max = 0.0
        v_mean = 0.0

    mean_viol_MPC.append(v_mean)
    max_viol_MPC.append(v_max)
    n_viol_MPC.append(nv)



    # MPC results
    t_MPC_P = data_MPC_P[i][:,0]
    x_r_MPC_P = data_MPC_P[i][:,1:1+nx]
    x_e_MPC_P = data_MPC_P[i][:,1+nx:1+2*nx]
    u_MPC_P = data_MPC_P[i][:,1+2*nx:1+2*nx+nu]
    p_r_MPC_P = data_MPC_P[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_MPC_P = data_MPC_P[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    # compute tension
    v_0 = NP.reshape(p_r_MPC_P[:,1],(-1,1))
    P_D = NP.square(v_0)/2.0
    E_0 = p_r_MPC_P[0,0]
    E = E_0 - c_tilde * NP.square(u_MPC_P)
    theta = NP.reshape(x_r_MPC_P[:,0],(-1,1))
    phi = NP.reshape(x_r_MPC_P[:,1],(-1,1))
    T_F = A * P_D * NP.square(NP.cos(theta)) * NP.sqrt(NP.square(E)+1) * (NP.cos(theta) * NP.cos(beta) + NP.sin(theta) * NP.sin(beta) * NP.sin(phi));
    TF_MPC_P.append(NP.mean(T_F))

    # compute violations
    h = L_tether * NP.sin(theta) * NP.cos(phi)
    v_ind = h < h_min
    nv = sum(v_ind)
    if nv > 0:
        v_array = h[v_ind]
        v_array = NP.abs(v_array - h_min)
        v_max = NP.max(v_array)
        v_mean = NP.mean(v_array)
    else:
        v_max = 0.0
        v_mean = 0.0

    mean_viol_MPC_P.append(v_mean)
    max_viol_MPC_P.append(v_max)
    n_viol_MPC_P.append(nv)



    # NN results
    t_NN = data_NN[i][:,0]
    x_r_NN = data_NN[i][:,1:1+nx]
    x_e_NN = data_NN[i][:,1+nx:1+2*nx]
    u_NN = data_NN[i][:,1+2*nx:1+2*nx+nu]
    p_r_NN = data_NN[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_NN = data_NN[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    # compute tension
    v_0 = NP.reshape(p_r_NN[:,1],(-1,1))
    P_D = NP.square(v_0)/2.0
    E_0 = p_r_NN[0,0]
    E = E_0 - c_tilde * NP.square(u_NN)
    theta = NP.reshape(x_r_NN[:,0],(-1,1))
    phi = NP.reshape(x_r_NN[:,1],(-1,1))
    T_F = A * P_D * NP.square(NP.cos(theta)) * NP.sqrt(NP.square(E)+1) * (NP.cos(theta) * NP.cos(beta) + NP.sin(theta) * NP.sin(beta) * NP.sin(phi));
    TF_NN.append(NP.mean(T_F))

    # compute violations
    h = L_tether * NP.sin(theta) * NP.cos(phi)
    v_ind = h < h_min
    nv = sum(v_ind)
    if nv > 0:
        v_array = h[v_ind]
        v_array = NP.abs(v_array - h_min)
        v_max = NP.max(v_array)
        v_mean = NP.mean(v_array)
    else:
        v_max = 0.0
        v_mean = 0.0

    mean_viol_NN.append(v_mean)
    max_viol_NN.append(v_max)
    n_viol_NN.append(nv)

# comparison
TF_MPC = NP.vstack(TF_MPC)
mean_viol_MPC = NP.vstack(mean_viol_MPC)
max_viol_MPC = NP.vstack(max_viol_MPC)
n_viol_MPC = NP.vstack(n_viol_MPC)

TF_NN = NP.vstack(TF_NN)
mean_viol_NN = NP.vstack(mean_viol_NN)
max_viol_NN = NP.vstack(max_viol_NN)
n_viol_NN = NP.vstack(n_viol_NN)



# NN results
t_NN_P = data_NN_P[i][:,0]
x_r_NN_P = data_NN_P[i][:,1:1+nx]
x_e_NN_P = data_NN_P[i][:,1+nx:1+2*nx]
u_NN_P = data_NN_P[i][:,1+2*nx:1+2*nx+nu]
p_r_NN_P = data_NN_P[i][:,1+2*nx+nu:1+2*nx+nu+np]
p_e_NN_P = data_NN_P[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

# compute tension
v_0 = NP.reshape(p_r_NN_P[:,1],(-1,1))
P_D = NP.square(v_0)/2.0
E_0 = p_r_NN_P[0,0]
E = E_0 - c_tilde * NP.square(u_NN_P)
theta = NP.reshape(x_r_NN_P[:,0],(-1,1))
phi = NP.reshape(x_r_NN_P[:,1],(-1,1))
T_F = A * P_D * NP.square(NP.cos(theta)) * NP.sqrt(NP.square(E)+1) * (NP.cos(theta) * NP.cos(beta) + NP.sin(theta) * NP.sin(beta) * NP.sin(phi));
TF_NN_P.append(NP.mean(T_F))

# compute violations
h = L_tether * NP.sin(theta) * NP.cos(phi)
v_ind = h < h_min
nv = sum(v_ind)
if nv > 0:
    v_array = h[v_ind]
    v_array = NP.abs(v_array - h_min)
    v_max = NP.max(v_array)
    v_mean = NP.mean(v_array)
else:
    v_max = 0.0
    v_mean = 0.0

mean_viol_NN_P.append(v_mean)
max_viol_NN_P.append(v_max)
n_viol_NN_P.append(nv)

# comparison
TF_MPC = NP.vstack(TF_MPC)
mean_viol_MPC = NP.vstack(mean_viol_MPC)
max_viol_MPC = NP.vstack(max_viol_MPC)
n_viol_MPC = NP.vstack(n_viol_MPC)

TF_NN = NP.vstack(TF_NN)
mean_viol_NN = NP.vstack(mean_viol_NN)
max_viol_NN = NP.vstack(max_viol_NN)
n_viol_NN = NP.vstack(n_viol_NN)


print("\t\tMPC \t\tMPC_P \t\tNN \t\tNN_P")
print("Tension:\t%8.4f,\t%8.4f,\t%8.4f,\t%8.4f" % (NP.mean(TF_MPC), NP.mean(TF_NN), NP.mean(TF_MPC_P), NP.mean(TF_NN_P)))
print("V (mean):\t%8.4f,\t%8.4f,\t%8.4f,\t%8.4f" % (NP.mean(mean_viol_MPC), NP.mean(mean_viol_NN), NP.mean(mean_viol_MPC_P), NP.mean(mean_viol_NN_P)))
print("V (max):\t%8.4f,\t%8.4f,\t%8.4f,\t%8.4f" % (NP.max(max_viol_MPC), NP.max(max_viol_NN), NP.max(max_viol_MPC_P), NP.max(max_viol_NN_P)))
# print("Violation (number):\tMPC\t%8.4f,\tNN\t%8.4f" % (NP.mean(TF_MPC), NP.mean(TF_NN)))
