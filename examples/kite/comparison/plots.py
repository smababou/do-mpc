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
pi = NP.pi
c_plot = NP.linspace(-70.0*2*pi/360,70.0*2*pi/360,100)
constraint = NP.sinh(h_min/(NP.cos(c_plot)*L_tether))*360/(2*pi)

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

# start plotting
plt.ion()
for i in range(n_batches):

    t_MPC = data_MPC[i][:,0]
    x_r_MPC = data_MPC[i][:,1:1+nx]
    x_e_MPC = data_MPC[i][:,1+nx:1+2*nx]
    u_MPC = data_MPC[i][:,1+2*nx:1+2*nx+nu]
    p_r_MPC = data_MPC[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_MPC = data_MPC[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    t_NN = data_NN[i][:,0]
    x_r_NN = data_NN[i][:,1:1+nx]
    x_e_NN = data_NN[i][:,1+nx:1+2*nx]
    u_NN = data_NN[i][:,1+2*nx:1+2*nx+nu]
    p_r_NN = data_NN[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_NN = data_NN[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    t_MPC_P = data_MPC_P[i][:,0]
    x_r_MPC_P = data_MPC_P[i][:,1:1+nx]
    x_e_MPC_P = data_MPC_P[i][:,1+nx:1+2*nx]
    u_MPC_P = data_MPC_P[i][:,1+2*nx:1+2*nx+nu]
    p_r_MPC_P = data_MPC_P[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_MPC_P = data_MPC_P[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    t_NN_P = data_NN_P[i][:,0]
    x_r_NN_P = data_NN_P[i][:,1:1+nx]
    x_e_NN_P = data_NN_P[i][:,1+nx:1+2*nx]
    u_NN_P = data_NN_P[i][:,1+2*nx:1+2*nx+nu]
    p_r_NN_P = data_NN_P[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e_NN_P = data_NN_P[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    fig, ax = plt.subplots()
    ax.plot(x_r_MPC[:,1]*180/pi,x_r_MPC[:,0]*180/pi,'-',label='MPC')
    ax.plot(x_r_NN[:,1]*180/pi,x_r_NN[:,0]*180/pi,'-',label='NN')
    ax.plot(x_r_MPC_P[:,1]*180/pi,x_r_MPC_P[:,0]*180/pi,'-',label='MPC with projection')
    ax.plot(x_r_NN_P[:,1]*180/pi,x_r_NN_P[:,0]*180/pi,'-',label='NN with projection')
    # ax.plot(x_e[:,1]*180/pi,x_e[:,0]*180/pi,'--',label='est')
    ax.plot(c_plot*360/(2*pi),constraint,c='#000000',label='con',linewidth=3.0)
    ax.set_title('kite position')
    ax.legend()
    ax.set_ylabel('$\Theta [rad]$')
    ax.set_xlabel('$\phi [rad]$')
    fig.align_ylabels(ax)

    # plt.figure()
    # plt.subplot(nx,1,1)
    # plt.plot(t,x_r_MPC[:,0])
    # plt.plot(t,x_r_NN[:,0])
    # plt.subplot(nx,1,2)
    # plt.plot(t,x_r_MPC[:,1])
    # plt.plot(t,x_r_NN[:,1])
    # plt.subplot(nx,1,3)
    # plt.plot(t,x_r_MPC[:,2],label='MPC')
    # plt.plot(t,x_r_NN[:,2],label='NN')

    # f, (ax1,ax2,ax3) = plt.subplots(nx, sharex=True)
    # ax1.plot(t_MPC,x_r_MPC[:,0])
    # ax1.plot(t_NN,x_r_NN[:,0])
    # ax2.plot(t_MPC,x_r_MPC[:,1])
    # ax2.plot(t_NN,x_r_NN[:,1])
    # ax3.plot(t_MPC,x_r_MPC[:,2],label='MPC')
    # ax3.plot(t_NN,x_r_NN[:,2],label='NN')
    # ax3.legend()

    # plt.figure()
    # plt.subplot(np,1,1)
    # plt.plot(t,p_r[:,0])
    # plt.plot(t,p_e[:,0])
    # plt.subplot(np,1,2)
    # plt.plot(t,p_r[:,1])
    # plt.plot(t,p_e[:,1])

    # fig, ax = plt.subplots()
    # ax.plot(t_MPC,u_MPC[:,0],label='MPC')
    # ax.plot(t_NN,u_NN[:,0],label='NN')
    # ax.legend()



# print('Press any button to end script ...')
plt.tight_layout()
plt.show()

input("--- Press any button to close all figures and exit the script ---")

plt.close('all')

# input()
