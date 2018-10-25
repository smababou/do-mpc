#load needed modules
import numpy as NP
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb

plt.rcParams.update({'font.size': 12, 'lines.linewidth' : 2.0,'svg.fonttype': 'none',
    'axes.unicode_minus':False, })

# parameters
path_to_data = '../data/2_uncertainties_NN'
# path_to_data = '../data'
offset = 45
n_batches = 5

# sizes
nx = 3
nu = 1
np = 2

# additional params
L_tether = 420.0
h_min = 100.0
pi = NP.pi
c_plot = NP.linspace(-70.0*2*pi/360,70.0*2*pi/360,100)
constraint = NP.sinh(h_min/(NP.cos(c_plot)*L_tether))*360/(2*pi)

# load data
data = []
for i in range(n_batches):
    data.append(NP.load(path_to_data + "/data_batch_" + str(i) + ".npy"))

# start plotting
plt.ion()
for i in range(n_batches):

    t = data[i][:,0]
    x_r = data[i][:,1:1+nx]
    x_e = data[i][:,1+nx:1+2*nx]
    u = data[i][:,1+2*nx:1+2*nx+nu]
    p_r = data[i][:,1+2*nx+nu:1+2*nx+nu+np]
    p_e = data[i][:,1+2*nx+nu+np:1+2*nx+nu+2*np]

    # plt.figure(1+i*nx)
    fig, ax = plt.subplots()
    ax.plot(x_r[:,1]*180/pi,x_r[:,0]*180/pi,'-',label='real')
    ax.plot(x_e[:,1]*180/pi,x_e[:,0]*180/pi,'--',label='est')
    ax.plot(c_plot*360/(2*pi),constraint,c='#000000',label='con',linewidth=3.0)
    ax.set_title('kite position')
    ax.legend()
    ax.set_ylabel('$\Theta$')
    ax.set_xlabel('$\phi [rad]$')
    # fig.align_ylabels(ax)

    # fig, ax = plt.subplot(nx,1,nx)
    plt.figure()
    plt.subplot(nx,1,1)
    plt.plot(t,x_r[:,0])
    plt.plot(t,x_e[:,0])
    plt.subplot(nx,1,2)
    plt.plot(t,x_r[:,1])
    plt.plot(t,x_e[:,1])
    plt.subplot(nx,1,3)
    plt.plot(t,x_r[:,2])
    plt.plot(t,x_e[:,2])
    #
    # plt.figure()
    # plt.subplot(np,1,1)
    # plt.plot(t,p_r[:,0])
    # plt.plot(t,p_e[:,0])
    # plt.subplot(np,1,2)
    # plt.plot(t,p_r[:,1])
    # plt.plot(t,p_e[:,1])



# print('Press any button to end script ...')
plt.tight_layout()
plt.show()

input("--- Press any button to close all figures and exit the script ---")

plt.close('all')

# input()
