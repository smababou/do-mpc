import numpy as NP
import pdb

# parameters
n_batches = 300
pi = NP.pi

param_table = NP.resize(NP.array([]),(0, 5))

for i in range(n_batches):

    # choose the real parameters
    E_batch = NP.random.uniform(4.0,6.0)
    c_batch = NP.random.uniform(0.005,0.04)

    w_mean = 8.0 + NP.random.uniform() * 4.0
    var_t = 0.05 + NP.random.uniform() * 0.1
    w_lb = NP.array([7.0])
    w_ub = NP.array([13.0])
    w_amp_max = NP.minimum(NP.abs(w_ub-w_mean),NP.abs(w_lb-w_mean))
    w_amp = NP.random.uniform() * w_amp_max
    w_shift = NP.random.uniform() * 2.0 * pi

    new_setting = NP.reshape(NP.array([E_batch,w_mean,var_t,w_amp,w_shift]),(1,-1))
    param_table = NP.append(param_table,new_setting,axis=0)

# save parameters
NP.save('table_of_parameter_settings',param_table)
