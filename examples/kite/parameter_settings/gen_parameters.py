import numpy as NP
import pdb

# parameters
n_batches = 100
pi = NP.pi

param_table = NP.resize(NP.array([]),(0, 5))

for i in range(n_batches):

    # choose the real parameters
    E_batch = NP.random.uniform(3.0,7.0)
    c_batch = NP.random.uniform(0.005,0.04)

    w_mean = 7.0 + NP.random.uniform() * 6.0
    var_t = 0.05 + NP.random.uniform() * 0.1
    w_lb = NP.array([5.0])
    w_ub = NP.array([15.0])
    w_amp_max = NP.minimum(NP.abs(w_ub-w_mean),NP.abs(w_lb-w_mean))
    w_amp = w_amp_max
    w_shift = NP.random.uniform() * 2.0 * pi

    new_setting = NP.reshape(NP.array([E_batch,w_mean,var_t,w_amp,w_shift]),(1,-1))
    param_table = NP.append(param_table,new_setting,axis=0)

# save parameters
NP.save('table_of_parameter_settings_extreme',param_table)
