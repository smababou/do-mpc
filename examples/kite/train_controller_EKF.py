import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import SGD
from keras import regularizers
#from keras.utils import plot_model
import numpy as NP
import os.path
import pdb

# Load data
path_to_data_1 = 'data/2_uncertainties_new/'
n_batches_train_1 = 170
n_batches_test_1 = 5
n_batches_train_test_1 = n_batches_train_1 + n_batches_test_1

# path_to_data_2 = '../training_data/data/N80_deltah/'
# n_batches_train_2 = 80
# n_batches_test_2 = 20
# n_batches_train_test_2 = n_batches_train_2 + n_batches_test_2

# Initialize vectors
states = []
controls = []
raw_data = []

nx = 3
nu = 1
np = 2
nt = 1

n_layers = 6
n_neurons = 10
act = 'relu'
output_act = 'linear'

for i in range(n_batches_train_test_1):
    # Format is: |time|states|states_est|controls|params|params_est|
    raw_data.append(NP.load(path_to_data_1+ "data_batch_" + str(i+0) + ".npy"))

# for i in range(n_batches_train_test_2):
#     # Format is: |time|states|controls|params|
#     raw_data.append(NP.load(path_to_data_2+ "data_batch_" + str(i+0) + ".npy"))

for i in range(n_batches_train_test_1):
    # remove the offset of one position in the state-control vector
    states.append(raw_data[i][0:-1,nt+nx:nt+2*nx])
    controls.append(raw_data[i][1:,nt+2*nx:nt+2*nx+nu])

# training data
x_train_1 = NP.vstack(states[0:n_batches_train_1])
# x_train_2 = NP.vstack(states[n_batches_train_test_1:n_batches_train_test_1+n_batches_train_2])
# x_train = NP.vstack([x_train_1,x_train_2])

y_train_1 = NP.vstack(controls[0:n_batches_train_1])
# y_train_2 = NP.vstack(controls[n_batches_train_test_1:n_batches_train_test_1+n_batches_train_2])
# y_train = NP.vstack([y_train_1,y_train_2])

# test data
x_test_1 = NP.vstack(states[n_batches_train_1:n_batches_train_test_1])
# x_test_2 = NP.vstack(states[n_batches_train_test_1+n_batches_train_2:n_batches_train_test_1+n_batches_train_test_2])
# x_test = NP.vstack([x_test_1,x_test_2])

y_test_1 = NP.vstack(controls[n_batches_train_1:n_batches_train_test_1])
# y_test_2 = NP.vstack(controls[n_batches_train_test_1+n_batches_train_2:n_batches_train_test_1+n_batches_train_test_2])
# y_test = NP.vstack([y_test_1,y_test_2])

u_lb = NP.array([-10.0])
u_dif = NP.array([20.0])
u_ub = u_lb + u_dif

x_lb = NP.array([0.15, -1.25,-3.3])
x_ub = NP.array([1.0, 1.3, 3.3])

# Scale the outputs between 0 and 1 based on input bounds for better loss management
y_train_1 = (y_train_1 - u_lb) / (u_ub - u_lb)

y_test_1  = (y_test_1  - u_lb) / (u_ub - u_lb)
x_train_1 = (x_train_1 - x_lb) / (x_ub - x_lb)
x_test_1  = (x_test_1  - x_lb) / (x_ub - x_lb)

main_input = Input(shape=(nx,))

l1_pen = 0.0000
x = Dense(n_neurons, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(main_input)

for i in range(n_layers-1):
    x = Dense(n_neurons, activation = act, kernel_regularizer=regularizers.l2(l1_pen))(x)

main_output = Dense(nu, activation = output_act, kernel_regularizer=regularizers.l2(l1_pen))(x)

model = Model(inputs = main_input, outputs = main_output)
model.compile(loss='mse',optimizer='adam')
model.fit(x_train_1, y_train_1,epochs=100,batch_size=100)
pdb.set_trace()
score = model.evaluate(x_test_1, y_test_1, batch_size=128)
print("The test score is: ")
print(score)

# If wanted save model
saveloc = "controller_"
n_save = 0
while(n_save < 1000): # to avoid overwriting
    if os.path.isfile(saveloc+str(n_save)+'.json'):
      n_save += 1
    else:
      saveloc += str(n_save)
      break
model_json = model.to_json()
with open(saveloc+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(saveloc+".h5")
print("Saved model " + saveloc + ".~ to disk")
