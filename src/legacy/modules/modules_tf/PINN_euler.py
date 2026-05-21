import wandb

from numpy import array

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/media/jflinux_main/HDD/Studies/Studies_MEngM/Navorsing/HeartPINN/Phase_2/modules/')
# from VDense import VDense

class PINN(keras.Model):
    def __init__(self, n_input, n_output, n_layers, n_nodes, precision="float32") -> None:
        super().__init__()

        # Creating PINN layers
        pinn_layers = {}
        pinn_layers[0] = layers.Dense(units=n_nodes, input_shape=(None, n_input),# input layer
                                    activation='linear', use_bias=False, name="input", dtype=precision)
        for i in range(n_layers+1):
            pinn_layers[1+i] = layers.Dense(n_nodes, activation="tanh", dtype=precision) # Hidden layers
            if i == n_layers-1:
                pinn_layers[1+i] = layers.Dense(n_output, activation="linear", use_bias=False, name="output", dtype=precision) # Output layer
                break

        self.pinn_layers = pinn_layers
        self.loss_weighting = [1.0,1,1,1,1]
        self.precision = precision

    @tf.function
    def call(self, X): # Forward pass to return Phi, P
        X = tf.convert_to_tensor(X, dtype=self.precision)
        for i in range(len(self.pinn_layers)):
            X = self.pinn_layers[i](X)
        return X

    def predict_vel(self, X): # Forward pass to return: u, v
        X = tf.convert_to_tensor(X, dtype=self.precision)
        with tf.GradientTape() as tape:
            tape.watch(X)
            Phi = self.call(X)[:, 0:1]
            dPhi = tape.gradient(Phi, X)
            u = dPhi[:, 0:1] 
            v = dPhi[:, 1:2]

        return array(u), array(v)

    @tf.function
    def loss_fn(self, X_fn): # Function loss
        X_fn = tf.convert_to_tensor(X_fn, dtype=self.precision)
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(X_fn)
            Phi = self.call(X_fn)[:, 0:1]
            dPhi = tape.gradient(Phi, X_fn)
            Phi_x = dPhi[:, 0:1] 
            Phi_y = dPhi[:, 1:2] 
        Phi_xx = tape.gradient(Phi_x, X_fn)[:, 0:1] 
        Phi_yy = tape.gradient(Phi_y, X_fn)[:, 1:2]
        del tape
        E = tf.reduce_mean(tf.square(Phi_xx + Phi_yy))

        return E
    
    @tf.function
    def loss_wall(self, X_wall, N_wall): # BC loss at wall
        X_wall = tf.convert_to_tensor(X_wall, dtype=self.precision)
        N_wall = tf.convert_to_tensor(N_wall, dtype=self.precision)

        with tf.GradientTape() as tape:
            tape.watch(X_wall)
            Phi = self.call(X_wall)[:, 0:1]
            dPhi = tape.gradient(Phi, X_wall)

        E = 0.0
        PhiN = tf.linalg.tensor_diag_part(tf.tensordot(N_wall, tf.transpose(dPhi), 1)) # Inefficient and can be improved
        E = E + tf.reduce_mean(tf.square(PhiN - 0.0))   # velocity normal to wall

        return E
    
    @tf.function
    def loss_in(self, X_in): # Bc loss at inlet
        X_in = tf.convert_to_tensor(X_in, dtype=self.precision)
        with tf.GradientTape() as tape:
            tape.watch(X_in)
            Phi = self.call(X_in)[:, 0:1]
            dPhi = tape.gradient(Phi, X_in)
            Phi_x = dPhi[:, 0:1] 
            Phi_y = dPhi[:, 1:2]

        E = 0.0
        E = E + tf.reduce_mean(tf.square(Phi_x - 2.0))   # u velocity at inlet
        E = E + tf.reduce_mean(tf.square(Phi_y - 0.0))   # v velocity at inlet

        return E
    
    @tf.function
    def loss_out(self, X_out): # Bc loss at outlet
        X_out = tf.convert_to_tensor(X_out, dtype=self.precision)
        with tf.GradientTape() as tape:
            tape.watch(X_out)
            Phi = self.call(X_out)[:, 0:1]
            dPhi = tape.gradient(Phi, X_out)
            Phi_x = dPhi[:, 0:1] 
            Phi_y = dPhi[:, 1:2]

        E = 0.0
        E = E + tf.reduce_mean(tf.square(Phi_x - 2.0))   # u velocity at outlet
        E = E + tf.reduce_mean(tf.square(Phi_y - 0.0))   # v velocity at outlet

        return E
    
    @tf.function
    def loss_data(self, Data): # Data loss
        X_data = Data[0]
        Y_data = Data[1]
        with tf.GradientTape() as tape:
            tape.watch(X_data)
            Phi = self.call(X_data)[:, 0:1]
            dPhi = tape.gradient(Phi, X_data)
            Phi_x = dPhi[:, 0:1] 
            Phi_y = dPhi[:, 1:2]

        E = 0.0
        E = E + tf.reduce_mean(tf.square(Phi_x - Y_data[:,0]))   # u velocity at outlet
        E = E + tf.reduce_mean(tf.square(Phi_y - Y_data[:,1]))   # v velocity at outlet

        return E

    def update_weighting(self, weights):
        self.loss_weighting = weights

    # @tf.function # This cannot be a graph since loss 
    def loss(self, X_fn, X_in, X_out, X_wall, Data, N_wall): # Total loss
        X_fn = tf.convert_to_tensor(X_fn, dtype=self.precision)
        X_in = tf.convert_to_tensor(X_in, dtype=self.precision)
        X_out = tf.convert_to_tensor(X_out, dtype=self.precision)
        X_wall = tf.convert_to_tensor(X_wall, dtype=self.precision)
        Data = Data
        N_wall = tf.convert_to_tensor(N_wall, dtype=self.precision)

        E =     self.loss_fn(X_fn)*self.loss_weighting[0]
        E = E + self.loss_in(X_in)*self.loss_weighting[1]
        E = E + self.loss_out(X_out)*self.loss_weighting[2]
        E = E + self.loss_wall(X_wall, N_wall)*self.loss_weighting[3]
        E = E + self.loss_data(Data)*self.loss_weighting[4]

        return E
