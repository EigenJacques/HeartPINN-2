#%%
from unittest import result
from numpy import array, tanh, zeros, product, ones, linspace, newaxis, squeeze, hstack, vstack
from numpy.random import random
from numpy.linalg import solve, norm

from pandas import DataFrame

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from time import time

import contextlib
import functools
import os
import time


import scipy as sp
from six.moves import urllib
from sklearn import preprocessing

# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()

import tensorflow_probability as tfp
#%%
class REG(keras.Model):
    def __init__(self, n_input, n_output, n_layers, n_nodes) -> None:
        super().__init__()

        # Creating PINN layers
        pinn_layers = {}
        pinn_layers[0] = layers.Dense(n_nodes, input_shape=(None, n_input),# input layer
                                    activation='linear', use_bias=True, name="input", dtype="float32", kernel_regularizer='l1')
        for i in range(n_layers):
            pinn_layers[1+i] = layers.Dense(n_nodes, activation="tanh", dtype="float32", kernel_regularizer='l1') # Hidden layers
            if i == n_layers-1:
                pinn_layers[1+i] = layers.Dense(n_output, activation="linear", use_bias=False, name="output", dtype="float32", kernel_regularizer='l1') # Output layer
                break

        self.pinn_layers = pinn_layers

    @tf.function
    def call(self, X): # Forward pass
        X = tf.convert_to_tensor(X, dtype="float32")
        for i in range(len(self.pinn_layers)):
            X = self.pinn_layers[i](X)
        return X

    @tf.function
    def loss(self, X, Y):
        l = tf.reduce_mean(tf.square( Y - self.call(X)[:,0] ))
        return l

class TRAIN():
    def __init__(self, model, X, Y):
        self.model = model
        self.X = X
        self.Y = Y

        self.L = list()
        self.optim = tf.keras.optimizers.Adam(learning_rate=1E-3)
        self.train_hist = list()

    def init_wb(self):
        self.model.call(self.X)

    @tf.function
    def update_adam(self):
        with tf.GradientTape() as tape:
            loss = self.model.loss(self.X, self.Y)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
    

    def update_de(self):

        iter_n = tf.Variable(0)

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.model.trainable_variables)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(shapes):
            n = product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.
            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.model.trainable_variables[i].assign(tf.reshape(param, shape))

        @tf.function
        def objective_fn(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.
            This function is created by function_factory.
            Args:
            params_1d [in]: a 1D tf.Tensor.
            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss = self.model.loss(self.X, self.Y)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss, self.model.trainable_variables)
            grads = tf.dynamic_stitch(idx, grads)

            # print out iteration & loss
            iter_n.assign_add(1)
            if iter_n%int(n_epoch/10) == 0:
                tf.print("Iter:", iter_n, "loss:", loss)
                # store loss value so we can retrieve later
                tf.py_function(self.train_hist.append, inp=[loss], Tout=[])

            return loss, grads

        @tf.function
        def min_fn():
            init_params = tf.dynamic_stitch(idx, self.model.trainable_variables)
            results = tfp.optimizer.lbfgs_minimize(
                    value_and_gradients_function=objective_fn,
                    initial_position=init_params,
                    max_iterations=tf.constant(n_epoch, dtype="int32"),
                    tolerance= tolerance)
            assign_new_model_parameters(results.position)
            tf.print(results.converged)
        min_fn()

    def update_bfgs(self, n_epoch, tolerance=1E-8):

        iter_n = tf.Variable(0)

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.model.trainable_variables)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(shapes):
            n = product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.
            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.model.trainable_variables[i].assign(tf.reshape(param, shape))

        @tf.function
        def objective_fn(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.
            This function is created by function_factory.
            Args:
            params_1d [in]: a 1D tf.Tensor.
            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss = self.model.loss(self.X, self.Y)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss, self.model.trainable_variables)
            grads = tf.dynamic_stitch(idx, grads)

            # print out iteration & loss
            iter_n.assign_add(1)
            if iter_n%int(n_epoch/10) == 0:
                tf.print("Iter:", iter_n, "loss:", loss)
                # store loss value so we can retrieve later
                tf.py_function(self.train_hist.append, inp=[loss], Tout=[])

            return loss, grads

        @tf.function
        def min_fn():
            init_params = tf.dynamic_stitch(idx, self.model.trainable_variables)
            results = tfp.optimizer.lbfgs_minimize(
                    value_and_gradients_function=objective_fn,
                    initial_position=init_params,
                    max_iterations=tf.constant(n_epoch, dtype="int32"),
                    tolerance= tolerance)
            assign_new_model_parameters(results.position)
            tf.print(results.converged)
        min_fn()


#%%
#================================================================================================================
# 1D cuve fit
#================================================================================================================

# @tf.function
# def f(X):
#     return (X-0.5)**2

# reg = REG(n_input=1, n_output=1, n_layers=3, n_nodes=2)
# Data_X = tf.convert_to_tensor(linspace(0,1,100)[:,newaxis], dtype="float32")
# # Data_X = tf.convert_to_tensor(random((100,1)), dtype="float32")
# Data_Y = tf.convert_to_tensor(f(Data_X.numpy())+0.1*random((100,1)), dtype="float32")

# fig, ax = plt.subplots(1,2)
# ax[0].scatter(Data_X,Data_Y, marker = ".")
# ax[1].scatter(Data_X,reg.call(Data_X), marker = ".")

# train = TRAIN(reg, Data_X, Data_Y)
# train.init_wb()
# for i in range(1_000):
#     train.update_adam()
#     if i%100 == 0:
#         print(i, train.model.loss(Data_X, Data_Y).numpy())
#         train.L.append(train.model.loss(Data_X, Data_Y).numpy())

# fig, ax = plt.subplots(1,2)
# ax[0].scatter(Data_X,Data_Y, marker = ".")
# ax[0].plot(Data_X,reg.call(Data_X), color = "red")
# ax[1].plot(range(len(train.L)), train.L, marker = ".")
# ax[1].set_yscale("log")
#%%
#================================================================================================================
# 2D curve fit
#================================================================================================================

def f(X):
    return 0.1*(X[:,0]-0.5)**2 + 0.1*(X[:,1]-0.5)**2

reg = REG(n_input=2, n_output=1, n_layers=3, n_nodes=5)
Data_X = tf.convert_to_tensor(random((1000,2)), dtype="float32")
Data_Y = tf.convert_to_tensor(f(Data_X.numpy()), dtype="float32")

fig, ax = plt.subplots(1,2)
ax[0].tricontourf(Data_X[:,0],Data_X[:,1], Data_Y)
ax[0].scatter(Data_X[:,0],Data_X[:,1], marker = ".")
ax[1].tricontourf(Data_X[:,0],Data_X[:,1], reg.call(Data_X).numpy()[:,0])

train = TRAIN(reg, Data_X, Data_Y)
train.init_wb()

# # Train some using differential evolution
# for i in range(2001):
#     train.update_de()
#     if i%100 == 0:
#         print(i, train.model.loss(Data_X, Data_Y).numpy())
#         train.train_hist.append(train.model.loss(Data_X, Data_Y).numpy())
        
# # Train some using adam
# for i in range(1000):
#     train.update_adam()
#     if i%10 == 0:
#         print(i, train.model.loss(Data_X, Data_Y).numpy())
#         train.train_hist.append(train.model.loss(Data_X, Data_Y).numpy())

# Train some using BFGS
train.update_bfgs(100)

fig = plt.figure(figsize = (5,5))
ax0 = fig.add_subplot(2,1,1)
im0 = ax0.tricontourf(Data_X[:,0],Data_X[:,1], Data_Y)

ax1 = fig.add_subplot(2,1,2)
im1 = ax1.tricontourf(Data_X[:,0],Data_X[:,1], reg.call(Data_X)[:,0])

divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

fig, ax = plt.subplots(1,1)
ax.plot(range(len(train.train_hist)), train.train_hist, marker = ".")
ax.set_yscale("log")

# %%
display(DataFrame(Data_X))
display(DataFrame(Data_Y))
display(DataFrame(pinn.call(Data_X)))
# %%
for i in range(2001):
    train.update_adam()
    if i%100 == 0:
        print(i, train.model.loss(Data_X, Data_Y).numpy())
        train.L.append(train.model.loss(Data_X, Data_Y).numpy())
fig = plt.figure(figsize = (5,5))
ax0 = fig.add_subplot(2,1,1)
im0 = ax0.tricontourf(Data_X[:,0],Data_X[:,1], Data_Y)

ax1 = fig.add_subplot(2,1,2)
im1 = ax1.tricontourf(Data_X[:,0],Data_X[:,1], reg.call(Data_X)[:,0])

divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

fig, ax = plt.subplots(1,1)
ax.plot(range(len(train.L)), train.L, marker = ".")
ax.set_yscale("log")
# %%
