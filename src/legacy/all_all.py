#%%
#=============================================================
# Import modules
#=============================================================
import tensorflow as tf
from tensorflow.keras import layers, Model, mixed_precision

from numpy.random import random
import numpy as np

from scipy import optimize

import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import json

import time

from scipy import stats

import sys
import os

import pickle

# Add path to local modules imports
sys.path.insert(1, './modules/modules_tf')
from save_and_recover3 import recover_model3, save_model3
from heartvalve_half import AoValveHalf

# tf.config.run_functions_eagerly(True)
#=============================================================
# Definitions
#=============================================================

def check_compat(methods):
    """ Check if the methods are compatible.
        Return the output dimensions given the chosen methods"""

    # Check compatibility of methods
    if methods["hard"] and methods["sp"]:
        raise ValueError(f"Cannot use methods hard and sp together. (fundamentally incompatible)")
    
    if methods["hard"] and methods["gcn"]:
        raise ValueError(f"Cannot use methods hard and gcn together (Not implemented yet)")
    
    # Return output dimensions
    if methods["sp"]:
        return 2
    else:
        return 3

def import_fvm_solution(cons:dict, path:str, npoints:int, prec=tf.float32, unitsin="mm"):
    ''' Import validation data from a fvm solution. '''

    if unitsin == "mm":
        scale_out = 1
    elif unitsin == "m":
        scale_out = 1000
    else:
        raise ValueError(f"Units {unitsin} not recognized.")

    Data        = pd.read_csv(path, sep=",", header=0)

    nsamples    = Data.shape[0]
    irand       = (nsamples*random(npoints)).astype("int")
    X_in        = Data[[" X [ m ]", " Y [ m ]"]].iloc[list(irand)]
    X_out       = Data[[" Velocity u [ m s^-1 ]", " Velocity v [ m s^-1 ]",
                        " Pressure [ Pa ]"]].iloc[list(irand)]

    # X_in    = X_in/cons["L"]*scale_out
    # X_out   = X_out*np.tile(np.array([[1/cons["U_inf"], 1/cons["U_inf"],
    #                                    1/(cons["rho"]*cons["U_inf"]**2)]]), (npoints,1))
    
    return [tf.convert_to_tensor(np.array(X_in), dtype=prec),
            tf.convert_to_tensor(np.array(X_out), dtype=prec)]


class GCN(layers.Layer):
    def __init__(self, n_nodes, activation = "tanh", use_bias = True, nettype = True):
        super(GCN, self).__init__()
        """ Return a keras graph convolutional neural network layer object.
    
        Parameters
        ----------
        n_nodes : int
            Number of nodes in the layer output
        activation : str
            Activation function to use.
        use_bias : bool
            Whether to use bias or not.

        Notes
        ----- 

        """ 
        self.n_nodes = n_nodes
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.nettype = nettype

        if nettype:
            self.call = self.call_gcn
        else:
            self.call = self.call_dense

    def build(self, input_shape):
        """ Initialize the layers. """

        # Weights Matrix of shape (n_in, n_out)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.n_nodes),
            initializer=self.kernel_initializer,
            trainable=True
        )

        # Bias of shape (n_out)
        if self.use_bias == True:
            self.b = self.add_weight(
                shape=(1, self.n_nodes), 
                initializer=self.bias_initializer, 
                trainable=True
            )
        elif self.use_bias == False:
            self.b = 0.0

        # Activation
        if self.activation == "tanh":
            self.activation = tf.keras.activations.tanh
        elif self.activation == "linear":
            self.activation = tf.keras.activations.linear
        else:
            pass

    @tf.function
    def call_gcn(self, X, A_hat, training):
        """ Perform forward pass through the layer . 
    
        Parameters
        ----------
        inputs : list(tf.Tensor)
            Input list with tensors (X (inputs) and A_hat (adjacency)).
        degree: tf.Variable
            Degree matrix of the graph

        Returns
        -------
        Y : tf.Tensor : shape (n_samples, n_out)
            Output tensor.

        """

        print("retrace call_gcn")

        Y = tf.sparse.sparse_dense_matmul(A_hat, X, adjoint_a=False, adjoint_b=False)
        Y = tf.tensordot(Y, self.w, axes=1)
        Y = self.activation(Y) + self.b

        return Y
    
    @tf.function
    def call_dense(self, X, A_hat, training):
        """ Perform forward pass through the layer . 
    
        Parameters
        ----------
        inputs : list(tf.Tensor)
            Input list with tensors (X (inputs) and A_hat (adjacency)).
        degree: tf.Variable
            Degree matrix of the graph

        Returns
        -------
        Y : tf.Tensor : shape (n_samples, n_out)
            Output tensor.

        """

        print("retrace call_dense")

        Y = tf.tensordot(X, self.w, axes=1)
        Y = self.activation(Y) + self.b

        return Y
    
 
class Network(Model):
    """ Neural Network class for the PINN. """

    def __init__(self, nodes:int, input_n:int, output_n:int, hiddenlayers:int,
                 fourier_sigma:float, methods:dict, prec=tf.float32) -> None:
        super().__init__()
            
        self.call_shape = [[10,2,0],[10,10,1]]

        self.prec       = prec

        self.nodes          = nodes
        self.input_n        = input_n
        self.output_n       = output_n
        self.hiddenlayers   = hiddenlayers
        self.fourier_sigma  = fourier_sigma

        self.initializer         = tf.keras.initializers.GlorotNormal(np.random.randint(0,100000))
        self.scaling_layer_w     = tf.Variable(np.diag(1/np.ones(input_n)), dtype=prec, trainable=False)
        self.scaling_layer_b     = tf.Variable(np.zeros(input_n), dtype=prec, trainable=False)
        self.fourier_initializer = tf.random_normal_initializer(mean=0.0, stddev=fourier_sigma, seed=None)

        if methods["fourier"]:
            self.indim = input_n*2
        else:
            self.indim = input_n

        if methods["sipinn"]:
            self.sipinn_XMN = self.sipinn_MN
        else:
            self.sipinn_XMN = self.sipinn_pass

        # Creating Pinn layers

            # Fourier layer
        self.pinn_fourier   = layers.Dense(units=nodes,activation="linear", use_bias=False, dtype=prec,
                                        kernel_initializer=self.fourier_initializer, trainable=False)

            # SIPINN layers
        self.pinn_M         = layers.Dense(units=nodes,activation='tanh', use_bias=True, name="M", kernel_initializer=self.initializer, dtype=prec)
        self.pinn_N         = layers.Dense(units=nodes,activation='tanh', use_bias=True, name="N", kernel_initializer=self.initializer, dtype=prec)

            # Input layer"
        self.pinn_in        = GCN(nodes, activation = "tanh", use_bias = True, nettype = methods["gcn"])
        
            # Hidden layers
        pinn_layers = []
        for i in range(hiddenlayers):
            pinn_layers.append(layers.Dense(nodes, activation="tanh", dtype=prec,
                                            kernel_initializer=self.initializer))
        self.pinn_hidden = pinn_layers

            # Output layer
        self.pinn_out   = layers.Dense(output_n, activation="linear", use_bias=False,
                                    name="output", kernel_initializer =self.initializer,
                                    dtype=prec)

        self.forward_in     = []
        self.forward_hidden = []

        #fourier
        if methods["fourier"]:
            self.forward_in.append(lambda h, A_hat, XM, XN: self.fourier(h))

        #input
        self.forward_in.append(lambda h, A_hat, XM, XN: self.pinn_in(h, A_hat))
        
        #sipinn
        if methods["sipinn"]:
            #in
            self.forward_in.append(lambda h, A_hat, XM, XN: self.sipinn_in(h, XM, XN))

            #hidden
            for i in range(hiddenlayers):
                self.forward_hidden.append(lambda h, XM, XN: self.sipinn(self.pinn_hidden[i](h), XM, XN))
        else:
            #in
            #nothing

            #hidden
            for i in range(hiddenlayers):
                self.forward_hidden.append(lambda h, XM, XN: self.pinn_hidden[i](h))

    @tf.function
    def sipinn_MN(self, h):
        XM = self.pinn_M(h)
        XN = self.pinn_N(h)
        return XM, XN

    @tf.function
    def sipinn_pass(self, h):
        return tf.constant(np.empty((0))), tf.constant(np.empty((0)), dtype=tf.float32)
        
    # @tf.function # Triggers retracing
    def sipinn_in(self, h, XM, XN):
        # print("retrace sipinn_in")
        h = tf.math.multiply(h, XM) + XN
        return h
    
    # @tf.function # Triggers retracing
    def sipinn(self, h, XM, XN):
        # print("retrace sipinn")
        h = tf.math.multiply(h, XM) + XN
        return h
    
    @tf.function
    def fourier(self, h):
        print("retrace fourier")
        psi_s = tf.math.sin(2.0*np.pi*self.pinn_fourier(h))
        psi_c = tf.math.cos(2.0*np.pi*self.pinn_fourier(h))
        h     = tf.concat([psi_s, psi_c], axis=1)
        return h

    @tf.function
    def call(self, X, A_hat):
        """ Forward pass of the PINN."""

        print("retrace call")

        # Scaling layer
        h = tf.tensordot(X,self.scaling_layer_w,axes=1) - self.scaling_layer_b

        XM, XN = self.sipinn_XMN(h)

        for i in range(len(self.forward_in)):
            h = self.forward_in[i](h, A_hat, XM, XN)
        for i in range(self.hiddenlayers):
            h = self.forward_hidden[i](h, XM, XN)
        h = self.pinn_out(h)

        return h

    @tf.function
    def predict(self, X, A_hat):
        """ Forward pass of the PINN, with outputs separated into entries of a list."""
        
        print("retrace predict")

        return self.call(X, A_hat)
    

class Pinn:
    """ Class that contains all the functions necessary to assemble the physics informed loss function. """

    def __init__(self, net, cons, methods:dict, active_bcs:dict) -> None:
        self.net  = net
        self.cons = cons
        
        self.train_hist = []

        if methods["gpinn"]:
            self.loss_fn = self.loss_fn_g
        else:
            self.loss_fn = self.loss_fn_ng

        if methods["hard"]:
            self.hard_transform = self.hard_transform_h
            self.loss_tot = self.loss_tot_hard
        else:
            self.hard_transform = self.hard_transform_nh
            self.loss_tot = self.loss_tot_soft

        if methods["sp"]:
            self.formulation = self.formulation_sp
        else:
            self.formulation = self.formulation_vp

        if methods["turb"]:
            self.turbmodel = self.turbmodel_ml
        else:
            self.turbmodel = self.turbmodel_lam

    def inletprofile(self, X, u_avg, h):
        """ Returns the inlet profile for the velocity. """

        u_profile = u_avg*1.5*4/((h/self.cons["L"])**2)*(X[:,1:2]+0)*(h/self.cons["L"]-X[:,1:2])
        return u_profile
    
    def domain(self, X, F, kind:str):
        """ Returns the boundary distance functions needed for the hard constraint method, 
            with their pre-computed gradients. 
        """

        @tf.custom_gradient
        def func(X):
            def gradient(x):
                g = tf.math.multiply(tf.tile(x, (1,2)),F[f"d{kind}"])
                return g
            return F[f"{kind}"], gradient
        return func(X)

    @tf.function
    def hard_transform_h(self, X, u, v, p, du, dv, dp, F):
        print("retrace hard h")

        """ Transforms the traial solution in order to constrain the solution space to exactly satisfy the boundary conditions."""

        F_in   = self.domain(X, F, "F_in")
        F_out  = self.domain(X, F, "F_out")
        F_wall = self.domain(X, F, "F_wall")
        # F_sym  = self.domain(X, F, "F_sym")

        w_in   = self.domain(X, F, "w_in")
        w_out  = self.domain(X, F, "w_out")
        w_wall = self.domain(X, F, "w_wall")
        # w_sym  = self.domain(X, F, "w_sym")

        u_avg       = 1.0
        u_profile   = self.inletprofile(X, u_avg, h)
        u_in        = u_profile
        u_wall      = 0
        # u_sym       = (u + F_sym*du[:,1:2])

        v_in    = 0
        v_wall  = 0
        v_sym   = 0

        p_out   = 0
        # p_sym   = (p + F_sym*dp[:,1:2])

        u = w_in*u_in + w_wall*u_wall + F_wall*F_in*u
        v = w_in*v_in + w_wall*v_wall + F_in*F_wall*v
        p = w_out*p_out + F_out*p
        # u = w_in*u_in + w_wall*u_wall + w_sym*u_sym + F_wall*F_in*F_sym**2*u
        # v = w_in*v_in + w_wall*v_wall + w_sym*v_sym + F_in*F_wall*F_sym**2*v
        # p = w_out*p_out + w_sym*p_sym + F_out*F_sym**2*p

        return u, v, p
    
    @tf.function
    def hard_transform_nh(self, X, u, v, p, du, dv, dp, F):
        """ Transforms the traial solution in order to constrain the solution space to exactly satisfy the boundary conditions."""

        print("retrace hard nh")

        return u, v, p

    # @tf.function # Triggers retracing
    def formulation_vp(self, X, A_hat, tape):
        psi = self.net.predict(X, A_hat)    
        # print("retrace vp")
        return psi[:,0:1], psi[:,1:2], psi[:,2:3]
    
    # @tf.function # Triggers retracing
    def formulation_sp(self, X, A_hat, tape):
        psi = self.net.predict(X, A_hat)    
        s, p = psi[:,0:1], psi[:,1:2]
        ds = tape.gradient(s, X)

        u = ds[:,1:2]
        v = -ds[:,0:1]
        # print("retrace sp")

        return u, v, p
    
    def turbmodel_ml(self, D, dmax, du_x, dv_y, du_y, dv_x):
        """Compute turbulent viscosity.

        Notes:
        Zero equation turbulence model from simnet paper.
        Non-dimensional parameters in
        Dimensional dynamic viscosity out. """

        rho, L, U_inf = self.cons["rho"], self.cons["L"], self.cons["U_inf"]

        # Dimensionalize
        G = 2*(tf.square(du_x) + tf.square(dv_y)) + tf.square(du_y + dv_x)
        G = G*(U_inf/L)**2
        D, dmax = D*L, dmax*L

        l_m = tf.math.minimum(0.419*D, 0.09*dmax)
        mu_t = rho*l_m**2*tf.sqrt(G)

        return mu_t
    
    def turbmodel_lam(self, D, dmax, du_x, dv_y, du_y, dv_x):
        """Compute turbulent viscosity.

        Notes:
        Zero equation turbulence model from simnet paper.
        Non-dimensional parameters in
        Dimensional dynamic viscosity out. """

        return 0.0

    @tf.function
    def loss_fn_ng(self, Xs, A_hat):
        """ Returns the mean magnitude of violation of the trial solution of the Navier-Stokes equations."""

        rho, mu, U_inf, L, dmax, alpha = self.cons["rho"], self.cons["mu"], self.cons["U_inf"], self.cons["L"], self.cons['dmax'], self.cons['alpha']
        X = Xs['X']
        D = Xs['D']

        print("retrace loss fn ng")

        with tf.GradientTape(persistent = True) as tape:
            tape.watch(X)

            u, v, p = self.formulation(X, A_hat, tape)

            du = tape.gradient(u, X)
            dv = tape.gradient(v, X)
            dp = tape.gradient(p, X)

            u, v, p = self.hard_transform(X, u, v, p, du, dv, dp, Xs)

            du = tape.gradient(u, X)
            dv = tape.gradient(v, X)
            dp = tape.gradient(p, X)

            du_x = tape.gradient(u, X)[:,0:1]
            dv_x = tape.gradient(v, X)[:,0:1]
            dp_x = tape.gradient(p, X)[:,0:1]
            du_y = tape.gradient(u, X)[:,1:2]
            dv_y = tape.gradient(v, X)[:,1:2]
            dp_y = tape.gradient(p, X)[:,1:2]

        ddu_xx = tape.gradient(du_x, X)[:,0:1]
        ddu_yy = tape.gradient(du_y, X)[:,1:2]
        ddv_xx = tape.gradient(dv_x, X)[:,0:1]
        ddv_yy = tape.gradient(dv_y, X)[:,1:2]

        del tape

        mu_t = self.turbmodel(D, dmax, du_x, dv_y, du_y, dv_x)

        conti = du_x + dv_y
        mom_x = u*du_x + v*du_y + alpha*dp_x - (mu + mu_t)/(U_inf*rho*L)*(ddu_xx + ddu_yy)
        mom_y = u*dv_x + v*dv_y + alpha*dp_y - (mu + mu_t)/(U_inf*rho*L)*(ddv_xx + ddv_yy)
        
        E_cont = tf.reduce_mean(tf.square(conti))
        E_mom = tf.reduce_mean(tf.square(mom_x)) + tf.reduce_mean(tf.square(mom_y))

        return E_cont, E_mom/2
    
    @tf.function
    def loss_fn_g(self, Xs, A_hat):
        """ Returns the mean magnitude of violation of the trial solution of the Navier-Stokes equations."""

        print("retrace loss fn g")

        rho, mu, U_inf, L, dmax, alpha = self.cons["rho"], self.cons["mu"], self.cons["U_inf"], self.cons["L"], self.cons['dmax'], self.cons['alpha']
        X = Xs['X']
        D = Xs['D']

        with tf.GradientTape(persistent = True) as tape:
            tape.watch(X)

            u, v, p = self.formulation(X, A_hat, tape)

            du = tape.gradient(u, X)
            dv = tape.gradient(v, X)
            dp = tape.gradient(p, X)

            u, v, p = self.hard_transform(X, u, v, p, du, dv, dp, Xs)

            du = tape.gradient(u, X)
            dv = tape.gradient(v, X)
            dp = tape.gradient(p, X)

            du_x = tape.gradient(u, X)[:,0:1]
            dv_x = tape.gradient(v, X)[:,0:1]
            dp_x = tape.gradient(p, X)[:,0:1]
            du_y = tape.gradient(u, X)[:,1:2]
            dv_y = tape.gradient(v, X)[:,1:2]
            dp_y = tape.gradient(p, X)[:,1:2]

            ddu_xx = tape.gradient(du_x, X)[:,0:1]
            ddu_yy = tape.gradient(du_y, X)[:,1:2]
            ddv_xx = tape.gradient(dv_x, X)[:,0:1]
            ddv_yy = tape.gradient(dv_y, X)[:,1:2]

            mu_t = self.turbmodel(D, dmax, du_x, dv_y, du_y, dv_x)

            conti = du_x + dv_y
            mom_x = u*du_x + v*du_y + alpha*dp_x - (mu + mu_t)/(U_inf*rho*L)*(ddu_xx + ddu_yy)
            mom_y = u*dv_x + v*dv_y + alpha*dp_y - (mu + mu_t)/(U_inf*rho*L)*(ddv_xx + ddv_yy)

            overall = tf.square(conti) + tf.square(mom_x) + tf.square(mom_y)
        d_overall = tape.gradient(overall, X)
        del tape

        E_cont = tf.reduce_mean(tf.square(conti))
        E_mom = tf.reduce_mean(tf.square(mom_x)) + tf.reduce_mean(tf.square(mom_y)) + tf.reduce_mean(tf.square(d_overall))

        return E_cont, E_mom/2
    
    @tf.function
    def loss_in(self, Xs):

        print("retrace loss_in")

        X = Xs['X']
        A_hat = Xs["A_hat"]

        u_avg = 1.0
        u_profile = self.inletprofile(X, u_avg, h)
        
        with tf.GradientTape() as tape:
            tape.watch(X)
            u, v, _ = self.formulation(X, A_hat, tape)
        del tape

        E = 0.0
        E = E + tf.reduce_mean(tf.square(u - u_profile))   
        E = E + tf.reduce_mean(tf.square(v - 0.0))   

        return E/2

    @tf.function
    def loss_out(self, Xs):

        print("retrace out")

        X = Xs['X']
        A_hat = Xs["A_hat"]

        with tf.GradientTape() as tape:
            tape.watch(X)
            _, _, p = self.formulation(X, A_hat, tape)
        del tape

        p = p*self.cons["alpha"]

        E = 0.0
        E = E + tf.reduce_mean(tf.square(p - 0.0))

        return E/1

    @tf.function
    def loss_wall(self, Xs):

        print("retrace wall")

        X = Xs['X']
        A_hat = Xs["A_hat"]

        with tf.GradientTape() as tape:
            tape.watch(X)
            u, v, _ = self.formulation(X, A_hat, tape)
        del tape

        E = 0.0
        E = E + tf.reduce_mean(tf.square(u - 0.0))   
        E = E + tf.reduce_mean(tf.square(v - 0.0))   

        return E/2

    @tf.function
    def loss_sym(self, Xs):

        print("retrace sym")

        X = Xs['X']
        A_hat = Xs["A_hat"]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            u, v, p = self.formulation(X, A_hat, tape)
            p = p*self.cons["alpha"]

        du = tape.gradient(u, X)
        dv = tape.gradient(v, X)
        dp = tape.gradient(p, X)

        du_y = du[:,1:2]
        dp_y = dp[:,1:2]

        del tape

        E = 0.0
        E = E + tf.reduce_mean(tf.square(du_y - 0.0))   
        E = E + tf.reduce_mean(tf.square(v - 0.0))   
        E = E + tf.reduce_mean(tf.square(dp_y - 0.0))   

        return E/2

    @tf.function
    def loss_tot_hard(self, Xs, weights):
        """ Legacy. Used to assemble a composite loss function where multiple losses are weighted and summed."""

        x_fluid, ah_fluid = Xs[0], Xs[1]

        print("retrace tot hard")

        return tf.reduce_sum(self.loss_fn(x_fluid, ah_fluid))
    
    @tf.function
    def loss_tot_soft(self, Xs, weights):
        """ Legacy. Used to assemble a composite loss function where multiple losses are weighted and summed."""

        print("retrace tot soft")

        E_cont, E_mom = self.loss_fn(Xs[0], Xs[1])

        E = 0.0
        E = E + weights[0]*E_cont
        E = E + weights[0]*E_mom
        E = E + weights[1]*self.loss_in(Xs[2])
        E = E + weights[2]*self.loss_out(Xs[3])
        E = E + weights[3]*self.loss_wall(Xs[4])
        # E = E + weights[4]*self.loss_sym(Xs[5])

        return E/5

    def loss_test(self, Xs):
        """ Returns the validation error of the trial solution. """

        X = Xs["X"]
        A_hat = Xs["A_hat"]
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(X)

            u, v, p = self.formulation(X, A_hat, tape)

        du = tape.gradient(u, X)
        dv = tape.gradient(v, X)
        dp = tape.gradient(p, X)
        del tape
        
        u_fluid, v_fluid, p_fluid = self.hard_transform(X, u, v, p, du, dv, dp, Xs)

        u_fluid = u_fluid*cons["U_inf"]
        v_fluid = v_fluid*cons["U_inf"]
        p_fluid = p_fluid*cons["rho"]*cons["U_inf"]**2/cons["alpha"]

        u_fvm, v_fvm, p_fvm = Xs["U"], Xs["V"], Xs["P"]

        assert u_fluid.shape == u_fvm.shape
        assert v_fluid.shape == v_fvm.shape
        assert p_fluid.shape == p_fvm.shape 

        return [tf.reduce_mean(tf.abs(u_fluid - u_fvm))/(self.cons["U_inf"])*100,
                tf.reduce_mean(tf.abs(v_fluid - v_fvm))/(self.cons["U_inf"])*100,
                tf.reduce_mean(tf.abs(p_fluid - p_fvm))/(tf.reduce_max(tf.abs(p_fvm)))*100]


class Train():
    """ Class that handles the training of the PINN. """

    def __init__(self, pinn, net, parent_dir, runname, valve, methods:dict, netshape:dict) -> None:

        # Initialize internal variables
        self.runID          = 0
        self.pinn           = pinn
        self.net            = net
        self.parent_dir     = parent_dir
        self.runname        = runname
        self.valve          = valve
        self.netshape       = netshape

        # Initialize checkpoints directory
            # Directory name
        self.logfilename = f"{self.runname}{int(np.random.random()*1E4)}"

        print("==============")
        print(f"Current run: {self.logfilename}")
        print("==============")
        
            # Create the logging textfile and write the header
        os.mkdir(f"../output/logging/{self.logfilename}")
        with open(f"../output/logging/{self.logfilename}/{self.logfilename}_log.csv", "w") as logfile:
            for line in logheader:
                logfile.writelines(line+"\n")
            logfile.writelines("runname,runID,learningrate,numbatches,optimizer,epoch,iter,time,loss_val,l_fluid,l_inlet,l_outlet,l_wall,l_symmetry,l_test_u,l_test_v,l_test_p,w1,w2,w3,w4 \n")

    def new_run(self, Xs):
        """ Initializes a new training run of the PINN. """
            
        # Initialize network with random weights
        self.net.predict(Xs["fluid"]["X"], self.valve.mesh(Xs["fluid"]["X"], n=self.netshape['graph_connect'])[2])
        for ix, layer in enumerate(self.net.layers):
            if hasattr(self.net.layers[ix], 'kernel_initializer') and \
                    hasattr(self.net.layers[ix], 'bias_initializer'):
                weight_initializer = self.net.layers[ix].kernel_initializer
                bias_initializer = self.net.layers[ix].bias_initializer

                if len(self.net.layers[ix].get_weights()) == 1:
                    old_weights = self.net.layers[ix].get_weights()[0]

                    self.net.layers[ix].set_weights([
                        weight_initializer(shape=old_weights.shape)])
                elif len(self.net.layers[ix].get_weights()) == 2:
                    old_weights, old_biases = self.net.layers[ix].get_weights()

                    self.net.layers[ix].set_weights([
                        weight_initializer(shape=old_weights.shape),
                        bias_initializer(shape=old_biases.shape)])

        # Initialize internal variables
        self.weights   = tf.Variable([1.0]*4, dtype=self.net.prec, trainable=False)
        self.iter      = 0
        self.epoch     = 0
        self.starttime = time.time()

        # Initialize data
            # Make number of collocation points a multiple of 8
        Xs = self.subsample(Xs, [8*int(Xs['fluid']["X"].shape[0]/8)]*7,  [True, False, False, False, False, False, False])
        self.Xs        = Xs
        self.xs_fluid  = tf.data.Dataset.from_tensor_slices(self.Xs["fluid"])
        self.x_inlet = self.Xs["inlet"]
        self.x_outlet = self.Xs["outlet"]
        self.x_wall = self.Xs["wall"]
        # self.x_symmetry = self.Xs["symmetry"]

        self.Ah_f = self.valve.mesh(self.Xs["fluid"]["X"], n=self.netshape['graph_connect'])[2]

        # Initialize scaling layer
        sigmas = []
        betas  = []
        for i in range(self.Xs["fluid"]["X"].shape[1]):
            sigmas.append(tf.reduce_max(self.Xs["fluid"]["X"][:,i])-tf.reduce_min(self.Xs["fluid"]["X"][:,i]))
            betas.append(tf.reduce_min(self.Xs["fluid"]["X"][:,i]))

        scale_layer_w = tf.convert_to_tensor(np.diag(1/np.array(sigmas)), dtype=self.net.prec)
        scale_layer_b = tf.convert_to_tensor(np.array(betas), dtype=self.net.prec)
        self.net.scaling_layer_w.assign(scale_layer_w)
        self.net.scaling_layer_b.assign(scale_layer_b)
        
        self.runID += 1

    def subsample(self, Xs, batches, batchmask):
        """ Return a subset of a given dataset."""

        for X, b, mask in zip(Xs.values(), batches, batchmask):
            if mask:
                for X_tilde in X.values():
                    assert b <= X_tilde.shape[0], "Batch size must be smaller than dataset size."

        xs = {}
        for (key1, X), batch, mask in zip(Xs.items(), batches, batchmask):
            if mask:
                subdict = {}
                for (key2, X_tilde) in X.items():
                    subdict[key2] = X_tilde[0:batch,:]
                xs[key1] = subdict
            else:
                xs[key1] = X
        return xs

    #define grad and loss return function
    @tf.function
    def get_grads(self, xs):
        """ Returns the loss function value and gradients of the PINN loss function with respect to the trainable variables."""

        with tf.GradientTape() as tape_main:
            tape_main.watch(self.net.trainable_variables)
            l = self.pinn.loss_tot(xs, self.weights)
        grad = tape_main.gradient(l, self.net.trainable_variables)
        del tape_main
        return l, grad
    
    def loss_ballance(self, pinn, Xs, A_hat):
        """ Returns the mean magnitude of violation of the trial solution of the Navier-Stokes equations."""

        rho, mu, U_inf, L, dmax, alpha = pinn.cons["rho"], pinn.cons["mu"], pinn.cons["U_inf"], pinn.cons["L"], pinn.cons['dmax'], pinn.cons['alpha']
        X = Xs['X']
        D = Xs['D']

        with tf.GradientTape(persistent = True) as tape:
            tape.watch(X)

            u, v, p = pinn.formulation(X, A_hat, tape)

            du = tape.gradient(u, X)
            dv = tape.gradient(v, X)
            dp = tape.gradient(p, X)

            u, v, p = pinn.hard_transform(X, u, v, p, du, dv, dp, Xs)

            du = tape.gradient(u, X)
            dv = tape.gradient(v, X)
            dp = tape.gradient(p, X)

            du_x = tape.gradient(u, X)[:,0:1]
            dv_x = tape.gradient(v, X)[:,0:1]
            dp_x = tape.gradient(p, X)[:,0:1]
            du_y = tape.gradient(u, X)[:,1:2]
            dv_y = tape.gradient(v, X)[:,1:2]
            dp_y = tape.gradient(p, X)[:,1:2]

        ddu_xx = tape.gradient(du_x, X)[:,0:1]
        ddu_yy = tape.gradient(du_y, X)[:,1:2]
        ddv_xx = tape.gradient(dv_x, X)[:,0:1]
        ddv_yy = tape.gradient(dv_y, X)[:,1:2]

        del tape

        def mean(x):
            x = x.numpy()
            x = np.median(x)
            x = "{:.2E}".format(x)
            return x
        
        # conti = du_x + dv_y
        # mom_x = u*du_x + v*du_y + alpha*dp_x - (mu + mu_t)/(U_inf*rho*L)*(ddu_xx + ddu_yy)
        # mom_y = u*dv_x + v*dv_y + alpha*dp_y - (mu + mu_t)/(U_inf*rho*L)*(ddv_xx + ddv_yy)

        # mags = [mean(u*du_x + v*du_y), mean(dp_x), mean(ddu_xx + ddu_yy), mean(u*dv_x + v*dv_y), mean(dp_y), mean(ddv_xx + ddv_yy)]
        # mags = [mean(u), mean(v), mean(p), mean(du_x), mean(du_y), mean(dv_x), mean(dv_y), mean(dp_x), mean(dp_y), mean(ddu_xx), mean(ddu_yy), mean(ddv_xx), mean(ddv_yy)]
        # print(mags)

        beta = 0.1
        alpha_old = pinn.cons['alpha']
        alpha_bal = abs(tf.reduce_mean(ddu_xx + ddu_yy)/tf.reduce_mean(dp_x))
        alpha_new = beta*alpha_bal + (1-beta)*alpha_old
        # pinn.cons['alpha'] = alpha_new
        print(alpha_new)

    
    def dirichlet(self, alp):
            """Inverse dirichlet weighting of the loss function weights. """

            w_old = self.weights
            w_lim = tf.constant([1e2]*4, dtype=self.net.prec)

            with tf.GradientTape(persistent=True) as tape:
                TV = self.net.trainable_variables
                tape.watch(TV)

                l_fn_c, l_fn_m  = self.pinn.loss_fn(self.Xs["fluid"], self.Ah_f)
                l_fn            = l_fn_c+l_fn_m
                l_in            = self.pinn.loss_in(self.Xs["inlet"])
                l_out           = self.pinn.loss_out(self.Xs["outlet"])
                l_wall          = self.pinn.loss_wall(self.Xs["wall"])
                
            grads_fn    = tape.gradient(l_fn, TV)
            grads_in    = tape.gradient(l_in, TV)
            grads_out   = tape.gradient(l_out, TV)
            grads_wall  = tape.gradient(l_wall, TV)

            del tape

            gl_fn   = np.zeros(1,None)
            gl_in   = np.zeros(1,None)
            gl_out  = np.zeros(1,None)
            gl_wall = np.zeros(1,None)
            for g in grads_fn:
                    gl_fn = np.concatenate([gl_fn, g.numpy().flatten()])
            for g in grads_in:
                    gl_in   = np.concatenate([gl_in, g.numpy().flatten()])
            for g in grads_out:
                    gl_out  = np.concatenate([gl_out, g.numpy().flatten()])
            for g in grads_wall:
                    gl_wall = np.concatenate([gl_wall, g.numpy().flatten()])

            gl_fn_var   = np.std(gl_fn[1::]   , axis=0)
            gl_in_var   = np.std(gl_in[1::]   , axis=0)
            gl_out_var  = np.std(gl_out[1::]  , axis=0)
            gl_wall_var = np.std(gl_wall[1::] , axis=0)

            gmax = np.max([gl_fn_var, gl_in_var, gl_out_var, gl_wall_var])

            w_dir = tf.constant(np.array([gl_fn_var, gl_in_var, gl_out_var, gl_wall_var])/gmax, dtype=self.net.prec)

            # Moving average of weights
            w_new = alp*w_old + (1-alp)*w_dir

            self.weights.assign(tf.math.minimum(w_new, w_lim))

    def cb_adam(self, loss_val):
        """ Callback function for the Adam optimizer."""

        l_fluid = tf.reduce_sum(self.pinn.loss_fn(self.Xs["fluid"], self.Ah_f))
        l_test  = self.pinn.loss_test(self.Xs["validation"])
        self.loss_ballance(self.pinn, self.Xs["fluid"], self.Ah_f)
        if methods["hard"]:
            l_in = 0.0
            l_out = 0.0
            l_wall = 0.0
            # l_sym = 0.0
        else:
            l_in = self.pinn.loss_in(self.Xs["inlet"])
            l_out = self.pinn.loss_out(self.Xs["outlet"])
            l_wall = self.pinn.loss_wall(self.Xs["wall"])
            # l_sym = self.pinn.loss_sym(self.Xs["symmetry"])

        print(f"(ADAM) Epoch:{self.epoch}, Iteration:{self.iter}, Walltime:{time.time() - self.starttime}, Loss:{loss_val}, Test_u:{l_test[0]}, Test_v:{l_test[1]}, Test_p:{l_test[2]}")

        pinn.train_hist.append([int(self.epoch), int(self.iter), float(time.time() - self.starttime),
                                 float(loss_val), float(l_fluid), float(l_in), float(l_out), float(l_wall), float(0.0), float(l_test[0]), float(l_test[1]), float(l_test[2])])

        datatolog = np.array([self.runname, self.runID, self.lr, self.numbatches, "ADAM"])[:,np.newaxis].T
        datatolog = np.hstack([datatolog, np.array(self.pinn.train_hist[-1])[:,np.newaxis].T])
        datatolog = np.hstack([datatolog, np.array(self.weights)[:,np.newaxis].T])
        pd.DataFrame(datatolog).to_csv(f"../output/logging/{self.logfilename}/{self.logfilename}_log.csv",
                                    index=False, mode="a", header=False)

        save_model3(f"../output/logging/{self.logfilename}/netparams{self.iter}_{self.runID}.json", self.net, prec=prec)
    
    #define first order gradient descent train function
    def train_adam(self, epochs, numbatches, lr):
        """ Trains the PINN with the Adam optimizer. """

        self.lr         = lr
        self.numbatches = numbatches

        optimizer = tf.keras.optimizers.Adam(lr)
        xb_fluid  = self.xs_fluid.batch(int(self.Xs["fluid"]["X"].shape[0]/numbatches))

        @tf.function
        def train_step(xs):
            """ Performs a single training step."""

            loss_val, grad  = self.get_grads(xs)
            optimizer.apply_gradients(zip(grad, self.net.trainable_variables))

            return loss_val
        
        Ah_fluid = []
        for x_fluid in xb_fluid:
            A, D, Ah = self.valve.mesh(x_fluid["X"], n=self.netshape['graph_connect'])
            Ah_fluid.append(Ah)
        
        for epoch in range(epochs):
            if self.epoch == 0:
                self.cb_adam(self.pinn.loss_tot([self.Xs["fluid"], self.Ah_f, self.x_inlet, self.x_outlet, self.x_wall], self.weights))
        
            for x_fluid, ah_fluid in zip(xb_fluid, Ah_fluid):
                loss_val = train_step([x_fluid, ah_fluid, self.x_inlet, self.x_outlet, self.x_wall])
                # loss_val = train_step([x_fluid, ah_fluid, self.x_inlet, self.x_outlet, self.x_wall, self.x_symmetry])
                self.iter += 1
            self.epoch += 1

            if self.epoch <= 100:
                if self.epoch % 10 == 0:
                    self.cb_adam(loss_val)
            else:
                if self.epoch % 100 == 0:
                    self.cb_adam(loss_val)

            if (self.epoch % 100 == 0) and (methods["lossweight"]):
                self.dirichlet(alp=0.7)

    def cb_lbfgs(self, xr=None):
            """ Callback function for the L-BFGS optimizer. """
            
            if self.iter%100 == 0:
                l_fluid = tf.reduce_sum(self.pinn.loss_fn(self.Xs["fluid"], self.Ah_f))
                l_test  = self.pinn.loss_test(self.Xs["validation"])
                self.loss_ballance(self.pinn, self.Xs["fluid"], self.Ah_f)
                
                if methods["hard"]:
                    l_in = 0.0
                    l_out = 0.0
                    l_wall = 0.0
                    # l_sym = 0.0
                else:
                    l_in = self.pinn.loss_in(self.Xs["inlet"])
                    l_out = self.pinn.loss_out(self.Xs["outlet"])
                    l_wall = self.pinn.loss_wall(self.Xs["wall"])
                    # l_sym = self.pinn.loss_sym(self.Xs["symmetry"])

                print(f"(LBFGS) Epoch:{self.epoch}, Iteration:{self.iter}, Walltime:{time.time() - self.starttime}, Loss:{self.lossval}, Test_u:{l_test[0]}, Test_v:{l_test[1]}, Test_p:{l_test[2]}")

                pinn.train_hist.append([int(self.epoch), int(self.iter), float(time.time() - self.starttime),
                                        float(self.lossval), float(l_fluid), float(l_in), float(l_out), float(l_wall), float(0.0), float(l_test[0]), float(l_test[1]),
                                        float(l_test[2])])

                datatolog = np.array([self.runname, self.runID, "N/A", 1, "LBFGS"])[:,np.newaxis].T
                datatolog = np.hstack([datatolog, np.array(self.pinn.train_hist[-1])[:,np.newaxis].T])
                datatolog = np.hstack([datatolog, np.array(self.weights)[:,np.newaxis].T])
                pd.DataFrame(datatolog).to_csv(f"../output/logging/{self.logfilename}/{self.logfilename}_log.csv",
                                            index=False, mode="a", header=False)

                save_model3(f"../output/logging/{self.logfilename}/netparams{self.iter}_{self.runID}.json", self.net, prec=prec)

    def train_LBFGS(self, method="L-BFGS-B", **kwargs):
        """ Trains the PINN with the L-BFGS optimizer. """

        # xs = self.subsample(self.Xs, [1000, 1, 1, 1, 1, 1], [False, False, False, False, False, False])
        xs = [self.Xs['fluid'], self.Ah_f, self.x_inlet, self.x_outlet, self.x_wall]
        # xs = [self.Xs['fluid'], self.Ah_f, self.x_inlet, self.x_outlet, self.x_wall, self.x_symmetry]

        def get_weight_tensor():
            weight_list = []
            shape_list = []
            for v in self.net.trainable_variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())

            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        theta, shape_list = get_weight_tensor()

        def set_weight_tensor(weight_list):
            idx = 0
            for v in self.net.trainable_variables:
                vs = v.shape
                if len(vs) == 2:
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                    idx += sw
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx += vs[0]
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                v.assign(tf.cast(new_val, self.net.prec))

        def get_loss_and_grad(w):
        
            set_weight_tensor(w)
            loss_val, grad  = self.get_grads(xs)

            loss, grad = loss_val, grad
            loss = loss.numpy().astype(np.float64)
            self.iter  += 1
            self.epoch += 1
            self.lossval = loss

            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            grad_flat = np.array(grad_flat, dtype="float64")
            return loss, grad_flat

        return optimize.minimize(
            fun      = get_loss_and_grad,
            x0       = theta,
            jac      = True, 
            method   = method,
            callback = self.cb_lbfgs,
            **kwargs 
        )
    

if __name__ == "__main__":

    # Additional imports, required for running on the cluster
    import matplotlib
    # matplotlib.use('agg')

    # Simulation settings
    runname     = "all_rect"
    parent_dir  = "/home/jdutoit1/lustre/"

    methods     = {"hard":False, "sipinn":True, "gpinn": False,
                   "gcn":False, "fourier":True, "lossweight":False, "sp":True,
                   "turb":False}
    active_bcs  = {"inlet":True, "outlet":True, "wall":True, "symmetry":False}
    n_output    = check_compat(methods)
    cons        = {"rho": 100.0, "mu": 0.1, "U_inf": 1.0, "L": 2.0, "alpha":2}
    angle       = 45
    h, w        = 2, 3
    prec        = tf.float32
    validation_collcount = 300

    netshape    = {"nodes": int(8*2), "layers": 4, "fourier_sigma":1.0, "fatinlet":1,
                "graph_connect":4}

    # Validation data import
    X_fvm, Y_fvm = import_fvm_solution(cons, "../data/FVM_laminar.csv", validation_collcount, prec=prec, unitsin="mm")
    try:
        Xs.keys()
    except:
        # Import/generate collocation data
        valve           = AoValveHalf(scaling_l=cons["L"], prec=prec)
        valve.condargs = {"sigma": 10.0, "threshold": 100, "smoothing":1}
        valve.init_rect(h, w)
        Xs              = valve.rect_coll(ncol=[10,5,5,5])
        Xs = list(Xs)

        Xs[0], [F_in, F_out, F_wall, dF_in, dF_out, dF_wall, w_in, w_out, w_wall, dw_in, dw_out, dw_wall], _  = valve.rect_floaters_hard(Xs[0], netshape["fatinlet"], [], do_condition=methods["hard"])

        A_inlet, D_inlet, Ah_inlet = valve.mesh(Xs[1], n=2)
        A_outlet, D_outlet, Ah_outlet = valve.mesh(Xs[2], n=2)
        A_wall, D_wall, Ah_wall = valve.mesh(Xs[3], n=2)

        D_fluid = valve.rect_floaters_distances(Xs, kind='wall')
        print(f"Max wall distance: {tf.reduce_max(D_fluid)}")
        cons['dmax'] = tf.reduce_max(D_fluid)

        # Import/generate validation data
        X_fvm, [F_in_fvm, F_out_fvm, F_wall_fvm, dF_in_fvm, dF_out_fvm, dF_wall_fvm, w_in_fvm, w_out_fvm, w_wall_fvm, dw_in_fvm, dw_out_fvm, dw_wall_fvm], Y_fvm_l = valve.rect_floaters_hard(X_fvm, netshape["fatinlet"], [Y_fvm[:,0:1], Y_fvm[:,1:2], Y_fvm[:,2:3]], do_condition=methods["hard"])

        A_fvm, D_fvm, Ah_fvm = valve.mesh(X_fvm, n=netshape['graph_connect'])

        # Assemble Xs dictionary
        Xs = {"fluid":{"X":Xs[0], "D":D_fluid,
                       "F_in":F_in, "F_out":F_out, "F_wall":F_wall,
                       "dF_in":dF_in,"dF_out":dF_out, "dF_wall":dF_wall,
                       "w_in":w_in, "w_out":w_out,"w_wall":w_wall, 
                       "dw_in":dw_in, "dw_out":dw_out, "dw_wall":dw_wall},
              "inlet":{"X":Xs[1][:,0:2], "A_hat":Ah_inlet},
              "outlet":{"X":Xs[2][:,0:2], "A_hat":Ah_outlet},
              "wall":{"X":Xs[3][:,0:2], "A_hat":Ah_wall},
              "validation":{"X":X_fvm, "F_in":F_in_fvm, "F_out":F_out_fvm, 
                            "F_wall":F_wall_fvm,"dF_in":dF_in_fvm,
                            "w_in":w_in_fvm, "w_out":w_out_fvm,"w_wall":w_wall_fvm,
                            "dw_in":dw_in_fvm,"dw_out":dw_out_fvm, 
                            "dw_wall":dw_wall_fvm,"U":Y_fvm_l[0], 
                            "V":Y_fvm_l[1], "P":Y_fvm_l[2],"A_hat":Ah_fvm}}

    cons['dmax'] = tf.reduce_max(D_fluid)

    # Plot collocation data
    X = Xs["fluid"]["X"]
    fig = plt.figure(figsize = (15,3))
    ax1 = fig.add_subplot(111)
    im1 = ax1.scatter(Xs["fluid"]["X"][:,0], Xs["fluid"]["X"][:,1], marker = ".", s=60); ax1.set_title("Collocation")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")
    plt.savefig("../output/test_coll.png")

    # Plot wall distance derivative
    X = Xs["fluid"]["X"]
    fig = plt.figure(figsize = (15,3))
    ax1 = fig.add_subplot(111)
    im1 = ax1.scatter(Xs["fluid"]["X"][:,0], Xs["fluid"]["X"][:,1], c=Xs["fluid"]["dw_wall"][:,0], marker = ".", s=60); ax1.set_title("walldist derivative")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")
    plt.savefig("../output/test_walldist.png")

    # Plot validation data
    X = Xs["validation"]["X"]
    fig = plt.figure(figsize = (15,3))
    ax1 = fig.add_subplot(111)
    im1 = ax1.scatter(Xs["validation"]["X"][:,0], Xs["validation"]["X"][:,1], c=Xs["validation"]["U"] , marker = ".", s=60); ax1.set_title("U_fvm")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")
    plt.savefig("../output/test_valid.png")

    # Plot connectivity
    # fig = plt.figure(figsize = (15,3))
    # for i in range(Xs["validation"]["X"].shape[0]):
    #     for j in range(Xs["validation"]["X"].shape[0]):
    #         if i != j and tf.sparse.to_dense(Xs["validation"]["A_hat"])[i,j] != 0:
    #             plt.plot([Xs["validation"]["X"][i,0], Xs["validation"]["X"][j,0]],[Xs["validation"]["X"][i,1], Xs["validation"]["X"][j,1]], c=plt.cm.viridis(tf.sparse.to_dense(Xs["validation"]["A_hat"])[i,j]))

    #=============================================================
    # Test Train
    #=============================================================

    network = Network(hiddenlayers=netshape["layers"], nodes=netshape["nodes"], input_n=2, output_n=n_output, fourier_sigma=netshape["fourier_sigma"], prec=prec, methods=methods)
    pinn = Pinn(network, cons, methods=methods, active_bcs=active_bcs)

    # Logfile header:
    logheader = ["==",
                 f"@ Technologies: #{methods['hard']}$hard ; #{methods['sipinn']}$sipinn ; #{methods['gpinn']}$gpinn ; #{methods['gcn']}$gcn ; #{methods['fourier']}$fourier ; #{methods['lossweight']}$lossweight ; #{methods['sp']}$sp ; #{methods['turb']}$turb",
                 f"NS formulation: {'sp' if methods['sp'] else 'vp'} ; #{'mixing length' if methods['turb'] else 'laminar'} ; incompressible; non-dimensionalised",
                 f"@ Valve angle: #{angle}$angle degrees",
                 f"@ Fluid properties: viscosity #{cons['mu']}$mu ; density #{cons['rho']}$rho ; p-scale #{cons['alpha']}$alpha",
                 f"@ Characteristic properties: U_inf #{cons['U_inf']}$U_inf ; L #{cons['L']}$L",
                 f"Collocation count (fluid): {Xs['fluid']['X'].shape[0]}",
                 "Details about validation data: ",
                 f"@ Hidden layers: #{network.hiddenlayers}$hiddenlayers",
                 f"@ Nodes per layer: #{network.nodes}$nodes_n",
                 f"@ Inlet fatness: #{netshape['fatinlet']}$fatinlet",
                 f"@ GCN connectivity: #{netshape['graph_connect']}$graph_connect ",
                 f"@ Fourier sigma: #{network.fourier_sigma}$fourier_sigma",
                 f"@ output shape: #{n_output}$n_output",
                 f"@ Active constraints: #{active_bcs['inlet']}$inlet ; #{active_bcs['outlet']}$outlet ; #{active_bcs['wall']}$wall ; #{active_bcs['symmetry']}$symmetry",
                 f"Floating point precision: {prec}",
                 f"Initialization: GolorotNormal",
                 f"Activation: tanh",
                 "=="]
    
    train = Train(pinn, network, parent_dir, runname, valve, methods, netshape)

    for i in range(1):
        train.new_run(Xs)
        train.weights = tf.Variable([1.0]*4, dtype=prec, trainable=False)
        train.train_adam(2_000, 2, 1E-3)
        train.train_LBFGS(options={'maxiter':2_000,
                                    'maxfun': 10000,
                                    'maxcor':50,
                                    'maxls':50,
                                    'ftol': np.finfo(float).eps})

    #=============================================================
    # Illustrate network inference after training
    #=============================================================
    X = Xs["fluid"]["X"]
    A_hat = valve.mesh(X, n=netshape['graph_connect'])[2]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        U, V, P = pinn.formulation(X, A_hat, tape)
    dU = tape.gradient(U, X)
    dV = tape.gradient(V, X)
    dP = tape.gradient(P, X)
    del tape 

    U, V, P = pinn.hard_transform(X, U, V, P, dU, dV, dP, Xs["fluid"])

    X     = X*cons["L"]
    U_dim = U*cons["U_inf"]
    V_dim = V*cons["U_inf"]
    P_dim = P*cons["rho"]*cons["U_inf"]**2/cons["alpha"]

    fig = plt.figure(figsize = (6,8))
    ax1 = fig.add_subplot(311)
                
    im1     = ax1.scatter(X[:,0], X[:,1], c=U_dim[:,0], marker = ".", s=60); ax1.set_title("u")
    divider = make_axes_locatable(ax1)
    cax     = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")

    ax2     = fig.add_subplot(312)
    im2     = ax2.scatter(X[:,0], X[:,1], c=V_dim[:,0], marker = ".", s=60); ax2.set_title("v")
    divider = make_axes_locatable(ax2)
    cax     = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation="vertical")

    ax3     = fig.add_subplot(313)
    im3     = ax3.scatter(X[:,0], X[:,1], c=P_dim[:,0], marker = ".", s=60); ax3.set_title("p")
    divider = make_axes_locatable(ax3)
    cax     = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation="vertical")

    plt.savefig("../output/test_result.png")
# %%
