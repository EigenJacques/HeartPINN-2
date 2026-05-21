import wandb

from numpy import array, product, vstack, newaxis

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

import matplotlib
from matplotlib import pyplot as plt

from pandas import DataFrame

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/media/jflinux_main/HDD/Studies/Studies_MEngM/Navorsing/HeartPINN/Phase_2/conference_paper/modules/')
from tfp_lbfgs import minimize as lbfgs_minimize
# import bcs
# from PINN_euler import PINN

from time import time

from pyDOE import lhs

class TRAIN():
    def __init__(self, collocation, pinn, data, precision, do_wandb_logging):

        X_internal, X_inlet, X_outlet, X_wall, N_wall = collocation
        train_hist = tf.expand_dims(tf.constant([0.0,0.0,0.0,0.0,0.0], dtype= precision), axis=0)

        self.X_internal = X_internal
        self.X_inlet = X_inlet
        self.X_outlet = X_outlet
        self.X_wall = X_wall
        self.N_wall = N_wall
        self.Data = data
        self.pinn = pinn
        self.train_hist = train_hist
        self.epoch = 0

        self.learning_rate = 1E-3
        self.precision = precision
        self.do_wandb_logging = do_wandb_logging
        self.init_time = time()

        # Logging of training settings

    def log_wandb(self):
            # execution time 
            wandb.log({"time after initialization": time()-self.init_time, "global_step": self.epoch})

            # hyper parameters
            wandb.log(dict(zip(["loss_weighting_fn", "loss_weighting_in", "loss_weighting_out", "loss_weighting_wall", "loss_weighting_data", "global_step"], tf.concat([self.pinn.loss_weighting, tf.expand_dims(tf.cast(self.epoch, dtype=self.precision), axis=0)], axis=0))))

            # loss
            wandb.log({"loss_fn": self.pinn.loss_fn(X_fn=self.X_internal), "global_step": self.epoch})
            wandb.log({"loss_wall": self.pinn.loss_wall(X_wall=self.X_wall, N_wall=self.N_wall), "global_step": self.epoch})
            wandb.log({"loss_in": self.pinn.loss_in(X_in=self.X_inlet), "global_step": self.epoch})
            wandb.log({"loss_out": self.pinn.loss_out(X_out=self.X_outlet), "global_step": self.epoch})
            wandb.log({"loss_data": self.pinn.loss_data(Data=self.Data), "global_step": self.epoch})
            wandb.log({"loss_total_weighted": self.pinn.loss(X_fn=self.X_internal, X_in=self.X_inlet, X_out=self.X_outlet, X_wall=self.X_wall, Data=self.Data, N_wall=self.N_wall), "global_step": self.epoch})
            
            # trainable variables
            # shapes = tf.shape_n(self.pinn.trainable_variables)
            # count = 0
            # idx = [] # stitch indices
            # for shape in shapes:
            #     n = product(shape)
            #     idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            # dp_current = tf.dynamic_stitch(idx, self.pinn.trainable_variables)

            # wandb.log({'TV': wandb.Table(data=DataFrame(dp_current.numpy()[:, newaxis].T,columns=[str(x).zfill(2) for x in range(len(dp_current))]))})
            # wandb.log(dict(zip([str(x).zfill(2) for x in range(len(dp_current))], dp_current)))

    def iterate_epoch_de(self, n_epoch, n_population=4, scale_init=1):

        assert n_population >= 4, "n_population must be greater than or equal to 4"

        #Initialize weights
        self.pinn.call(self.Data[0])

        start_time = time()

        # Get shape of trainable parameters
        shapes = tf.shape_n(self.pinn.trainable_variables)
        n_tensors = len(shapes)

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
            """ Update the model parameters with a 1D tf.Tensor. 
        
            Arguments
            ----------
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.

            """ 

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.pinn.trainable_variables[i].assign(tf.reshape(param, shape))

        # @tf.function
        def objective_fn(params_2d):
            """ Evaluate the model loss and loss gradients. 
        
            Arguments
            ----------
                params_2d [in]: a 2D tf.Tensor of shape [n_population, n_trainable_variables].

            Returns:
                A 1D tf.Tensor of loss values.
            """

            losses = tf.constant([], dtype=self.precision)
            for individual in range(params_2d.shape[0]):
                assign_new_model_parameters(params_2d[individual,:])
                l = tf.expand_dims(self.pinn.loss(X_fn=self.X_internal, X_in=self.X_inlet, X_out=self.X_outlet, X_wall=self.X_wall, Data=self.Data, N_wall=self.N_wall), axis=0)
                losses = tf.concat([losses, l], axis=0)
            
            return losses

        def init_params(n_population):
            scale = scale_init
            dp_current = tf.dynamic_stitch(idx, self.pinn.trainable_variables)

            # dp_population = scale*(lhs(dp_current.shape[0], samples=n_population-1)-0.5)
            dp_population = scale*(lhs(dp_current.shape[0], samples=n_population-1)-0.5 + dp_current/scale)
            dp_population = vstack([dp_population, dp_current])

            return tf.convert_to_tensor(dp_population, dtype=self.precision)

        # @tf.function
        def min_fn(pop):
            """ Minimize the objective function. """

            results = tfp.optimizer.differential_evolution_one_step(
                    objective_function=objective_fn,
                    population=pop,
                    differential_weight=0.5,
                    crossover_prob=tf.constant(0.9))

            return results[0], results[1]

        pop = init_params(n_population)
        for i in range(n_epoch):
            self.epoch += 1
            pop, errors = min_fn(pop)
            
            if i > 10:
                if i%int(n_epoch/10) == 0: # Monitor progress
                    best_ind = tf.math.top_k(1/errors, k=2)
                    assign_new_model_parameters(pop[0][best_ind.indices[0],:])

                    loss_fn = self.pinn.loss_fn(X_fn=self.X_internal)
                    loss_wall = self.pinn.loss_wall(X_wall=self.X_wall, N_wall=self.N_wall)
                    loss_in = self.pinn.loss_in(X_in=self.X_inlet)
                    loss_out = self.pinn.loss_out(X_out=self.X_outlet)
                    loss_data = self.pinn.loss_data(Data=self.Data)
                    self.train_hist = tf.concat([self.train_hist, [[loss_fn, loss_wall, loss_in, loss_out, loss_data]]], axis=0)
                    
                    print("Iteration (de): {}, Time: {:.5f}, Loss_fn: {:.5f}, Loss_wall: {:.5f}, Loss_in: {:.5f}, Loss_out: {:.5f}, Loss_data: {:.5f}".format(i, time()-start_time, loss_fn, loss_wall, loss_in, loss_out, loss_data))
                    
                    # wandb logging
                    if self.do_wandb_logging == True:
                        self.log_wandb()
                        wandb.log({"opt": 0, "global_step": self.epoch})

        best_ind = tf.math.top_k(1/errors, k=2)
        assign_new_model_parameters(pop[0][best_ind.indices[0],:])
        
    def iterate_epoch_lbfgs(self, n_epoch, parallel=1, tolerance=1E-8):

        #Initialize weights
        self.pinn.call(self.Data[0])

        iter_n = tf.Variable(0)
        start_time = time()

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.pinn.trainable_variables)
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
                self.pinn.trainable_variables[i].assign(tf.reshape(param, shape))

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
                loss = self.pinn.loss(X_fn=self.X_internal, X_in=self.X_inlet, X_out=self.X_outlet, X_wall=self.X_wall, Data=self.Data, N_wall=self.N_wall)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss, self.pinn.trainable_variables)
            grads = tf.dynamic_stitch(idx, grads)

            return loss, grads

        def logging(epoch, dp):

            dp = dp
            i = epoch

            self.epoch += 1
                
            loss_fn = self.pinn.loss_fn(X_fn=self.X_internal)
            loss_wall = self.pinn.loss_wall(X_wall=self.X_wall, N_wall=self.N_wall)
            loss_in = self.pinn.loss_in(X_in=self.X_inlet)
            loss_out = self.pinn.loss_out(X_out=self.X_outlet)
            loss_data = self.pinn.loss_data(Data=self.Data)
            self.train_hist = tf.concat([self.train_hist, [[loss_fn, loss_wall, loss_in, loss_out, loss_data]] ], axis=0)
            # tf.print("Iteration (lbfgs): ", i, "Time: ", time()-start_time , "Loss_fn: ", loss_fn, "Loss_wall: ", loss_wall, "Loss_in: ", loss_in, "Loss_out: ", loss_out, "Loss_data: ", loss_data)
            tf.print("Iteration (lbfgs): {}, Time: {:.5f}, Loss_fn: {:.5f}, Loss_wall: {:.5f}, Loss_in: {:.5f}, Loss_out: {:.5f}, Loss_data: {:.5f}".format(i, time()-start_time, loss_fn, loss_wall, loss_in, loss_out, loss_data))

            # wandb logging
            if self.do_wandb_logging == True:
                self.log_wandb()
                wandb.log({"opt": 1, "global_step": self.epoch})

        # @tf.function
        def min_fn():
            init_params = tf.dynamic_stitch(idx, self.pinn.trainable_variables)
            results = lbfgs_minimize(
                    value_and_gradients_function=objective_fn,
                    initial_position=init_params,
                    max_iterations=tf.constant(n_epoch, dtype="int32"),
                    tolerance= tolerance,
                    parallel_iterations=parallel,
                    logger = logging)
            assign_new_model_parameters(results.position)
            tf.print("converged?", results.converged)
        min_fn()

    def iterate_epoch_adam(self, n_epoch):

        optim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        @tf.function
        def improve(self):
            with tf.GradientTape() as tape:
                loss = self.pinn.loss(X_fn=self.X_internal, X_in=self.X_inlet, X_out=self.X_outlet, X_wall=self.X_wall, Data=self.Data, N_wall=self.N_wall)
            grads = tape.gradient(loss, self.pinn.trainable_variables)
            optim.apply_gradients(zip(grads, self.pinn.trainable_variables))

        start_time = time()
        for i in range(n_epoch):
            self.epoch += 1
            improve(self)

            if i%int(n_epoch/10) == 0: # Monitor progress
                loss_fn = self.pinn.loss_fn(X_fn=self.X_internal)
                loss_wall = self.pinn.loss_wall(X_wall=self.X_wall, N_wall=self.N_wall)
                loss_in = self.pinn.loss_in(X_in=self.X_inlet)
                loss_out = self.pinn.loss_out(X_out=self.X_outlet)
                loss_data = self.pinn.loss_data(Data=self.Data)
                self.train_hist = tf.concat([self.train_hist, [[loss_fn, loss_wall, loss_in, loss_out, loss_data]]], axis=0)

                print("Iteration (adam): {}, Time: {:.5f}, Loss_fn: {:.5f}, Loss_wall: {:.5f}, Loss_in: {:.5f}, Loss_out: {:.5f}, Loss_data: {:.5f}".format(i, time()-start_time, loss_fn, loss_wall, loss_in, loss_out, loss_data))
                
                # wandb logging
                if self.do_wandb_logging == True:
                    self.log_wandb()
                    wandb.log({"opt": 2, "global_step": self.epoch})

    def update_train(self, speed):
        self.learning_rate = speed
        self.optim = tf.keras.optimizers.Adam(learning_rate=speed)

    def illustrate_geometry(self):
        plt.figure()
        plt.scatter(self.X_internal[:, 0], self.X_internal[:, 1], marker = ".")
        plt.scatter(self.X_wall[:, 0], self.X_wall[:, 1], marker = ".")
        plt.scatter(self.X_inlet[:, 0], self.X_inlet[:, 1], marker = ".")
        plt.scatter(self.X_outlet[:, 0], self.X_outlet[:, 1], marker = ".")
        plt.quiver(self.X_wall[:, 0], self.X_wall[:, 1], self.N_wall[:,0], self.N_wall[:,1])
        plt.show()

        if self.do_wandb_logging == True:
            wandb.log({"plot: geometry": plt})

    def illustrate_solution(self):
        # pinn_predict = self.pinn.predict_vel(self.X_internal)[0]
        # plt.figure()
        # plt.scatter(self.X_internal[:, 0], self.X_internal[:, 1], c = pinn_predict/max(abs(pinn_predict)), marker = ".", vmin=0)
        # plt.colorbar()
        # plt.title("U velocity")
        # print(max(pinn_predict))

        # pinn_predict = self.pinn.predict_vel(self.X_internal)[1]
        # plt.figure()
        # plt.scatter(self.X_internal[:, 0], self.X_internal[:, 1], c = pinn_predict/max(abs(pinn_predict)), marker = ".")
        # plt.colorbar()
        # plt.title("V velocity")
        # print(max(pinn_predict))

        pinn_predict = array(self.pinn.call(self.X_internal))[:,0]
        plt.figure()
        plt.tricontour(self.X_internal[:, 0], self.X_internal[:, 1], pinn_predict, levels = 20)
        plt.title("Potential lines")
        plt.colorbar()

        if self.do_wandb_logging == True:
            wandb.log({"plot: potential": wandb.Image(plt)})

        # Internal points quiver
        norm = matplotlib.colors.Normalize(vmin=0)
        cm = matplotlib.cm.viridis; sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        plt.figure(figsize = (20,10))
        pinn_predict = self.pinn.predict_vel(self.X_internal)
        max_vect = max(abs(pinn_predict[0]))
        vmag = (pinn_predict[0]**2 + pinn_predict[1]**2)**0.5
        plt.quiver(self.X_internal[:, 0], self.X_internal[:, 1], pinn_predict[0]/max_vect, pinn_predict[1]/max_vect, color = cm(norm(vmag)))
        plt.colorbar(sm)

        # Inlet quiver
        pinn_predict = self.pinn.predict_vel(self.X_inlet)
        plt.quiver(self.X_inlet[:, 0], self.X_inlet[:, 1], pinn_predict[0]/max_vect, pinn_predict[1]/max_vect)

        # Outlet quiver
        pinn_predict = self.pinn.predict_vel(self.X_outlet)
        plt.quiver(self.X_outlet[:, 0], self.X_outlet[:, 1], pinn_predict[0]/max_vect, pinn_predict[1]/max_vect)

        # Wall quiver
        pinn_predict = self.pinn.predict_vel(self.X_wall)
        plt.quiver(self.X_wall[:, 0], self.X_wall[:, 1], pinn_predict[0]/max_vect, pinn_predict[1]/max_vect)

        # # Data quiver
        # pinn_predict = pinn.predict_vel(X_data)
        # plt.quiver(X_data[:, 0], X_data[:, 1], pinn_predict[0]/max_vect, pinn_predict[1]/max_vect, color = "red")

        if self.do_wandb_logging == True:
            wandb.log({"plot: velocity_vectors": wandb.Image(plt)})

        train_hist = array(self.train_hist)
        plt.figure()
        plt.plot(range(train_hist[:,0].shape[0]), train_hist[:,0], label= "function")
        plt.plot(range(train_hist[:,1].shape[0]), train_hist[:,1], label = "wall")
        plt.plot(range(train_hist[:,2].shape[0]), train_hist[:,2], label = "inlet")
        plt.plot(range(train_hist[:,3].shape[0]), train_hist[:,3], label = "outlet")
        plt.plot(range(train_hist[:,4].shape[0]), train_hist[:,4], label = "data")
        plt.yscale("log")
        plt.legend()

        plt.show()