import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

# Usage in PINN class
        # # Creating PINN layers
        # pinn_layers = {}
        # pinn_layers[0] = VDense(n_nodes, n_population,# input layer
        #                             activation='linear', use_bias = False, is_input = True)
        # for i in range(n_layers+1):
        #     pinn_layers[1+i] = VDense(n_nodes, n_population, activation="tanh", use_bias = True) # Hidden layers
        #     if i == n_layers-1:
        #         pinn_layers[1+i] = VDense(n_output, n_population, activation="linear", use_bias = False) # Output layer
        #         break

class VDense(keras.layers.Layer):
    def __init__(self, n_nodes, n_vect, activation = "tanh", use_bias = True, is_input=False):
        super(VDense, self).__init__()
        """ Return a keras layer object which has vectorized dense layers.
    
        Parameters
        ----------
        n_nodes : int
            Number of nodes in the layer output
        n_vect : int
            Number of parallel networks
        activation : str
            Activation function to use.
        use_bias : bool
            Whether to use bias or not.
        is_input : bool
            Whether the layer is the input layer or not.

        Notes
        ----- 
        This is a custom layer to be used when training with 
        particle swarm/genetic algorithms/differential evolution etc.

        Multiple design points can be evaluated in a vectorized fashion using this network architechture.

        This is in essence an architechture where mutliple neural networks exist in parallel, 
        but do not communicate at all.
        """ 
        self.n_nodes = n_nodes
        self.n_vect = n_vect
        self.activation = activation
        self.use_bias = use_bias
        self.is_input = is_input

    def build(self, input_shape):
        """ Initialize the layers. """
        
        # Kernel of shape (n_in, n_vect, n_out)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.n_vect, self.n_nodes),
            initializer="random_normal",
            trainable=True,
        )

        # Bias of shape (n_vect, n_out)
        if self.use_bias == True:
            self.b = self.add_weight(
                shape=(self.n_vect, self.n_nodes), initializer="random_normal", trainable=True
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

    def call(self, inputs):
        """ Perform forward pass through the layer . 
    
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        Y : tf.Tensor : shape (n_samples, n_vect, n_out)
            Output tensor.
        """

        if self.is_input == True:
            Y = self.activation(tf.tensordot(inputs, self.w, axes=1) + self.b)
            return Y
        elif self.is_input == False:
            Y = self.activation(tf.tensordot(inputs, self.w, axes=1) + self.b)
            Y = tf.transpose(Y, perm=[0,3,1,2])
            Y = tf.linalg.diag_part(Y)
            Y = tf.transpose(Y, perm=[0,2,1])
            return Y
        else:
            pass