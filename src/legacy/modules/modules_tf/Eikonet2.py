import tensorflow as tf
from tensorflow.keras import layers, Model
from numpy import pi


class Eikonet2(Model):
    def __init__(self, nodes, input, output, prec, fourier_sigma=0.2) -> None:
        super().__init__()
        
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.fourier_initializer = tf.random_normal_initializer(mean=0.0, stddev=fourier_sigma, seed=None)

        # Creating PINN layers
            # Fourier layer
        self.pinn_fourier = layers.Dense(units=nodes, input_shape=(None, input),
                                        activation="linear", use_bias=False, dtype=prec,
                                        kernel_initializer=self.fourier_initializer, trainable=False)

            # SIPINN layers
        self.pinn_M = layers.Dense(units=nodes, input_shape=(None, input*2),
                                    activation='tanh', use_bias=True, name="M", dtype=prec, kernel_initializer=self.initializer)
        self.pinn_N = layers.Dense(units=nodes, input_shape=(None, input*2),
                                    activation='tanh', use_bias=True, name="N", dtype=prec, kernel_initializer=self.initializer)

            # Input layer
        self.pinn_in = layers.Dense(units=nodes, input_shape=(None, input*2),
                                    activation='tanh', use_bias=True, name="input", dtype=prec, kernel_initializer=self.initializer)

            # Hidden layers
        pinn_layers = []
        pinn_layers.append(layers.Dense(nodes, activation="tanh", dtype=prec, kernel_initializer=self.initializer))
        pinn_layers.append(layers.Dense(nodes, activation="tanh", dtype=prec, kernel_initializer=self.initializer))
        pinn_layers.append(layers.Dense(nodes, activation="tanh", dtype=prec, kernel_initializer=self.initializer))
        pinn_layers.append(layers.Dense(nodes, activation="tanh", dtype=prec, kernel_initializer=self.initializer))
        self.pinn_hidden = pinn_layers

            # Output layer
        self.pinn_out = layers.Dense(output, activation="linear", use_bias=False, name="output", dtype=prec, kernel_initializer=self.initializer)

    @tf.function
    def call(self, X):

        # fourier layer
        psi_s = tf.math.sin(2.0*pi*self.pinn_fourier(X))
        psi_c = tf.math.cos(2.0*pi*self.pinn_fourier(X))
        X = tf.concat([psi_s, psi_c], axis=1)

        # SIpinn
        XM, XN = self.pinn_M(X), self.pinn_N(X)

        # Input
        h = tf.math.multiply(self.pinn_in(X), XM) + XN

        # Hidden
        h = tf.math.multiply(self.pinn_hidden[0](h), XM) + XN
        h = tf.math.multiply(self.pinn_hidden[1](h), XM) + XN
        h = tf.math.multiply(self.pinn_hidden[2](h), XM) + XN
        h = tf.math.multiply(self.pinn_hidden[3](h), XM) + XN

        # Output
        h = self.pinn_out(h)

        return h

    @tf.function
    def predict(self, X):

        out = self.call(X)
        l = out[:,0:1]
        return l


    @tf.function
    def predict_dist(self, X):

        with tf.GradientTape() as tape:
            tape.watch(X)

            l = self.predict(X)

            dl = tape.gradient(l, X)

        dl_x = dl[:,0:1]
        dl_y = dl[:,1:2]

        d = -tf.sqrt(tf.square(dl_x)+tf.square(dl_y)) + tf.sqrt(tf.square(dl_x)+tf.square(dl_y) + 2*l)

        d = tf.math.maximum(d, 0.0)
        return d

def notouch2(eikonet, Xs, eps):

    X = Xs["fluid"]["X"]
    D = Xs["fluid"]["D"]

    d = eikonet.predict_dist(X)

    mask = (tf.sqrt(tf.square(d))>eps)[:,0]

    Xs_out = Xs.copy()
    Xs_out["fluid"]["X"] = tf.boolean_mask(Xs["fluid"]["X"],mask)
    Xs_out["fluid"]["D"] = tf.boolean_mask(Xs["fluid"]["D"],mask)

    return Xs_out