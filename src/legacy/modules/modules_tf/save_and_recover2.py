import tensorflow as tf

from numpy import array
from numpy.random import random

def save_model2(filename, model, n_input):
    """ Save the trainable parameters of a model to a json file. """

    model.call(tf.Variable(random((100, n_input)), dtype="float32"))

    def dump_json(filename):
        import json

        wb_arrays = list()
        for l in model.get_weights():
            wb_arrays.append(list(l.tolist()))

        with open(filename, "w") as outfile:
            json.dump(wb_arrays, outfile)

    dump_json(filename)

def recover_model2(filename, model):
    """ Assign trainable parameters to a model from an existing json file. """

    def load_json(filename):
        import json

        with open(filename) as json_file:
            wb_arrays = json.load(json_file)
        trainable_variables = list()
        for i in range(len(wb_arrays)):
            trainable_variables.append(tf.Variable(array(wb_arrays[i]), dtype="float32"))
        return trainable_variables

    new_wb = load_json(filename)
    
    model.call(random((100, new_wb[0].shape[0])))
    model.set_weights(new_wb)
    # for i in range(len(model.trainable_variables)):   
    #     model.trainable_variables[i].assign(new_wb[i])
        # model.predict_vel(random((100, new_wb[0].shape[0])))
