import tensorflow as tf

from numpy import array
from numpy.random import random

def save_model(filename, model, n_input):
    """ Save the trainable parameters of a model to a json file. """

    model.call(tf.Variable(random((100, n_input)), dtype="float32"))

    def dump_json(filename):
        import json

        names = list()
        arrays = list()
        for i in range(len(model.trainable_variables)):
            names.append(model.trainable_variables[i].name)
            arrays.append(list(model.trainable_variables[i].numpy().tolist()))

        with open(filename, "w") as outfile:
            json.dump(dict(zip(names, arrays)), outfile)

    dump_json(filename)

def recover_model(filename, model):
    """ Assign trainable parameters to a model from an existing json file. """

    def load_json(filename):
        import json

        with open(filename) as json_file:
            data = json.load(json_file)
        names = list(data.keys())
        arrays = list(data.values())
        trainable_variables = list()
        for i in range(len(names)):
            trainable_variables.append(tf.Variable(array(arrays[i]), dtype="float32", name = names[i]))
        return trainable_variables

    new_wb = load_json(filename)
    
    model.predict_vel(random((100, new_wb[0].shape[0])))
    for i in range(len(model.trainable_variables)):   
        model.trainable_variables[i].assign(new_wb[i])
        # model.predict_vel(random((100, new_wb[0].shape[0])))