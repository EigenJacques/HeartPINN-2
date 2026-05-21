import tensorflow as tf

from numpy import array
from numpy.random import random

def save_model3(filename, model, prec):
    """ Save the trainable parameters of a model to a json file. """

    call_shape = model.call_shape

    testinputs = []
    for cshape in call_shape:
        if cshape[2] == 0:
            testinputs.append(tf.convert_to_tensor(random((cshape[0], cshape[1])), dtype=prec))
        elif cshape[2] == 1:
            testinputs.append(tf.sparse.from_dense(tf.convert_to_tensor(random((cshape[0], cshape[1])), dtype=prec)))
        else:
            print("err")
    model.call(*testinputs)

    def dump_json(filename):
        import json

        wb_arrays = list()
        for l in model.get_weights():
            wb_arrays.append(list(l.tolist()))

        wb_arrays.append(list(model.scaling_layer_w.numpy().tolist()))
        wb_arrays.append(list(model.scaling_layer_b.numpy().tolist()))

        with open(filename, "w") as outfile:
            json.dump(wb_arrays, outfile)

    dump_json(filename)

def recover_model3(filename, model, prec):
    """ Assign trainable parameters to a model from an existing json file. """

    call_shape = model.call_shape

    testinputs = []
    for cshape in call_shape:
        if cshape[2] == 0:
            testinputs.append(tf.convert_to_tensor(random((cshape[0], cshape[1])), dtype=prec))
        elif cshape[2] == 1:
            testinputs.append(tf.sparse.from_dense(tf.convert_to_tensor(random((cshape[0], cshape[1])), dtype=prec)))
        else:
            print("err")

    def load_json(filename):
        import json

        with open(filename) as json_file:
            wb_arrays = json.load(json_file)
        trainable_variables = list()
        for i in range(len(wb_arrays)):
            trainable_variables.append(tf.Variable(array(wb_arrays[i]), dtype=prec))

        # Make scaling layer non trainable
        trainable_variables[-2] = tf.Variable(trainable_variables[-2],  dtype=prec, trainable=False)
        trainable_variables[-1] = tf.Variable(trainable_variables[-1], dtype=prec, trainable=False)
        return trainable_variables[0:-2], trainable_variables[-2], trainable_variables[-1]

    new_wb, scaling_layer_w, scaling_layer_b = load_json(filename)
    
    model.scaling_layer_w = scaling_layer_w
    model.scaling_layer_b = scaling_layer_b
    model.call(*testinputs)
    model.set_weights(new_wb)
    
    # for i in range(len(model.trainable_variables)):   
    #     model.trainable_variables[i].assign(new_wb[i])
        # model.predict_vel(random((100, new_wb[0].shape[0])))
