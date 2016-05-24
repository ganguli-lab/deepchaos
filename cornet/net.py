

import theano.tensor as T
import theano

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.noise import GaussianDropout
from keras.initializations import normal, orthogonal
import keras.backend as K
import numpy as np


def hard_tanh(xx):
    return K.minimum(K.relu(xx + 1) - 1, 1)


def randomize_weight(w, mu=None, sigma=None):
    if mu is None:
        mu = w.mean()
    if sigma is None:
        sigma = w.std()
    return sigma * np.random.randn(*w.shape) + mu

def randomize_weights(weights, bias_sigma=0.1, weight_sigma=1.0):
    random_weights = []
    for w in weights:
        if w.ndim == 1:
            rw = randomize_weight(w, mu=0, sigma=bias_sigma)
            rw = -rw
        else:
            sigma = weight_sigma * 1.0 / np.sqrt(w.shape[0])
            rw = randomize_weight(w, mu=0, sigma=sigma)
        random_weights.append(rw)
    return random_weights

class RandNet(object):
    """Simple wrapper around Keras model that throws in some useful functions like randomization"""
    def __init__(self, input_dim, n_hidden_units, n_hidden_layers, nonlinearity='tanh', bias_sigma=0.0, weight_sigma=1.25, input_layer=None, flip=False, output_dim=None):
        #if input_layer is not None:
        #    assert input_layer.output_shape[1] == input_dim
        self.input_dim = input_dim
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.nonlinearity = nonlinearity
        self.bias_sigma = bias_sigma
        self.weight_sigma = weight_sigma
        self.input_layer = input_layer

        if output_dim is None:
            output_dim = n_hidden_units
        self.output_dim = output_dim

        model = Sequential()
        if input_layer is not None:
            model.add(input_layer)
        for i in xrange(n_hidden_layers):
            nunits = n_hidden_units if i < n_hidden_layers - 1 else output_dim
            if flip:
                model.add(Activation(nonlinearity, input_shape=(input_dim,), name='_a%d'%i))
                model.add(Dense(nunits, name='_d%d'%i))
            else:
                model.add(Dense(nunits, input_shape=(input_dim,), name='_d%d'%i))
                if i <  n_hidden_layers - 1 or self.output_dim == self.n_hidden_units:
                    model.add(Activation(nonlinearity, name='_a%d'%i))
                else:
                    # Theano is optimizing out the nonlinearity if it can which is breaking shit
                    # Give it something that it won't optimize out.
                    model.add(Activation(lambda x: T.minimum(x, 999999.999),  name='_a%d'%i))

        model.build()
        self.model = model
        self.weights = model.get_weights()
        self.dense_layers = filter(lambda x:  x.name.startswith('_d'), model.layers)
        self.hs = [h.output for h in self.dense_layers]
        self.act_layers = filter(lambda x: x.name.startswith('_a'), model.layers)
        self.f_acts = self.f_jac = self.f_jac_hess = self.f_act = None

        vec = K.ones_like(self.model.input)
        self.Js = [T.Rop(h, self.model.input, vec) for h in self.hs]
        self.Hs = [T.Rop(J, self.model.input, vec) for J in self.Js]

        # Need to compile so predict funciton works
        #model.compile('adagrad', 'mse')
        #randomize_model(model, bias_sigma, weight_sigma)

    def compile(self, jacobian=False):
        #self.model.compile('adagrad', 'mse')
        self.f_acts = K.function([self.model.input], self.hs)

    def get_acts(self, xs):
        if self.f_acts is None:
            self.f_acts = K.function([self.model.input], self.hs)
        return self.f_acts((xs,))

    def get_act(self, xs):
        if self.f_act is None:
            self.f_act = K.function([self.model.input], self.hs[-1])
        return self.f_act((xs,))

    def get_jacobians(self, xs):
        assert self.model.input_shape[1] == 1
        if self.f_jac is None:
            self.f_jac = K.function([self.model.input], self.Js)
        return self.f_jac((xs,))

    def get_acts_and_derivatives(self, xs, include_hessian=False):
        assert self.model.input_shape[1] == 1
        if self.f_jac_hess is None:
            if include_hessian:
                self.f_jac_hess = K.function([self.model.input], self.hs + self.Js + self.Hs)
            else:
                self.f_jac_hess = K.function([self.model.input], self.hs + self.Js)
        return self.f_jac_hess((xs,))


    def randomize(self, bias_sigma=None, weight_sigma=None):
        """Randomize the weights and biases in a model.

        Note this overwrites the current weights in the model.
        """

        if bias_sigma is None:
            bias_sigma = self.bias_sigma
        if weight_sigma is None:
            weight_sigma = self.weight_sigma
        self.model.set_weights(randomize_weights(self.weights, bias_sigma, weight_sigma))

    def randomize_trained(self):
        weights = self.model.get_weights()
        rand_weights = randomize_weights(weights)
        self.model.set_weights(weights)

from keras.layers.core import Layer
class GreatCircle(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        U, _, _ = np.linalg.svd(np.random.randn(output_dim, 2), full_matrices=False)
        self.U = K.variable(U.T)
        self.scale = K.variable(1.0)
        kwargs['input_shape'] = (1, )
        super(GreatCircle, self).__init__(**kwargs)

    def set_scale(self, scale):
        self.scale.set_value(np.array(scale).astype(np.float32))

    def call(self, x, mask=None):
        return self.scale * K.dot(K.concatenate((K.cos(x), K.sin(x))), self.U)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

class InterpLine(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.x1 = K.variable(np.random.randn(1, output_dim))
        self.x2 = K.variable(np.random.randn(1, output_dim))
        #U, _, _ = np.linalg.svd(np.random.randn(output_dim, 2), full_matrices=False)
        #self.U = K.variable(U.T)
        self.scale = K.variable(1.0)
        kwargs['input_shape'] = (1, )
        super(InterpLine, self).__init__(**kwargs)

    def set_scale(self, scale):
        self.scale.set_value(np.array(scale).astype(np.float32))

    def set_points(self, x1, x2):
        self.x1.set_value(x1[None, :].astype(np.float32))
        self.x2.set_value(x2[None, :].astype(np.float32))

    def call(self, x, mask=None):
        #return self.x1 * (1.0 - x) + self.x2 * x
        return T.dot(K.cos(x), self.x1) + T.dot(K.sin(x), self.x2)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)



def great_circle(dim, n_interp):
    """Generate circle dataset"""
    ts = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)
    u, _, _ = np.linalg.svd(np.random.randn(dim, 2), full_matrices=False)
    xs = np.dot(u, np.vstack((np.cos(ts), np.sin(ts)))).T
    return ts, xs
