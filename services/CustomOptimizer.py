from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class Optimizer(object):
  """Abstract optimizer base class.

  Note: this is the parent class of all optimizers, not an actual optimizer
  that can be used for training models.

  All Keras optimizers support the following keyword arguments:

      clipnorm: float >= 0. Gradients will be clipped
          when their L2 norm exceeds this value.
      clipvalue: float >= 0. Gradients will be clipped
          when their absolute value exceeds this value.
  """

  def __init__(self, **kwargs):
    allowed_kwargs = {'clipnorm', 'clipvalue'}
    for k in kwargs:
      if k not in allowed_kwargs:
        raise TypeError('Unexpected keyword argument '
                        'passed to optimizer: ' + str(k))
      # checks that clipnorm >= 0 and clipvalue >= 0
      if kwargs[k] < 0:
        raise ValueError('Expected {} >= 0, received: {}'.format(k, kwargs[k]))
    self.__dict__.update(kwargs)
    self.updates = []
    self.weights = []

  def get_updates(self, loss, params):
    raise NotImplementedError

  def get_gradients(self, loss, params):
    """Returns gradients of `loss` with respect to `params`.

    Arguments:
        loss: Loss tensor.
        params: List of variables.

    Returns:
        List of gradient tensors.

    Raises:
        ValueError: In case any gradient cannot be computed (e.g. if gradient
          function not implemented).
    """
    grads = K.gradients(loss, params)
    if None in grads:
      raise ValueError('An operation has `None` for gradient. '
                       'Please make sure that all of your ops have a '
                       'gradient defined (i.e. are differentiable). '
                       'Common ops without gradient: '
                       'K.argmax, K.round, K.eval.')
    if hasattr(self, 'clipnorm'):
      grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
    if hasattr(self, 'clipvalue'):
      grads = [
          clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
          for g in grads
      ]
    return grads

  def set_weights(self, weights):
    """Sets the weights of the optimizer, from Numpy arrays.

    Should only be called after computing the gradients
    (otherwise the optimizer has no weights).

    Arguments:
        weights: a list of Numpy arrays. The number
            of arrays and their shape must match
            number of the dimensions of the weights
            of the optimizer (i.e. it should match the
            output of `get_weights`).

    Raises:
        ValueError: in case of incompatible weight shapes.
    """
    params = self.weights
    if len(params) != len(weights):
      raise ValueError(
          'Length of the specified weight list (' + str(len(weights)) +
          ') does not match the number of weights '
          'of the optimizer (' + str(len(params)) + ')')
    weight_value_tuples = []
    param_values = K.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError(
            'Optimizer weight shape ' + str(pv.shape) + ' not compatible with '
            'provided weight shape ' + str(w.shape))
      weight_value_tuples.append((p, w))
    K.batch_set_value(weight_value_tuples)

  def get_weights(self):
    """Returns the current value of the weights of the optimizer.

    Returns:
        A list of numpy arrays.
    """
    return K.batch_get_value(self.weights)

  def get_config(self):
    config = {}
    if hasattr(self, 'clipnorm'):
      config['clipnorm'] = self.clipnorm
    if hasattr(self, 'clipvalue'):
      config['clipvalue'] = self.clipvalue
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class RMSPropWithMomentum(Optimizer):
  def __init__(self, lr=0.01, epsilon=None, decay=0., momentum=0., **kwargs):
    super(RMSPropWithMomentum, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.lr = K.variable(lr, name='lr')
      self.decay = K.variable(decay, name='decay')
      self.iterations = K.variable(0, dtype='int64', name='iterations')
    if epsilon is None:
      epsilon = K.epsilon()
    self.momentum = momentum
    self.epsilon = epsilon
    self.initial_decay = decay

  def adagrad(self, a, g):
    new_a = a + math_ops.square(g)  # update accumulator
    self.updates.append(state_ops.assign(a, new_a))
    scaled_gradient = g / (K.sqrt(new_a) + self.epsilon)
    return scaled_gradient

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    shapes = [K.int_shape(p) for p in params]
    accumulators = [K.zeros(shape) for shape in shapes]
    vs = [K.zeros(shape) for shape in shapes]
    self.updates = [state_ops.assign_add(self.iterations, 1)]
    self.weights = accumulators + vs

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    for p, g, a, v in zip(params, grads, accumulators, vs):
      g = self.adagrad(a, g)
      if self.momentum > 0:
        v2 = self.momentum * v - lr * g
        self.updates.append(state_ops.assign(v, v2))
        new_p = p + v2
      else:
        new_p = p - lr * g

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(RMSPropWithMomentum, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

