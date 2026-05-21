# %% [markdown]

import contextlib
import functools
import os
import time

import numpy as np
import pandas as pd
import scipy as sp
from six.moves import urllib
from sklearn import preprocessing

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp

# %% [markdown]
# ## BFGS and L-BFGS Optimizers
# 
# Quasi Newton methods are a class of popular first order optimization algorithm. These methods use a positive definite approximation to the exact Hessian to find the search direction.
# 
# The Broyden-Fletcher-Goldfarb-Shanno
# algorithm ([BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)) is a specific implementation of this general idea. It is applicable and is the method of choice for medium sized problems
# where the gradient is continuous everywhere (e.g. linear regression with an $L_2$ penalty).
# 
# [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) is a limited-memory version of BFGS that is useful for solving larger problems whose Hessian matrices cannot be computed at a reasonable cost or are not sparse. Instead of storing fully dense $n \times n$ approximations of Hessian matrices, they only save a few vectors of length $n$ that represent these approximations implicitly.
# 

# %%
#@title Helper functions

CACHE_DIR = os.path.join(os.sep, 'tmp', 'datasets')


def make_val_and_grad_fn(value_fn):
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return tfp.math.value_and_gradient(value_fn, x)
  return val_and_grad


@contextlib.contextmanager
def timed_execution():
  t0 = time.time()
  yield
  dt = time.time() - t0
  print('Evaluation took: %f seconds' % dt)


def np_value(tensor):
  """Get numpy value out of possibly nested tuple of tensors."""
  if isinstance(tensor, tuple):
    return type(tensor)(*(np_value(t) for t in tensor))
  else:
    return tensor.numpy()

def run(optimizer):
  """Run an optimizer and measure it's evaluation time."""
  optimizer()  # Warmup.
  with timed_execution():
    result = optimizer()
  return np_value(result)

# %% [markdown]
# ### L-BFGS on  a simple quadratic function

# %%
# Fix numpy seed for reproducibility
np.random.seed(12345)

# The objective must be supplied as a function that takes a single
# (Tensor) argument and returns a tuple. The first component of the
# tuple is the value of the objective at the supplied point and the
# second value is the gradient at the supplied point. The value must
# be a scalar and the gradient must have the same shape as the
# supplied argument.

# The `make_val_and_grad_fn` decorator helps transforming a function
# returning the objective value into one that returns both the gradient
# and the value. It also works for both eager and graph mode.

dim = 10
minimum = np.ones([dim])
scales = np.exp(np.random.randn(dim))

@make_val_and_grad_fn
def quadratic(x):
  return tf.reduce_sum(scales * (x - minimum) ** 2, axis=-1)

# The minimization routine also requires you to supply an initial
# starting point for the search. For this example we choose a random
# starting point.
start = np.random.randn(dim)

# Finally an optional argument called tolerance let's you choose the
# stopping point of the search. The tolerance specifies the maximum
# (supremum) norm of the gradient vector at which the algorithm terminates.
# If you don't have a specific need for higher or lower accuracy, leaving
# this parameter unspecified (and hence using the default value of 1e-8)
# should be good enough.
tolerance = 1e-10

@tf.function
def quadratic_with_lbfgs():
  return tfp.optimizer.lbfgs_minimize(
    quadratic,
    initial_position=tf.constant(start),
    tolerance=tolerance)

results = run(quadratic_with_lbfgs)

# The optimization results contain multiple pieces of information. The most
# important fields are: 'converged' and 'position'.
# Converged is a boolean scalar tensor. As the name implies, it indicates
# whether the norm of the gradient at the final point was within tolerance.
# Position is the location of the minimum found. It is important to check
# that converged is True before using the value of the position.

print('L-BFGS Results')
print('Converged:', results.converged)
print('Location of the minimum:', results.position)
print('Number of iterations:', results.num_iterations)

# %% [markdown]
# ### Same problem with BFGS

# %%
@tf.function
def quadratic_with_bfgs():
  return tfp.optimizer.bfgs_minimize(
    quadratic,
    initial_position=tf.constant(start),
    tolerance=tolerance)

results = run(quadratic_with_bfgs)

print('BFGS Results')
print('Converged:', results.converged)
print('Location of the minimum:', results.position)
print('Number of iterations:', results.num_iterations)

# %% [markdown]
# ## Linear Regression with L1 penalty: Prostate Cancer data
# 
# Example from the Book: *The Elements of Statistical Learning, Data Mining, Inference, and Prediction* by Trevor Hastie, Robert Tibshirani and Jerome Friedman.
# 
# Note this is an optimization problem with L1 penalty.

# %% [markdown]
# ### Obtain dataset

# %%
def cache_or_download_file(cache_dir, url_base, filename):
  """Read a cached file or download it."""
  filepath = os.path.join(cache_dir, filename)
  if tf.io.gfile.exists(filepath):
    return filepath
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.makedirs(cache_dir)
  url = url_base + filename
  print("Downloading {url} to {filepath}.".format(url=url, filepath=filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath

def get_prostate_dataset(cache_dir=CACHE_DIR):
  """Download the prostate dataset and read as Pandas dataframe."""
  url_base = 'http://web.stanford.edu/~hastie/ElemStatLearn/datasets/'
  return pd.read_csv(
      cache_or_download_file(cache_dir, url_base, 'prostate.data'),
      delim_whitespace=True, index_col=0)

prostate_df = get_prostate_dataset()

# %% [markdown]
# ### Problem definition

# %%
np.random.seed(12345)

feature_names = ['lcavol', 'lweight',	'age',	'lbph',	'svi', 'lcp',	
                 'gleason',	'pgg45']

# Normalize features
scalar = preprocessing.StandardScaler()
prostate_df[feature_names] = pd.DataFrame(
    scalar.fit_transform(
        prostate_df[feature_names].astype('float64')))

# select training set
prostate_df_train = prostate_df[prostate_df.train == 'T'] 

# Select features and labels 
features = prostate_df_train[feature_names]
labels =  prostate_df_train[['lpsa']]

# Create tensors
feat = tf.constant(features.values, dtype=tf.float64)
lab = tf.constant(labels.values, dtype=tf.float64)

dtype = feat.dtype

regularization = 0 # regularization parameter
dim = 8 # number of features

# We pick a random starting point for the search
start = np.random.randn(dim + 1)

def regression_loss(params):
  """Compute loss for linear regression model with L1 penalty

  Args:
    params: A real tensor of shape [dim + 1]. The zeroth component
      is the intercept term and the rest of the components are the
      beta coefficients.

  Returns:
    The mean square error loss including L1 penalty.
  """
  params = tf.squeeze(params)
  intercept, beta  = params[0], params[1:]
  pred = tf.matmul(feat, tf.expand_dims(beta, axis=-1)) + intercept
  mse_loss = tf.reduce_sum(
      tf.cast(
        tf.losses.mean_squared_error(y_true=lab, y_pred=pred), tf.float64))
  l1_penalty = regularization * tf.reduce_sum(tf.abs(beta))
  total_loss = mse_loss + l1_penalty
  return total_loss

# %% [markdown]
# ### Solving with L-BFGS
# 
# Fit using L-BFGS. Even though the L1 penalty introduces derivative discontinuities, in practice, L-BFGS works quite well still.
# 

# %%
@tf.function
def l1_regression_with_lbfgs():
  return tfp.optimizer.lbfgs_minimize(
    make_val_and_grad_fn(regression_loss),
    initial_position=tf.constant(start),
    tolerance=1e-8)

results = run(l1_regression_with_lbfgs)
minimum = results.position
fitted_intercept = minimum[0]
fitted_beta = minimum[1:]

print('L-BFGS Results')
print('Converged:', results.converged)
print('Intercept: Fitted ({})'.format(fitted_intercept))
print('Beta:      Fitted {}'.format(fitted_beta))

# %% [markdown]
# ### Solving with Nelder Mead
# 
# The [Nelder Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) is one of the most popular derivative free minimization methods. This optimizer doesn't use gradient information and makes no assumptions on the differentiability of the target function; it is therefore appropriate for non-smooth objective functions, for example optimization problems with L1 penalty.
# 
# For an optimization problem in $n$-dimensions it maintains a set of
# $n+1$ candidate solutions that span a non-degenerate simplex. It successively modifies the simplex based on a set of moves (reflection, expansion, shrinkage and contraction) using the function values at each of the vertices.
# 

# %%
# Nelder mead expects an initial_vertex of shape [n + 1, 1].
initial_vertex = tf.expand_dims(tf.constant(start, dtype=dtype), axis=-1)

@tf.function
def l1_regression_with_nelder_mead():
  return tfp.optimizer.nelder_mead_minimize(
      regression_loss,
      initial_vertex=initial_vertex,
      func_tolerance=1e-10,
      position_tolerance=1e-10)

results = run(l1_regression_with_nelder_mead)
minimum = results.position.reshape([-1])
fitted_intercept = minimum[0]
fitted_beta = minimum[1:]

print('Nelder Mead Results')
print('Converged:', results.converged)
print('Intercept: Fitted ({})'.format(fitted_intercept))
print('Beta:      Fitted {}'.format(fitted_beta))

# %% [markdown]
# ## Logistic Regression with L2 penalty
# 
# For this example, we create a synthetic data set for classification and use the L-BFGS optimizer to fit the parameters.

# %%
np.random.seed(12345)

dim = 5  # The number of features
n_obs = 10000  # The number of observations

betas = np.random.randn(dim)  # The true beta
intercept = np.random.randn()  # The true intercept

features = np.random.randn(n_obs, dim)  # The feature matrix
probs = sp.special.expit(
    np.matmul(features, np.expand_dims(betas, -1)) + intercept)

labels = sp.stats.bernoulli.rvs(probs)  # The true labels

regularization = 0.8
feat = tf.constant(features)
lab = tf.constant(labels, dtype=feat.dtype)

@make_val_and_grad_fn
def negative_log_likelihood(params):
  """Negative log likelihood for logistic model with L2 penalty

  Args:
    params: A real tensor of shape [dim + 1]. The zeroth component
      is the intercept term and the rest of the components are the
      beta coefficients.

  Returns:
    The negative log likelihood plus the penalty term. 
  """
  intercept, beta  = params[0], params[1:]
  logit = tf.matmul(feat, tf.expand_dims(beta, -1)) + intercept
  log_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
      labels=lab, logits=logit))
  l2_penalty = regularization * tf.reduce_sum(beta ** 2)
  total_loss = log_likelihood + l2_penalty
  return total_loss

start = np.random.randn(dim + 1)

@tf.function
def l2_regression_with_lbfgs():
  return tfp.optimizer.lbfgs_minimize(
      negative_log_likelihood,
      initial_position=tf.constant(start),
      tolerance=1e-8)

results = run(l2_regression_with_lbfgs)
minimum = results.position
fitted_intercept = minimum[0]
fitted_beta = minimum[1:]

print('Converged:', results.converged)
print('Intercept: Fitted ({}), Actual ({})'.format(fitted_intercept, intercept))
print('Beta:\n\tFitted {},\n\tActual {}'.format(fitted_beta, betas))

# %% [markdown]
# ## Batching support
# 
# Both BFGS and L-BFGS support batched computation, for example to optimize a single function from many different starting points; or multiple parametric functions from a single point.

# %% [markdown]
# ### Single function, multiple starting points
# 
# Himmelblau's function is a standard optimization test case. The function is given by:
# 
# $$f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$$
# 
# The function has four minima located at:
# - (3, 2),
# - (-2.805118, 3.131312),
# - (-3.779310, -3.283186),
# - (3.584428, -1.848126).
# 
# All these minima may be reached from appropriate starting points.
# 

# %%
# The function to minimize must take as input a tensor of shape [..., n]. In
# this n=2 is the size of the domain of the input and [...] are batching
# dimensions. The return value must be of shape [...], i.e. a batch of scalars
# with the objective value of the function evaluated at each input point.

@make_val_and_grad_fn
def himmelblau(coord):
  x, y = coord[..., 0], coord[..., 1]
  return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2

starts = tf.constant([[1, 1],
                      [-2, 2],
                      [-1, -1],
                      [1, -2]], dtype='float64')

# The stopping_condition allows to further specify when should the search stop.
# The default, tfp.optimizer.converged_all, will proceed until all points have
# either converged or failed. There is also a tfp.optimizer.converged_any to
# stop as soon as the first point converges, or all have failed.

@tf.function
def batch_multiple_starts():
  return tfp.optimizer.lbfgs_minimize(
      himmelblau, initial_position=starts,
      stopping_condition=tfp.optimizer.converged_all,
      tolerance=1e-8)

results = run(batch_multiple_starts)
print('Converged:', results.converged)
print('Minima:', results.position)

# %% [markdown]
# ### Multiple functions
# 
# For demonstration purposes, in this example we simultaneously optimize a large number of high dimensional randomly generated quadratic bowls.

# %%
np.random.seed(12345)

dim = 100
batches = 500
minimum = np.random.randn(batches, dim)
scales = np.exp(np.random.randn(batches, dim))

@make_val_and_grad_fn
def quadratic(x):
  return tf.reduce_sum(input_tensor=scales * (x - minimum)**2, axis=-1)

# Make all starting points (1, 1, ..., 1). Note not all starting points need
# to be the same.
start = tf.ones((batches, dim), dtype='float64')

@tf.function
def batch_multiple_functions():
  return tfp.optimizer.lbfgs_minimize(
      quadratic, initial_position=start,
      stopping_condition=tfp.optimizer.converged_all,
      max_iterations=100,
      tolerance=1e-8)

results = run(batch_multiple_functions)
print('All converged:', np.all(results.converged))
print('Largest error:', np.max(results.position - minimum))



