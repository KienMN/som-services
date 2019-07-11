import numpy as np
from .base import CompetitiveNetwork
from .utils import default_learning_rate_decay_function
from .utils import default_sigma_decay_function
from .utils import weights_initialize

class SOM(CompetitiveNetwork):
  """Self organizing map.
  
  Parameters
  ----------
  n_rows : int
    Number of rows in SOM.

  n_cols : int
    Number of cols in SOM.

  Attributes
  ----------
  _quantization_error : 1D numpy array, shape (n_iterations,)
    Quantization error after each iterations, where n_iterations is the number of total iterations.

  _competitive_layer_weights : 2D numpy array, shape (n_neurons, n_features)
    Weights of neurons in competitive layer, where n_neurons is the number of total neurons
    and n_features is the number features of the input data.

  _bias : 1D numpy array, shape (n_neurons)
    Bias of each neurons that is used in conscience.
  """

  def __init__(self, n_rows, n_cols):
    super().__init__(n_rows, n_cols)
    self._quantization_error = np.array([])

  def find_nearest_node(self, x):
    """Find nearest node (best matching unit) according to input vector x.
    
    Parameters
    ----------
    x : 1D numpy array
      The input vector.

    Returns
    -------
    index : int
      Index of nearest node (BMU).
    """
    return np.argmin(np.sum((self._competitive_layer_weights - x) ** 2, axis = 1) + self._bias ** 2)

  def neighborhood_functions(self, win_idx, sigma):
    """Determine neiborhood coefficients.

    Parameters
    ----------
    win_idx : int
      Index of winning neuron.

    sigma : float
      Radius of the neighborhood.

    Returns
    -------
    neighbors : 1D numpy array
      Neighborhood coefficients of each neurons according to the winning neuron.
    """
    win_x, win_y = win_idx // self._n_cols, win_idx % self._n_cols
    X1, X2 = np.meshgrid(np.arange(self._n_cols), np.arange(self._n_rows))
    distance = np.reshape(np.sqrt((X1 - win_y) ** 2 + (X2 - win_x) ** 2), (self._n_nodes,))
    neighbors = np.zeros(self._n_nodes)
    if self._neighborhood == "bubble":
      neighbors[distance <= sigma] = 1
    elif self._neighborhood == "gaussian":
      neighbors = np.exp(- distance / (2 * (sigma ** 2)))
    return neighbors

  def unsup_fitting(self, X, num_iters, batch_size):
    """Fit the model according to the given training data in the unsupervised learning phase.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples in the number of samples and n_features is the number of features.

    num_iters : int
      Number of iterations that unsupervised learning phase passes.

    batch_size : int
      Number of samples in the same batch that uses the same learning rate and neighborhood's radius.

    Returns
    -------
    self : object
      Return self.
    """
    n_samples = X.shape[0]
    verbose = self._verbose
    for i in range(num_iters):
      sample_idx = np.random.randint(0, n_samples)
      x = X[sample_idx]
      n_batches = i // batch_size
      self.update(x = x, batch = n_batches)
      self._quantization_error = np.append(self._quantization_error, self.quantization_error(X))
      if verbose:
        print('Unsupervised learning iteration {}/{}: quantization error: {:.4f}'.format(i + 1, num_iters, self._quantization_error[-1]))
    return self

  def update(self, x, batch):
    """Update competitive layer weights according to the winning node of sample x.

    Parameters
    ----------
    x : 1D numpy array
      The input vector x.

    batch : int
      The sequence number of current batch.

    Returns
    -------
    self : object
      Return self.
    """
    # Computing the learning rate and sigma for current batch.
    learning_rate = self._learning_rate_decay_funtion(self._initial_learning_rate, self._learning_decay_rate, batch)
    sigma = self._sigma_decay_function(self._initial_sigma, self._sigma_decay_rate, batch)
    
    # Determining the winning node and neighborhood coefficients.
    winning_node_idx = self.find_nearest_node(x)
    neighbors = self.neighborhood_functions(winning_node_idx, sigma)
    
    # Reshaping neighborhood coefficients, it needs to have shape of (n_nodes, 1).
    neighbors = neighbors.reshape(-1 ,1)
    
    # Updating competitive layer weights.
    self._competitive_layer_weights = self._competitive_layer_weights + ((x - self._competitive_layer_weights) * learning_rate) * neighbors

    # Updating biases.
    if self._conscience:
      self._bias[:winning_node_idx] *= 0.9
      self._bias[winning_node_idx] += 0.1
      self._bias[winning_node_idx + 1:] *= 0.9
    
  def fit(self, X, weights_init = 'random', num_iters = 100, batch_size = 32, neighborhood = "bubble",
          learning_rate = 0.5, learning_decay_rate = 1, learning_rate_decay_function = None,
          sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
          conscience = False, verbose = False):
    """Fit the model according to the input data.
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.
    
    weights_init : str, options: ['random', 'sample', 'pca'], default: random
      Strategies to initialize the weights of the neurons.

    num_iters : int, default: 100
      Number of iterations that unsupervised learning phase passes.

    batch_size : int, default: 32
      Number of samples in the same batch that uses the same learning rate and neighborhood's radius.

    neighborhood : str, options: ['bubble', 'gaussian']
      Neighborhood function that is used to compute neighborhood coefficients.

    learning_rate : float, default: 0.5
      The learning rate that is used during training process.

    learning_decay_rate : float, default: 1
      The rate that is used to decrease the learning rate after each batches.

    learning_rate_decay_function : function, default: None
      The function that is used to decrease the learning rate after each batches.

    sigma : float, default: 1
      The radius that is used to compute neighborhood coefficients.

    sigma_decay_rate : float, default: 1
      The rate that is used to decrease the sigma after each batches.

    sigma_decay_function : function, default: None
      The function that is used to decrease the sigma after each batches.

    conscience : boolean, default: False
      The technique that tracks the frequency of winning of each neurons to avoid a neuron from winning too many times.

    verbose : boolean, optional, default False
      Verbose mode when fitting the model.

    Returns
    -------
    self : object
      Return self.
    """
    if len(X.shape) != 2:
      raise Exception("Dataset need to be 2 dimensions")

    if weights_init in ['random','sample', 'pca']:
      if verbose:
        print('Using {} initialization for neurons\' weights.'.format(weights_init))
      self._competitive_layer_weights = weights_initialize(X, self._n_rows, self._n_cols, method = weights_init)
    else:
      if verbose:
        print('No weights init specified, using random initialization instead.')
      self._competitive_layer_weights = weights_initialize(X, self._n_rows, self._n_cols, method = 'random')  
    
    self._initial_learning_rate = learning_rate
    self._learning_decay_rate = learning_decay_rate
    self._initial_sigma = sigma
    self._sigma_decay_rate = sigma_decay_rate
    self._conscience = conscience
    self._bias = np.zeros(self._n_nodes)
    self._neighborhood = neighborhood
    self._verbose = verbose

    if learning_rate_decay_function:
      self._learning_rate_decay_funtion = learning_rate_decay_function
    else:
      self._learning_rate_decay_funtion = default_learning_rate_decay_function

    if sigma_decay_function:
      self._sigma_decay_function = sigma_decay_function
    else:
      self._sigma_decay_function = default_sigma_decay_function

    self.unsup_fitting(X, num_iters, batch_size)
    return self

  def predict(self, X):
    """Unsupervised learning model has no predict method."""
    pass

  def quantization_error(self, X):
    """
    Returns quantization error computed from the average distance of input vectors, x to its best matching unit.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    quantization_error : float
      Quantization error of the network according to the input data.
    """
    if len(X.shape) != 2:
      raise Exception("Dataset need to be 2 dimensions")
    error = 0
    for x in X:
      error += np.sqrt(np.sum((x - self._competitive_layer_weights[self.find_nearest_node(x)]) ** 2))
    return error / len(X)

class CombineSomLvq(SOM):
  """Combination of SOM and LVQ.
  
  Parameters
  ----------
  n_rows : int
    Number of rows in SOM.

  n_cols : int
    Number of cols in SOM.

  Attributes
  ----------
  _quantization_error : 1D numpy array, shape (n_iterations,)
    Quantization error after each iterations, where n_iterations is the number of total iterations.

  _competitive_layer_weights : 2D numpy array, shape (n_neurons, n_features)
    Weights of neurons in competitive layer, where n_neurons is the number of total neurons
    and n_features is the number features of the input data.

  _bias : 1D numpy array, shape (n_neurons,)
    Bias of each neurons that is used in conscience.

  _nodes_label : 1D numpy array, shape (n_neurons,)
    Label of each neurons.
  """

  def __init__(self, n_rows, n_cols):
    super().__init__(n_rows, n_cols)

  def label_nodes(self, X, y, labels_init = None):
    """Label the neurons according to the input data and current weights of the neurons.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples in the number of samples and n_features is the number of features.

    y : 1D numpy array, shape (n_samples,)
      Training label vector, where n_samples in the number of samples.

    labels_init : str, default: None
      Criteria to initialize the label of the neurons.
      The default criteria is depending on the most occurence label of nearest data points.

    Returns
    -------
    self : object
      Return self.
    """
    self._nodes_label = np.zeros(self._n_nodes).astype(np.int8)
    m = min(20, X.shape[0])
    for i in range (self._n_nodes):
      near_samples_idx = np.argpartition(np.sum((self._competitive_layer_weights[i] - X) ** 2, axis = 1), m - 1)[:m]
      l = np.argmax(np.bincount(y[near_samples_idx]))
      self._nodes_label[i] = l

  def sup_fitting(self, X, y, num_iters, batch_size, n_trained_batches):
    """Fit the model according to the given training data in the supervised learning phase.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples in the number of samples and n_features is the number of features.

    y : 1D numpy array, shape (n_samples,)
      Training label vector, where n_samples in the number of samples.

    num_iters : int
      Number of iterations that supervised learning phase passes.

    batch_size : int
      Number of samples in the same batch that uses the same learning rate and neighborhood's radius.

    n_trained_batches : int
      Number of trained batches in the unsupervised learning phase.

    Returns
    -------
    self : object
      Return self.
    """
    n_samples = X.shape[0]
    verbose = self._verbose
    for i in range(num_iters):
      sample_idx = np.random.randint(0, n_samples)
      x = X[sample_idx]
      y_i = y[sample_idx]
      n_batches = n_trained_batches + i // batch_size
      self.sup_update(x = x, y = y_i, batch = n_batches)
      self._quantization_error = np.append(self._quantization_error, self.quantization_error(X))
      if verbose:
        print('Supervised learning iteration {}/{}: quantization error: {:.4f}'.format(i + 1, num_iters, self._quantization_error[-1]))
    return self

  def sup_update(self, x, y, batch):
    """Update competitive layer weights according to the winning node of sample x.

    Parameters
    ----------
    x : 1D numpy array
      The input vector x.

    y : int
      The label of the input vector x.

    batch : int
      The sequence number of current batch.

    Returns
    -------
    self : object
      Return self.
    """
    # Computing the learning rate and sigma for current batch.
    learning_rate = self._learning_rate_decay_funtion(self._initial_learning_rate, self._learning_decay_rate, batch)
    sigma = self._sigma_decay_function(self._initial_sigma, self._sigma_decay_rate, batch)
    
    # Determining the winning node and neighborhood coefficients.
    winning_node_idx = self.find_nearest_node(x)
    neighbors = self.neighborhood_functions(winning_node_idx, sigma)
    
    # neighborhood coefficients need to have shape of (n_nodes, 1)
    neighbors = neighbors.reshape(-1 ,1)
    
    # Sign
    sign = np.ones((self._n_nodes, 1))
    sign[self._nodes_label == y] = 1
    sign[self._nodes_label == y] = -1 / 3

    # Updating competitive layer weights
    self._competitive_layer_weights = self._competitive_layer_weights + ((x - self._competitive_layer_weights) * learning_rate) * neighbors * sign

    # Updating bias
    if self._conscience:
      self._bias[:winning_node_idx] *= 0.9
      self._bias[winning_node_idx] += 0.1
      self._bias[winning_node_idx + 1:] *= 0.9

  def fit(self, X, y, weights_init = 'random', labels_init = None,
          unsup_num_iters = 100, unsup_batch_size = 32,
          sup_num_iters = 100, sup_batch_size = 32,
          neighborhood = "bubble",
          learning_rate = 0.5, learning_decay_rate = 1, learning_rate_decay_function = None,
          sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
          conscience = False, verbose = False):
    """Fit the model according to the input data.
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.

    y : 1D numpy array, shape (n_samples,)
      Training label vector, where n_samples in the number of samples.

    weights_init : str, options: ['random', 'sample', 'pca'], default: random
      Strategies to initialize the weights of the neurons.

    labels_init : str, default: None
      Criteria to initialize the labels of the neurons.

    unsup_num_iters : int
      Number of iterations that unsupervised learning phase passes.

    unsup_batch_size : int
      Number of samples in the same batch during unsupervised learning phase,
      that uses the same learning rate and neighborhood's radius.

    sup_num_iters : int
      Number of iterations that supervised learning phase passes.

    sup_batch_size : int
      Number of samples in the same batch during supervised learning phase,
      that uses the same learning rate and neighborhood's radius.

    neighborhood : str, options: ['bubble', 'gaussian']
      Neighborhood function that is used to compute neighborhood coefficients.

    learning_rate : float, default: 0.5
      The learning rate that is used during training process.

    learning_decay_rate : float, default: 1
      The rate that is used to decrease the learning rate after each batches.

    learning_rate_decay_function : function, default: None
      The function that is used to decrease the learning rate after each batches.

    sigma : float, default: 1
      The radius that is used to compute neighborhood coefficients.

    sigma_decay_rate : float, default: 1
      The rate that is used to decrease the sigma after each batches.

    sigma_decay_function : function, default: None
      The function that is used to decrease the sigma after each batches.

    conscience : boolean, default: False
      The technique that tracks the frequency of winning of each neurons to avoid a neuron from winning too many times.

    verbose : boolean, optional, default False
      Verbose mode when fitting the model.

    Returns
    -------
    self : object
      Return self.
    """
    print(X, y)
    if len(X.shape) != 2:
      raise Exception("Dataset need to be 2 dimensions")
      
    if weights_init in ['random','sample', 'pca']:
      if verbose:
        print('Using {} initialization for neurons\' weights.'.format(weights_init))
      self._competitive_layer_weights = weights_initialize(X, self._n_rows, self._n_cols, method = weights_init)
    else:
      if verbose:
        print('No weights init specified, using random initialization instead.')
      self._competitive_layer_weights = weights_initialize(X, self._n_rows, self._n_cols, method = 'random')

    self._initial_learning_rate = learning_rate
    self._initial_sigma = sigma
    self._learning_decay_rate = learning_decay_rate
    self._sigma_decay_rate = sigma_decay_rate
    self._conscience = conscience
    self._bias = np.zeros(self._n_nodes)
    self._neighborhood = neighborhood
    self._verbose = verbose
    
    if learning_rate_decay_function:
      self._learning_rate_decay_funtion = learning_rate_decay_function
    else:
      self._learning_rate_decay_funtion = default_learning_rate_decay_function

    if sigma_decay_function:
      self._sigma_decay_function = sigma_decay_function
    else:
      self._sigma_decay_function = default_sigma_decay_function
    
    # Unsupervised learning phase
    self.unsup_fitting(X, unsup_num_iters, unsup_batch_size)

    # Supervised learning phase
    self.label_nodes(X, y, labels_init)
    self.sup_fitting(X, y, sup_num_iters, sup_batch_size, n_trained_batches = unsup_num_iters // unsup_batch_size)
    self.label_nodes(X, y, labels_init)

  def predict(self, X, confidence_score = False, distance_to_bmu = False):
    """Predict using the fitted model.
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Input vectors, where n_samples is the number of samples and n_features is the number of features.

    confidence_score : boolean, default: False
      Return confidence score of not.
    
    distance_to_bmu : boolean, default: False
      Return distance to the BMU of each input vectors or not.

    Returns
    -------
    y_pred : 1D numpy array, shape (n_samples,)
      Predicted label vector, where n_samples in the number of samples.

    score : 1D numpy array, shape (n_samples,)
      Confidence score corresponding to the prediction.

    distance : 1D numpy array, shape (n_samples,)
      Distance of each input vectors to its corresponding BMU.
    """

    n_samples = X.shape[0]
    y_pred = np.zeros(n_samples).astype(np.int8)
    score = np.zeros(n_samples)
    distances = np.zeros(n_samples)

    if confidence_score or distance_to_bmu:
      m = min(self._n_nodes, 5)
      for i in range (n_samples):
        winning_node_idx = self.find_nearest_node(X[i])
        y_pred[i] = self._nodes_label[winning_node_idx]
        square_distances = np.sum((self._competitive_layer_weights - X[i]) ** 2, axis = 1)
        distances[i] = np.sqrt(square_distances[winning_node_idx])
        near_node_idx = np.argpartition(square_distances, m - 1)[:m]
        total_inverse_distance = 0
        for j in near_node_idx:
          if square_distances[j] == 0:
            if self._nodes_label[j] == y_pred[i]:
              score[i] = 1
            else:
              score[i] = 0
            total_inverse_distance = 1
            break
          if self._nodes_label[j] == y_pred[i]:
            score[i] += 1 / np.sqrt(square_distances[j])
          total_inverse_distance += 1 / np.sqrt(square_distances[j])
        score[i] /= total_inverse_distance
      if confidence_score and distance_to_bmu:
        return y_pred, score, distances
      elif confidence_score:
        return y_pred, score
      elif distance_to_bmu:
        return y_pred, distances
    else:
      for i in range (n_samples):
        winning_node_idx = self.find_nearest_node(X[i])
        y_pred[i] = self._nodes_label[winning_node_idx]
      return y_pred