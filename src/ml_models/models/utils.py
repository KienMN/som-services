import numpy as np
from sklearn.decomposition import PCA

def default_learning_rate_decay_function(learning_rate_0, learning_decay_rate, batch):
  """Function to decrease learning rate after number of batches.
  
  Parameters
  ----------
  learning_rate_0 : float
    Initial learning rate.

  learning_decay_rate : float
    Rate that is used to decrease learning rate.

  batch : int
    Sequence number of current batch.

  Returns
  -------
  learning_rate : float
    Learning rate after being decreased.
  """
  return (learning_rate_0) / (1 + batch * learning_decay_rate)

def default_sigma_decay_function(sigma_0, sigma_decay_rate, batch):
  """Function to decrease sigma after number of batches.
  
  Parameters
  ----------
  sigma_0 : float
    Initial sigma.

  sigma_decay_rate : float
    Rate that is used to decrease sigma.

  batch : int
    Sequence number of current batch.

  Returns
  -------
  sigma : float
    Sigma after being decreased.
  """
  return (sigma_0) / (1 + batch * sigma_decay_rate)

def weights_initialize(data, n_rows, n_cols, method = 'random'):
  """Function to initialize weights of neurons.
  
  Parameters
  ----------
  data : 2D numpy array, shape (n_samples, n_features)
    The input dataset, where n_samples is the number of samples and n_features is the number of features.

  n_rows : int
    Number of rows of the network.

  n_cols : int
    Number of columns of the network.

  method : str, options: ['random', 'sample', 'pca'], default: random
    Method to initialize the weights.
    'random': weights are initialized randomly.
    'sample': weights are initialized by picking random samples from the dataset.
    'pca': weights are initialized by applying pca on the dataset.

  Returns
  -------
  weights : 2D numpy array, shape (n_nodes, n_features)
    Weights of neurons in the network, where n_nodes is the total number of neurons
    and n_features is the number of features.
  """
  n_nodes = n_rows * n_cols
  n_samples, n_features = data.shape

  if method not in ['random', 'sample', 'pca']:
    raise Exception('No method specified')

  weights = np.random.rand(n_nodes, n_features)

  if method == 'sample':
    for i in range (n_nodes):
      rand_idx = np.random.randint(0, n_samples)
      weights[i] = data[rand_idx].copy()
  elif method == 'pca':
    # Pca parameters
    pca_number_of_components = None
    coord = None

    if n_cols == 1 or n_rows == 1 or data.shape[1] == 1:
      pca_number_of_components = 1
      if n_cols == 1 and n_rows == 1:
        coord = np.array([[1], [0]])
        # print(coord)
        # print(coord[0][0])
      else:  
        coord = np.zeros((n_nodes, 1))
        for i in range (n_nodes):
          coord[i][0] = i
    else:
      pca_number_of_components = 2
      coord = np.zeros((n_nodes, 2))
      for i in range (n_nodes):
        coord[i][0] = i // n_cols
        coord[i][1] = i % n_cols
    
    mx = np.amax(coord, axis = 0)
    mn = np.amin(coord, axis = 0)
    coord = (coord - mn) / (mx - mn)
    coord = (coord - 0.5) * 2
    pca = PCA(n_components = pca_number_of_components)
    pca.fit(data)
    eigvec = pca.components_
    # print(eigvec)
    # print(coord)
    for i in range (n_nodes):
      for j in range (eigvec.shape[0]):
        weights[i] = coord[i][j] * eigvec[j]
      # if fast_norm(self._competitive_layer_weights[i]) == 0:
      #   weights[i] = 0.01 * eigvec[0]

  return weights