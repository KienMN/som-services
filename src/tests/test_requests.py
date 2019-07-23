import requests
import json
import pytest
import sys
from sklearn.datasets import load_iris

@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_create():
  base_url = 'http://127.0.0.1:6789/api'
  model_id = 'test_01'
  body = {
    'model_id': model_id,
    'n_rows': 10,
    'n_cols': 10
  }
  res = requests.post(base_url + '/model/create/som', data = json.dumps(body), headers = {'Content-Type': 'application/json'})
  print()
  print('Test create:')
  print('Resquest:', json.dumps(body))
  print('Response:', res.json())

  assert res.status_code == 201

@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_train():
  base_url = 'http://127.0.0.1:6789/api'
  model_id = 'test_01'
  X, y = load_iris(return_X_y=True)
  body = {
    'model_id': model_id,
    #'features': [[1, 1, 2], [2, 1, 3], [1, 2, 3], [1, 2, 2]],
    #'target': [1, 2, 2],
    'features': X.T.tolist(),
    'target': y.tolist(),
    'weights_init': 'sample',
    'unsup_num_iters': 200,
    'sup_num_iters': 200
  }
  res = requests.post(base_url + '/model/train', data = json.dumps(body), headers = {'Content-Type': 'application/json'})
  print()
  print('Test create:')
  print('Resquest:', json.dumps(body))
  print('Response:', res.json())
  assert res.status_code == 201

@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_predict():
  base_url = 'http://127.0.0.1:6789/api'
  model_id = 'test_01'
  X, _ = load_iris(return_X_y=True)
  body = {
    'model_id': model_id,
    #'features': [[1, 1, 2], [2, 1, 3], [1, 2, 3], [1, 2, 2]]
    'features': X.T.tolist()
  }
  res = requests.post(base_url + '/model/predict', data = json.dumps(body), headers = {'Content-Type': 'application/json'})
  print()
  print('Test create:')
  print('Resquest:', json.dumps(body))
  print('Response:', res.json())
  assert res.status_code == 201 and len(res.json()['target']) == len(body['features'][0])
