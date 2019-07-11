import requests
import json

def test_create():
  base_url = 'http://127.0.0.1:5000/api'
  model_id = 'test_01'
  body = {
    'model_id': model_id,
    'n_rows': 3,
    'n_cols': 3
  }
  res = requests.post(base_url + '/model/create/som', data = json.dumps(body), headers = {'Content-Type': 'application/json'})
  print('Test create:')
  print('Resquest:', json.dumps(body))
  print('Response:', res.json())

def test_train():
  base_url = 'http://127.0.0.1:5000/api'
  model_id = 'test_01'
  body = {
    'model_id': model_id,
    'features': [[1, 1, 2], [2, 1, 3], [1, 2, 3], [1, 2, 2]],
    'target': [1, 2, 2],
    'weights_init': 'random',
    'unsup_num_iters': 10,
    'sup_num_iters': 10
  }
  res = requests.post(base_url + '/model/train', data = json.dumps(body), headers = {'Content-Type': 'application/json'})
  print('Test train:')
  print('Resquest:', json.dumps(body))
  print('Response:', res.json())

def test_predict():
  base_url = 'http://127.0.0.1:5000/api'
  model_id = 'test_01'
  body = {
    'model_id': model_id,
    'features': [[1, 1, 2], [2, 1, 3], [1, 2, 3], [1, 2, 2]]
  }
  res = requests.post(base_url + '/model/predict', data = json.dumps(body), headers = {'Content-Type': 'application/json'})
  print('Test predict:')
  print('Resquest:', json.dumps(body))
  print('Response:', res.json())

if __name__ == '__main__':
  test_create()
  test_train()
  test_predict()