import secrets

from src import model_helper as helper
from src import config
from src import ml_models

from .create import *
import numpy as np

import flask

@helper.parse_body_request
def train(*args, **kwargs):
    body = kwargs.get('body')
    model_id = body.get('model_id')
    features = body.get('features')
    target = body.get('target')
    model_params_name = ['weights_init', 'unsup_num_iters', 'unsup_batch_size', 'sup_num_iters', 'sup_batch_size',
                        'neighborhood', 'learning_rate', 'learning_decay_rate', 'sigma', 'sigma_decay_rate']
    model_params = {'model__{}'.format(p): v for p, v in body.items() if p in model_params_name}
    # print(model_params)
    model_params['model__verbose'] = True

    X_train = np.array(features).T
    y_train = np.array(target)

    try:
        result_ml = helper.model_train(model_id, X_train, y_train, **model_params)
    except Exception as err:
        config.logger.error(str(err))
        err_message = ml_models.result.ErrorResult()
        return err_message()
    else:
        success_message = ml_models.get_result(model_id)(**result_ml)
        return success_message()

@helper.parse_body_request
def predict(model_id, features):
    X = np.array(features).T
    try:
        target = helper.model_predict(model_id, X)
    except Exception as err:
        config.logger.error(str(err))
        err_message = ml_models.result.ErrorResult()
        return err_message()
    else:
        success_message = ml_models.result.SuccessResult()
        success_message.add("target", target)
        return success_message()

def get_model(model_id):
    try:
        model = helper.model_load(model_id)
    except Exception as err:
        config.logger.error(str(err))
        err_message = ml_models.result.ErrorResult()
        return err_message()
    else:
        # print(model.named_steps)
        success_message = ml_models.result.SuccessResult()
        scaler = model.named_steps['minmax-scaler']
        model = model.named_steps['model']
        n_rows = model._n_rows
        n_cols = model._n_cols
        n_features = model._competitive_layer_weights[0].shape[0]
        
        distribution_map_weights = []
        distribution_map_labels = model._nodes_label.reshape(n_rows, n_cols)
        #print(model._competitive_layer_weights)
        inversed_competitive_layer_weights = scaler.inverse_transform(model._competitive_layer_weights)
        #print(inversed_competitive_layer_weights)
        scaled_competitive_layer_weights = inversed_competitive_layer_weights / np.amax(inversed_competitive_layer_weights, axis = 0)

        tmp_label = model._nodes_label.copy().reshape(n_rows, n_cols)
        
        distribution_maps_data = []
        visualization_map_data = []

        headers = ['DT', 'GR', 'NPHI', 'RHOB']
        for i in range (n_features):
            tmp_weights = inversed_competitive_layer_weights[:, i].copy()
            tmp_weights[tmp_weights < 0] = 0
            tmp_weights = tmp_weights.reshape(n_rows, n_cols)
            tmp_weights = np.round(tmp_weights, 2)

            tmp_scaled_weights = scaled_competitive_layer_weights[:, i].copy()
            tmp_scaled_weights[tmp_scaled_weights < 0] = 0
            tmp_scaled_weights = tmp_scaled_weights.reshape(n_rows, n_cols)    
            tmp_scaled_weights = np.round(tmp_scaled_weights, 2)

            distribution_map = {}
            distribution_map['header'] = headers[i]
            rows = []
            for j in range (n_rows):
                row = {}
                cells = []
                for k in range (n_cols):
                    cells.append({
                        'weight': tmp_weights[j, k],
                        'scaledWeight': tmp_scaled_weights[j, k],
                        'label': int(tmp_label[j, k])
                    })
                
                row['cells'] = cells
                rows.append(row)

            distribution_map['rows'] = rows
            distribution_maps_data.append(distribution_map)

        for i in range (n_rows):
            row = {}
            cells = []
            for j in range (n_cols):
                features = []
                tmp_weights = inversed_competitive_layer_weights[i * n_cols + j, :].copy()
                tmp_weights[tmp_weights < 0] = 0
                tmp_weights = np.round(tmp_weights, 2)

                tmp_scaled_weights = scaled_competitive_layer_weights[i * n_cols + j, :].copy()
                tmp_scaled_weights[tmp_scaled_weights < 0] = 0
                tmp_scaled_weights = np.round(tmp_scaled_weights, 2)
                for k in range (n_features):
                    features.append({
                        'weight': tmp_weights[k],
                        'scaledWeight': tmp_scaled_weights[k],
                        'header': headers[k]
                    })
                cells.append({
                    'features': features,
                    'label': int(tmp_label[i][j])
                })
            row['cells'] = cells
            visualization_map_data.append(row)

        for i in range (n_features):
            tmp = inversed_competitive_layer_weights[:, i].copy()
            tmp[tmp < 0] = 0
            tmp = tmp.reshape(n_rows, n_cols)
            distribution_map_weights.append(tmp)
        distribution_map_weights = np.round(np.array(distribution_map_weights), 2)

        fitted_model = {
            # "competitiveWeights": model._competitive_layer_weights.tolist(),
            "inversedCompetitiveWeights": inversed_competitive_layer_weights.tolist(),
            "nodesLabel": model._nodes_label.tolist(),
            "distributionMapWeights": distribution_map_weights.tolist(),
            # "distribution_map_labels": distribution_map_labels.tolist(),
        }

        success_message.add("fitted_model", fitted_model)
        success_message.add("distributionMaps", distribution_maps_data)
        success_message.add("visualizationMap", visualization_map_data)
        # print(distribution_map_weights.shape)
        # print(distribution_map_labels.shape)
        return success_message()
