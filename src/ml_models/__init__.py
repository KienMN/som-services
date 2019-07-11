import os
import shutil
import importlib
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from src import config
from . import result
from . import validator


def get_component_from_module(name_component, module):
    sub_components = dir(module)
    estimator = list(filter(lambda x: x.endswith(name_component), sub_components))
    if len(estimator) > 0:
        return getattr(module, estimator[0])
    else:
        return None


def load_modules():
    """Dynamic loader
    """
    # Load all modules
    models_package_dir = os.path.join("src", "ml_models", "models")
    models_package = models_package_dir.replace("/", ".")
    module_files = os.listdir(models_package_dir)
    try:
        module_files.remove("__init__.py")
    except Exception:
        pass
    modules = {}
    for module_file in module_files:
        module_file = module_file.split(".")[0]
        modules[module_file.lower()] = importlib.import_module(models_package + "." + module_file)

    # Load component constructor for each module
    global model_constructors
    global validator_constructors
    global result_constructors

    for name, module in modules.items():
        model_constructors[name] = get_estimator_from_module(module)        
        result_constructors[name] = get_result_from_module(module)        
        validator_constructors[name] = get_validator_from_module(module)        


def names():
    global model_constructors
    return model_constructors.keys()


def build_model(model_type, params):
    global model_constructors

    # Checking 
    support_type = model_constructors.keys()
    assert (model_type in support_type), 'Expected one of value {}'.format(','.join(support_type))

    constructorFunc = model_constructors[model_type]

    estimator = Pipeline([
        ('minmax-scaler', MinMaxScaler(feature_range=(-1, 1))),
        # ('standard-scaler', StandardScaler()),
        ("model", constructorFunc(**params))
    ])

    return estimator

def save_model(model, save_path='./'):
    """Implement function save model to disk
    Parameters:
    -----------
    model: object
    save_path: str

    Returns:
    --------
    """
    if os.path.exists(save_path):
        if os.path.isfile(save_path):
            os.remove(save_path)
        else:
            shutil.rmtree(save_path)
    joblib.dump(model, save_path)
    return None


def load_model(model_path):
    """Implement function load model from disk
    Parameters:
    -----------
    model_path: str, path to model storage

    Returns:
    --------
    estimator: object
    """
    with open(model_path, 'rb') as model_f:
        estimator = joblib.load(model_path)
    return estimator


def get_validator(type_validator):
    global validator_constructors
    validator_def = validator_constructors[type_validator]
    return validator_def


def get_result(model_id):
    global result_constructors
    model_path = os.path.join(config.model_dir, model_id+'.joblib')
    model = load_model(model_path)
    for (name, model_def) in model_constructors.items():
        if isinstance(model.named_steps["model"], model_def):
            return result_constructors[name]
    raise ValueError("{} not exist".format(type(model)))


model_constructors = {}
validator_constructors = {}
result_constructors = {}

get_result_from_module = partial(get_component_from_module, "Result")
get_estimator_from_module = partial(get_component_from_module, "Estimator")
get_validator_from_module = partial(get_component_from_module, "Validator")

load_modules()
