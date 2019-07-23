from src.ml_models import result
from src.ml_models import validator
from .Som_Estimator import WiEstimator

SomParameters = [
    {
        "name": "n_rows",
        "type": int
    },
    {
        "name": "n_cols",
        "type": int
    },
]

class SomValidator(validator.BaseValidator):
    def __init__(self, **kwargs):
        self.props = SomParameters
        validator.BaseValidator.doInit(self, **kwargs)

    def __call__(self):
        return_params = super(SomValidator, self).__call__()
        # additional validation goes here
        return return_params


class SomResult(result.SuccessResult):
    def __init__(self, accuracy, *args, **kwargs):
        super(SomResult, self).__init__()
        self.add("accuracy", accuracy)


class SomEstimator(WiEstimator):
    pass