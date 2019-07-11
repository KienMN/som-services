import secrets
from src import model_helper
from src.ml_models import config
from src.ml_models import result


@model_helper.parse_body_request
def som(parameters):
    if parameters.get('model_id'):
        model_id = parameters.get('model_id')
    else:
        model_id = secrets.token_urlsafe(16)
    try:
        model_helper.model_create(model_id, "som", parameters)
    except Exception as err:
        config.logger.error(str(err))
        err_message = result.ErrorResult()
        return err_message()
    else:
        success_message = result.SuccessResult()
        success_message.add("model_id", model_id)
        return success_message()