class BaseResult(object):
    def __init__(self, message, status):
        self._message = message
        self._status = status
        self._extend = {}

    def __call__(self):
        fixed = {"message": self._message}
        return { **fixed, **self._extend }, self._status

    def message(self, message):
        self._message = message
        return self

    def status(self, status):
        self._status = status
        return self

    def add(self, name, value):
        self._extend[name] = value


class SuccessResult(BaseResult):
    def __init__(self):
        super(SuccessResult, self).__init__("successfully", 201)
    

class ErrorResult(BaseResult):
    def __init__(self):
        super(ErrorResult, self).__init__("failure", 400)

