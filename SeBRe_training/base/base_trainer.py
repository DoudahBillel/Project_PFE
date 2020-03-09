class BaseTrain(object):
    def __init__(self, model, train_data, config):
        self.model = model
        self.data = data
        self.config = config

    def train(self):
        raise NotImplementedError
