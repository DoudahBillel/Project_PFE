class BaseModel(object):
    def __init__(self, mode, config, model_dir):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.keras_model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.keras_model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.keras_model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load_weights(self, checkpoint_path):
        raise NotImplementedError

    def build(self, mode, config):
        raise NotImplementedError    
